#!/usr/bin/env python3
"""Background job registry for quantization scripts.

Layout:
    jobs/<id>/
      meta.json
      script.py
      stdout.log
      stderr.log
      exit_code
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import secrets
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths
from io_utils import atomic_write_text

# Job ids are minted only by new_job_id; anything else reaching a JOBS_ROOT / job_id
# join is caller-supplied and must be rejected before it can traverse out of the jobs tree.
JOB_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$")


@dataclass
class JobMeta:
    job_id: str
    method_id: str
    model_id: str
    venv: str
    script_path: str
    output_dir: str
    pid: int
    started_at: str
    finished_at: str | None = None
    exit_code: int | None = None
    status: str = "running"  # running | completed | failed | killed | timeout
    pgid: int | None = None
    parent_job_id: str | None = None
    attempt: int = 1
    fix_note: str | None = None
    tune_iter: int = 0
    hyperparameters: dict | None = None
    metrics: dict | None = None
    inference: dict | None = None
    verification_status: str | None = None  # passed | failed
    verification_error: str | None = None
    terminal_reason: str | None = None
    termination_confirmed: bool | None = None
    manifest_path: str | None = None
    execution_mode: str = "host"
    overlay_path: str | None = None
    overlay_sha256: str | None = None
    worktree_path: str | None = None
    benchmark_status: str | None = None  # passed | failed
    benchmark_error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{secrets.token_hex(3)}"


def valid_job_id(job_id: str) -> bool:
    """True if job_id matches the minted format (safe to use in a JOBS_ROOT join)."""
    return isinstance(job_id, str) and bool(JOB_ID_RE.match(job_id))


def require_valid_job_id(job_id: str) -> None:
    if not valid_job_id(job_id):
        raise FileNotFoundError(f"No such job: {job_id!r}")


def jobs_root() -> Path:
    return paths.JOBS_ROOT


def job_dir(job_id: str) -> Path:
    require_valid_job_id(job_id)
    return jobs_root() / job_id


def _pid_alive(pid: int) -> bool:
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, just not ours


def _job_alive(meta: JobMeta) -> bool:
    """True if the job's process is still running AND is the same one we launched."""
    if not _pid_alive(meta.pid):
        return False
    if meta.pgid is None:
        return True
    try:
        return os.getpgid(meta.pid) == meta.pgid
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _cleanup_meta_worktree(meta: JobMeta) -> None:
    if not meta.worktree_path:
        return
    from overlay import cleanup_overlay_worktree

    worktree = Path(meta.worktree_path)
    repo = paths.VENV_ROOT / meta.method_id / "repo"
    cleanup_overlay_worktree(worktree, repo if repo.exists() else None)


def read_meta(job_id: str) -> JobMeta:
    require_valid_job_id(job_id)
    path = job_dir(job_id) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"No such job: {job_id}")
    with path.open() as f:
        payload = json.load(f)
    return JobMeta(**payload)


def write_meta(meta: JobMeta) -> None:
    """Atomically persist metadata under a cross-process per-job lock."""
    directory = job_dir(meta.job_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "meta.json"
    lock_path = directory / "meta.lock"
    with lock_path.open("a+") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            atomic_write_text(path, meta.to_json())
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def refresh_status(job_id: str) -> JobMeta:
    """Re-evaluate whether the job is still alive and persist the result."""
    meta = read_meta(job_id)
    if meta.status in {"completed", "failed", "killed", "timeout", "termination_failed"}:
        _cleanup_meta_worktree(meta)
        return meta

    exit_sentinel = job_dir(job_id) / "exit_code"
    if exit_sentinel.exists():
        raw = exit_sentinel.read_text().strip()
        try:
            code = int(raw)
        except ValueError:
            return meta
        meta.exit_code = code
        meta.status = "completed" if code == 0 else "failed"
        meta.finished_at = datetime.now(timezone.utc).isoformat()
        _cleanup_meta_worktree(meta)
        write_meta(meta)
        return meta

    if not _job_alive(meta):
        meta.status = "killed"
        meta.finished_at = datetime.now(timezone.utc).isoformat()
        _cleanup_meta_worktree(meta)
        write_meta(meta)
    return meta


def _terminate_process_group(meta: JobMeta, grace_s: float = 3.0) -> bool:
    """TERM, wait, KILL, then confirm the recorded process group is gone."""
    try:
        live_pgid = os.getpgid(meta.pid)
    except (ProcessLookupError, PermissionError):
        return not _pid_alive(meta.pid)
    if meta.pgid is not None and live_pgid != meta.pgid:
        return False

    try:
        os.killpg(live_pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return not _job_alive(meta)

    deadline = time.monotonic() + max(grace_s, 0.0)
    while time.monotonic() < deadline:
        if not _job_alive(meta):
            return True
        time.sleep(0.05)

    try:
        if meta.pgid is None or os.getpgid(meta.pid) == meta.pgid:
            os.killpg(live_pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not _job_alive(meta):
            return True
        time.sleep(0.05)
    return not _job_alive(meta)


def _tail_file(path: Path, n_lines: int, max_bytes: int = 2_000_000) -> str:
    """Read at most ``max_bytes`` from the end of a potentially huge log."""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        pos = end
        chunks: list[bytes] = []
        newlines = 0
        while pos > 0 and newlines <= n_lines and end - pos < max_bytes:
            size = min(8192, pos, max_bytes - (end - pos))
            if size <= 0:
                break
            pos -= size
            f.seek(pos)
            chunk = f.read(size)
            chunks.append(chunk)
            newlines += chunk.count(b"\n")
    data = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
    return "\n".join(data.splitlines()[-n_lines:])


def tail(job_id: str, n_lines: int = 80) -> dict[str, str]:
    directory = job_dir(job_id)
    if not directory.exists():
        raise FileNotFoundError(f"No such job: {job_id}")
    n_lines = max(1, min(int(n_lines), 10_000))
    result = {}
    for name in ("stdout.log", "stderr.log"):
        path = directory / name
        if not path.exists():
            result[name] = ""
            continue
        result[name] = _tail_file(path, n_lines)
    return result


def list_jobs() -> list[JobMeta]:
    root = jobs_root()
    if not root.exists():
        return []
    metas = []
    for directory in sorted(root.iterdir(), reverse=True):
        if not directory.is_dir() or not (directory / "meta.json").exists():
            continue
        try:
            metas.append(refresh_status(directory.name))
        except Exception:  # noqa: BLE001 — skip corrupt job dirs
            continue
    return metas


def kill(job_id: str) -> JobMeta:
    meta = read_meta(job_id)
    if meta.status not in {"running"}:
        return meta
    confirmed = _terminate_process_group(meta)
    meta.status = "killed" if confirmed else "termination_failed"
    meta.terminal_reason = "user requested termination"
    meta.termination_confirmed = confirmed
    meta.finished_at = datetime.now(timezone.utc).isoformat()
    _cleanup_meta_worktree(meta)
    write_meta(meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect background quantization jobs")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List jobs")
    status_p = sub.add_parser("status", help="Show one job's metadata")
    status_p.add_argument("job_id")
    logs_p = sub.add_parser("logs", help="Tail stdout/stderr")
    logs_p.add_argument("job_id")
    logs_p.add_argument("-n", type=int, default=80)
    kill_p = sub.add_parser("kill", help="Terminate a running job")
    kill_p.add_argument("job_id")
    args = parser.parse_args()

    try:
        if args.command == "list":
            metas = list_jobs()
            if not metas:
                print("No jobs.")
                return 0
            for meta in metas:
                print(
                    f"{meta.job_id}  {meta.status:<10} {meta.method_id:<8} "
                    f"{meta.model_id}  pid={meta.pid}"
                )
            return 0
        if args.command == "status":
            print(refresh_status(args.job_id).to_json())
            return 0
        if args.command == "logs":
            logs = tail(args.job_id, n_lines=args.n)
            print("=== stdout ===")
            print(logs["stdout.log"])
            print("=== stderr ===")
            print(logs["stderr.log"])
            return 0
        print(kill(args.job_id).to_json())
        return 0
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
