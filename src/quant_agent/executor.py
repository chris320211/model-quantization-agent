"""Background job registry + launcher for quantization scripts.

Jobs run under their method-specific venv via ``setsid`` so they survive SSH
disconnects. State is persisted to ``<repo>/jobs/<job_id>/`` so status can be
checked across agent invocations.

Layout:
    jobs/<id>/
      meta.json     # method, model_id, script_path, pid, started_at, ...
      script.py     # snapshot of the generated script
      stdout.log
      stderr.log
      exit_code     # written by the wrapper after the python process exits
"""
from __future__ import annotations

import json
import fcntl
import os
import re
import secrets
import shlex
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

from .config import REPO_ROOT, child_env, load_settings, require_host_execution

JOBS_ROOT = REPO_ROOT / "jobs"
VENV_ROOT = REPO_ROOT / ".venvs"

# Venvs resolve by convention: .venvs/<method_id>/bin/python. Method venvs are
# built on-demand by the Adapt agent via tools.repo_tool.install_method_venv,
# which also clones the method's repo into .venvs/<method_id>/repo/.


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
    fix_note: str | None = None  # what the fix agent changed before relaunching as this job
    tune_iter: int = 0
    hyperparameters: dict | None = None
    metrics: dict | None = None
    terminal_reason: str | None = None
    termination_confirmed: bool | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# Job ids are minted only by _new_job_id; anything else reaching a JOBS_ROOT / job_id
# join is caller-supplied (LLM tools pass job_id) and must be rejected before it can
# traverse out of the jobs tree.
_JOB_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$")


def _new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{secrets.token_hex(3)}"


def valid_job_id(job_id: str) -> bool:
    """True if job_id matches the minted format (safe to use in a JOBS_ROOT join)."""
    return isinstance(job_id, str) and bool(_JOB_ID_RE.match(job_id))


def _require_valid_job_id(job_id: str) -> None:
    if not valid_job_id(job_id):
        raise FileNotFoundError(f"No such job: {job_id!r}")


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
    """True if the job's process is still running AND is the same one we launched.

    When a pgid was recorded, require the live process's group to still match it —
    a recycled PID belonging to an unrelated process will have a different pgid, so
    we don't mistake it for the job (which would otherwise wedge status at 'running').
    """
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


def venv_python(method_id: str) -> Path:
    """Path to the python binary inside a method-specific venv (.venvs/<method_id>/bin/python)."""
    return VENV_ROOT / method_id / "bin" / "python"


def default_output_dir(method_id: str, model_id: str) -> str:
    """Canonical on-disk location for a method's quantized weights.

    Single source of truth so the generated script's save path and JobMeta.output_dir
    (used by the tune-loop pruner) never disagree.
    """
    return f"./quantized/{method_id}-{model_id.replace('/', '__')}"


def launch(
    method_id: str,
    model_id: str,
    script_code: str,
    output_dir: str,
    parent_job_id: str | None = None,
    attempt: int = 1,
    fix_note: str | None = None,
    tune_iter: int = 0,
    hyperparameters: dict | None = None,
) -> JobMeta:
    """Spawn the quantization script in its method venv, detached from the agent."""
    require_host_execution("quantization launch")
    py = venv_python(method_id)
    if not py.exists():
        raise RuntimeError(
            f"Venv python not found at {py}. The Adapt agent should have built it via "
            f"install_method_venv before reaching this point."
        )

    job_id = _new_job_id()
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    script_path = job_dir / "script.py"
    script_path.write_text(script_code)

    stdout = (job_dir / "stdout.log").open("wb")
    stderr = (job_dir / "stderr.log").open("wb")
    exit_sentinel = job_dir / "exit_code"

    # wrapper records the real exit code so status is known even after the
    # child process has been reaped. Paths go through shlex.quote so method_id
    # or job_id values never cross into shell-interpreted territory.
    wrapper = (
        f"{shlex.quote(str(py))} {shlex.quote(str(script_path))}; "
        f"echo $? > {shlex.quote(str(exit_sentinel))}"
    )
    proc = subprocess.Popen(
        ["bash", "-c", wrapper],
        stdout=stdout,
        stderr=stderr,
        stdin=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
        start_new_session=True,  # setsid: survives SSH disconnect (SIGHUP)
        # Minimal allowlisted env: the LLM-authored quantization script must never
        # inherit the parent's cloud secrets (see config.child_env). HF token only.
        env=child_env(include_hf=True),
    )

    # With start_new_session the child leads its own process group (pgid == pid).
    # Recording it lets us detect PID reuse later before signaling.
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = proc.pid

    meta = JobMeta(
        job_id=job_id,
        method_id=method_id,
        model_id=model_id,
        venv=method_id,
        script_path=str(script_path),
        output_dir=output_dir,
        pid=proc.pid,
        started_at=datetime.now(timezone.utc).isoformat(),
        pgid=pgid,
        parent_job_id=parent_job_id,
        attempt=attempt,
        fix_note=fix_note,
        tune_iter=tune_iter,
        hyperparameters=hyperparameters,
    )
    write_meta(meta)
    return meta


def _read_meta(job_id: str) -> JobMeta:
    _require_valid_job_id(job_id)
    path = JOBS_ROOT / job_id / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"No such job: {job_id}")
    with path.open() as f:
        d = json.load(f)
    return JobMeta(**d)


@contextmanager
def _meta_lock(job_id: str):
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    lock_path = job_dir / "meta.lock"
    with lock_path.open("a+") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def write_meta(meta: JobMeta) -> None:
    """Atomically persist metadata under a cross-process per-job lock."""
    path = JOBS_ROOT / meta.job_id / "meta.json"
    with _meta_lock(meta.job_id):
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(meta.to_json())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()


_write_meta = write_meta  # private compatibility alias


def refresh_status(job_id: str) -> JobMeta:
    """Re-evaluate whether the job is still alive and persist the result."""
    meta = _read_meta(job_id)
    if meta.status in {"completed", "failed", "killed", "timeout", "termination_failed"}:
        return meta

    exit_sentinel = JOBS_ROOT / job_id / "exit_code"
    if exit_sentinel.exists():
        raw = exit_sentinel.read_text().strip()
        # The wrapper creates the sentinel (via `> file`) and then writes the code,
        # so there's a brief window where it exists but is empty or partial. Treat
        # unparseable content as "still running" rather than a spurious failure.
        try:
            code = int(raw)
        except ValueError:
            return meta
        meta.exit_code = code
        meta.status = "completed" if code == 0 else "failed"
        meta.finished_at = datetime.now(timezone.utc).isoformat()
        _write_meta(meta)
        return meta

    if not _job_alive(meta):
        # process gone but no sentinel — probably killed externally
        meta.status = "killed"
        meta.finished_at = datetime.now(timezone.utc).isoformat()
        _write_meta(meta)
    return meta


DEFAULT_MAX_WAIT_S = 6 * 3600  # a single quantization/measurement job upper bound


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


def wait_for_job(
    job_id: str,
    poll_interval: float = 2.0,
    max_wait_s: float | None = DEFAULT_MAX_WAIT_S,
) -> JobMeta:
    """Block until a job leaves the 'running' state. Returns the final meta.

    Bounded by ``max_wait_s`` (None disables the bound): a hung job that never writes
    its exit sentinel would otherwise block the orchestrator forever. On timeout the
    job's process group is terminated and the status is set to 'timeout'.
    """
    deadline = time.monotonic() + max_wait_s if max_wait_s is not None else None
    while True:
        meta = refresh_status(job_id)
        if meta.status != "running":
            return meta
        if deadline is not None and time.monotonic() >= deadline:
            confirmed = _terminate_process_group(meta)
            meta.status = "timeout" if confirmed else "termination_failed"
            meta.terminal_reason = f"exceeded max_wait_s={max_wait_s}"
            meta.termination_confirmed = confirmed
            meta.finished_at = datetime.now(timezone.utc).isoformat()
            _write_meta(meta)
            return meta
        time.sleep(poll_interval)


# A line that looks like the root cause of a Python failure: "XxxError: ...",
# "SomeException", "error: ...", or a loud "ERROR ..." log line.
_ERROR_LINE_RE = re.compile(r"[A-Za-z_.]*(?:Error|Exception)\b|\berror:|\bERROR\b")


def error_signature(job_id: str) -> str | None:
    """Best-effort root-cause line from a job's logs, for cross-attempt comparison.

    Scans BOTH streams (stderr first) for the LAST line that looks like a Python
    exception or error message — for a traceback that's the final ``XxxError:
    message`` line. An error-pattern match in either stream beats any fallback:
    stderr often ends in run-varying progress noise (tqdm/HF-hub download bars)
    while the real error was print()ed to stdout, and taking the noise line would
    make two identical failures look different. Only when neither stream has an
    error-looking line does it fall back to the last non-empty line (stderr, then
    stdout). Returns None when the job has no logs on disk. Two failed jobs with
    equal signatures are treated by the supervisor as "the fix changed nothing".
    """
    try:
        logs = tail(job_id, n_lines=120)
    except FileNotFoundError:
        return None
    streams: list[list[str]] = []
    for name in ("stderr.log", "stdout.log"):
        lines = [ln.strip() for ln in logs.get(name, "").splitlines() if ln.strip()]
        streams.append(lines)
        for ln in reversed(lines):
            if _ERROR_LINE_RE.search(ln):
                return ln
    for lines in streams:
        if lines:
            return lines[-1]
    return None


def tail(job_id: str, n_lines: int = 80) -> dict[str, str]:
    _require_valid_job_id(job_id)
    job_dir = JOBS_ROOT / job_id
    if not job_dir.exists():
        raise FileNotFoundError(f"No such job: {job_id}")
    n_lines = max(1, min(int(n_lines), 10_000))
    result = {}
    for name in ("stdout.log", "stderr.log"):
        p = job_dir / name
        if not p.exists():
            result[name] = ""
            continue
        result[name] = _tail_file(p, n_lines)
    return result


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


def list_jobs() -> list[JobMeta]:
    if not JOBS_ROOT.exists():
        return []
    metas = []
    for d in sorted(JOBS_ROOT.iterdir(), reverse=True):
        if not d.is_dir() or not (d / "meta.json").exists():
            continue
        try:
            metas.append(refresh_status(d.name))
        except Exception:  # noqa: BLE001 — skip corrupt job dirs
            continue
    return metas


def kill(job_id: str) -> JobMeta:
    meta = _read_meta(job_id)
    if meta.status not in {"running"}:
        return meta
    confirmed = _terminate_process_group(meta)
    meta.status = "killed" if confirmed else "termination_failed"
    meta.terminal_reason = "user requested termination"
    meta.termination_confirmed = confirmed
    meta.finished_at = datetime.now(timezone.utc).isoformat()
    _write_meta(meta)
    return meta


# Make load_settings import reachable for any future use
_ = load_settings
