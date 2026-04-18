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
import os
import secrets
import signal
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import REPO_ROOT, load_settings

JOBS_ROOT = REPO_ROOT / "jobs"
VENV_ROOT = REPO_ROOT / ".venvs"

# Map catalog method_id -> venv directory name (matches scripts/bootstrap_ec2.sh).
# A value of None means the method has no pip-installable package; dry-import validation
# is skipped (ast.parse only) and end-to-end launch is unsupported (the generated script
# must be run manually from inside the method's cloned repo).
METHOD_TO_VENV: dict[str, str | None] = {
    "awq": "awq",
    "gptq": "gptq",
    "hqq": "hqq",
    "bnb_nf4": "bnb",
    "bnb_llm_int8": "bnb",
    "flatquant": None,
}


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
    status: str = "running"  # running | completed | failed | killed

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{secrets.token_hex(3)}"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, just not ours


def venv_python(venv_name: str) -> Path:
    """Path to the python binary inside a method-specific venv."""
    return VENV_ROOT / venv_name / "bin" / "python"


def launch(method_id: str, model_id: str, script_code: str, output_dir: str) -> JobMeta:
    """Spawn the quantization script in its method venv, detached from the agent."""
    venv = METHOD_TO_VENV.get(method_id)
    if venv is None:
        raise ValueError(
            f"No venv mapping for method '{method_id}'. Run scripts/bootstrap_ec2.sh first "
            f"or add '{method_id}' to METHOD_TO_VENV."
        )
    py = venv_python(venv)
    if not py.exists():
        raise RuntimeError(
            f"Venv python not found at {py}. Run scripts/bootstrap_ec2.sh {venv} first."
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
    # child process has been reaped.
    wrapper = (
        f'"{py}" "{script_path}"; echo $? > "{exit_sentinel}"'
    )
    proc = subprocess.Popen(
        ["bash", "-c", wrapper],
        stdout=stdout,
        stderr=stderr,
        stdin=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
        start_new_session=True,  # setsid: survives SSH disconnect (SIGHUP)
        env={**os.environ},
    )

    meta = JobMeta(
        job_id=job_id,
        method_id=method_id,
        model_id=model_id,
        venv=venv,
        script_path=str(script_path),
        output_dir=output_dir,
        pid=proc.pid,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    (job_dir / "meta.json").write_text(meta.to_json())
    return meta


def _read_meta(job_id: str) -> JobMeta:
    path = JOBS_ROOT / job_id / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"No such job: {job_id}")
    with path.open() as f:
        d = json.load(f)
    return JobMeta(**d)


def _write_meta(meta: JobMeta) -> None:
    (JOBS_ROOT / meta.job_id / "meta.json").write_text(meta.to_json())


def refresh_status(job_id: str) -> JobMeta:
    """Re-evaluate whether the job is still alive and persist the result."""
    meta = _read_meta(job_id)
    if meta.status in {"completed", "failed", "killed"}:
        return meta

    exit_sentinel = JOBS_ROOT / job_id / "exit_code"
    if exit_sentinel.exists():
        code = int(exit_sentinel.read_text().strip() or "-1")
        meta.exit_code = code
        meta.status = "completed" if code == 0 else "failed"
        meta.finished_at = datetime.now(timezone.utc).isoformat()
        _write_meta(meta)
        return meta

    if not _pid_alive(meta.pid):
        # process gone but no sentinel — probably killed externally
        meta.status = "killed"
        meta.finished_at = datetime.now(timezone.utc).isoformat()
        _write_meta(meta)
    return meta


def tail(job_id: str, n_lines: int = 80) -> dict[str, str]:
    job_dir = JOBS_ROOT / job_id
    if not job_dir.exists():
        raise FileNotFoundError(f"No such job: {job_id}")
    result = {}
    for name in ("stdout.log", "stderr.log"):
        p = job_dir / name
        if not p.exists():
            result[name] = ""
            continue
        lines = p.read_text(errors="replace").splitlines()
        result[name] = "\n".join(lines[-n_lines:])
    return result


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
    try:
        os.killpg(os.getpgid(meta.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    meta.status = "killed"
    meta.finished_at = datetime.now(timezone.utc).isoformat()
    _write_meta(meta)
    return meta


# Make load_settings import reachable for any future use
_ = load_settings
