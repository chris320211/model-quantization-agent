from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path

from langchain_core.tools import tool

from .. import executor


@tool
def execute_quantization(
    method_id: str,
    model_id: str,
    script_code: str,
    options: dict | None = None,
) -> str:
    """Launch the quantization script in the background on this EC2 box and return a job_id.

    The script runs in the method-specific venv at .venvs/<method_id>/ (built by
    the Adapt agent via install_method_venv) and survives SSH disconnects. The tool
    returns immediately — use `check_job(job_id)` to poll status and `tail_job_logs(job_id)`
    to read recent logs.

    Args:
        method_id:   Catalog id (e.g. 'awq', 'gptq', 'hqq', 'bnb_nf4').
        model_id:    HuggingFace model id being ported.
        script_code: Full Python source to run (produced by the Adapt agent).
        options:     Optional dict; currently used only for output_dir override.
    """
    opts = dict(options or {})
    output_dir = opts.get("output_dir") or executor.default_output_dir(method_id, model_id)

    try:
        meta = executor.launch(method_id, model_id, script_code, output_dir)
    except (ValueError, RuntimeError) as e:
        return json.dumps({"error": str(e)})
    return json.dumps(
        {
            "job_id": meta.job_id,
            "pid": meta.pid,
            "status": meta.status,
            "method_id": meta.method_id,
            "model_id": meta.model_id,
            "output_dir": meta.output_dir,
            "script_path": meta.script_path,
            "message": (
                f"Job {meta.job_id} started. Use check_job('{meta.job_id}') to poll, "
                f"tail_job_logs('{meta.job_id}') for logs."
            ),
        },
        indent=2,
    )


@tool
def check_job(job_id: str) -> str:
    """Return current status of a quantization job (running | completed | failed | killed)."""
    try:
        meta = executor.refresh_status(job_id)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    return json.dumps(asdict(meta), indent=2)


@tool
def list_jobs() -> str:
    """List all quantization jobs on this box, newest first, with their current status."""
    metas = executor.list_jobs()
    return json.dumps([asdict(m) for m in metas], indent=2)


@tool
def tail_job_logs(job_id: str, n_lines: int = 80) -> str:
    """Return the last n lines of stdout and stderr for a job."""
    try:
        logs = executor.tail(job_id, n_lines=n_lines)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    return json.dumps(logs, indent=2)


@tool
def kill_job(job_id: str) -> str:
    """Terminate a running quantization job (SIGTERM to the process group)."""
    try:
        meta = executor.kill(job_id)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    return json.dumps(asdict(meta), indent=2)


@tool
def read_job_logs(job_id: str, n_lines: int = 200) -> str:
    """Return the tail of stdout + stderr for a failed job so the Fix agent can diagnose.

    Same data as `tail_job_logs`, scoped with a larger default tail because fix
    agents typically need more surrounding context than a status check.
    """
    try:
        logs = executor.tail(job_id, n_lines=n_lines)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    return json.dumps(
        {"stdout": logs.get("stdout.log", ""), "stderr": logs.get("stderr.log", "")},
        indent=2,
    )


_SCRIPT_READ_MAX_BYTES = 20_000


@tool
def read_script(job_id: str, max_bytes: int = _SCRIPT_READ_MAX_BYTES) -> str:
    """Return the current contents of a job's saved script (jobs/<id>/script.py).

    Call this BEFORE edit_script so you know the exact text to replace. Truncates
    at ``max_bytes`` and reports the truncation so you can re-read with a larger
    cap if needed.
    """
    if not executor.valid_job_id(job_id):
        return json.dumps({"status": "error", "error": f"invalid job_id: {job_id!r}"})
    script = executor.JOBS_ROOT / job_id / "script.py"
    if not script.exists():
        return json.dumps({"status": "error", "error": f"no such script: {script}"})
    data = script.read_bytes()
    truncated = len(data) > max_bytes
    text = data[:max_bytes].decode("utf-8", errors="replace")
    return json.dumps(
        {
            "status": "ok",
            "path": str(script),
            "size_bytes": len(data),
            "truncated": truncated,
            "content": text,
        },
        indent=2,
    )


@tool
def edit_script(job_id: str, old: str, new: str) -> str:
    """Apply a single str.replace edit to the failed job's saved script (jobs/<id>/script.py).

    `old` must appear exactly once in the file. After the edit, the new content is
    validated with ast.parse; syntax errors roll back the change. Returns JSON with
    status + the number of bytes written on success.
    """
    if not executor.valid_job_id(job_id):
        return json.dumps({"status": "error", "error": f"invalid job_id: {job_id!r}"})
    script = executor.JOBS_ROOT / job_id / "script.py"
    if not script.exists():
        return json.dumps({"status": "error", "error": f"no such script: {script}"})

    text = script.read_text()
    count = text.count(old)
    if count == 0:
        return json.dumps(
            {"status": "error", "error": "old string not found in script.py"}
        )
    if count > 1:
        return json.dumps(
            {
                "status": "error",
                "error": f"old string matches {count} times; make it unique",
            }
        )

    new_text = text.replace(old, new, 1)
    try:
        ast.parse(new_text)
    except SyntaxError as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"edit produced syntax error: {e.msg} at line {e.lineno}",
            }
        )
    script.write_text(new_text)
    return json.dumps(
        {"status": "ok", "path": str(script), "bytes": len(new_text)}, indent=2
    )


@tool
def relaunch_job(job_id: str, fix_description: str) -> str:
    """Re-launch a failed job's (possibly edited) script under the same method venv.

    ``fix_description`` is a one-sentence summary of the fix you just applied
    (e.g. "pinned transformers==4.46.3 in the awq venv" or "changed --model_path
    flag to --model"). It is recorded on the new job's meta so that, if this
    relaunch also fails, the next repair attempt sees what was already tried and
    does not repeat it.

    Reads jobs/<job_id>/script.py and the parent job's meta.json (for method_id,
    model_id, output_dir), then spawns a new background job whose meta.parent_job_id
    points at the failed job. Returns the new job_id — call this LAST; it terminates
    the Fix agent loop.
    """
    try:
        parent = executor.refresh_status(job_id)
    except FileNotFoundError as e:
        return json.dumps({"status": "error", "error": str(e)})

    script_path = Path(parent.script_path)
    if not script_path.exists():
        script_path = executor.JOBS_ROOT / job_id / "script.py"
    if not script_path.exists():
        return json.dumps(
            {"status": "error", "error": f"script not found for job {job_id}"}
        )

    script_code = script_path.read_text()
    try:
        meta = executor.launch(
            method_id=parent.method_id,
            model_id=parent.model_id,
            script_code=script_code,
            output_dir=parent.output_dir,
            parent_job_id=job_id,
            attempt=parent.attempt + 1,
            fix_note=fix_description.strip() or None,
        )
    except (ValueError, RuntimeError) as e:
        return json.dumps({"status": "error", "error": str(e)})

    return json.dumps(
        {
            "status": "ok",
            "new_job_id": meta.job_id,
            "parent_job_id": job_id,
            "pid": meta.pid,
            "attempt": meta.attempt,
            "fix_note": meta.fix_note,
        },
        indent=2,
    )
