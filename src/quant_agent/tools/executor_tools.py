from __future__ import annotations

import json
from dataclasses import asdict

from langchain_core.tools import tool

from .. import executor
from .script_generator import render


@tool
def execute_quantization(
    method_id: str,
    model_id: str,
    options: dict | None = None,
) -> str:
    """Launch the quantization script in the background on this EC2 box and return a job_id.

    The script runs in the method-specific venv (set up by scripts/bootstrap_ec2.sh)
    and survives SSH disconnects. The tool returns immediately — use `check_job(job_id)`
    to poll status and `tail_job_logs(job_id)` to read recent logs.

    Args:
        method_id: Catalog id from recommend_quantization (e.g. 'awq', 'gptq', 'hqq', 'bnb_nf4').
        model_id:  HuggingFace model id being ported.
        options:   Optional overrides forwarded to the template (bits, group_size, output_dir, ...).

    Only call this after the user has approved the recommended method.
    """
    opts = dict(options or {})
    output_dir = opts.get("output_dir") or f"./quantized/{method_id}-{model_id.replace('/', '__')}"
    opts["output_dir"] = output_dir

    code = render(method_id, model_id, options=opts)
    try:
        meta = executor.launch(method_id, model_id, code, output_dir)
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
