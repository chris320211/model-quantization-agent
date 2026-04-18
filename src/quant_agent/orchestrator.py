"""Top-level flow: Research → user selection → Adapt → (optional) execute.

The orchestrator owns the stdin/stderr handshake so the agents themselves stay pure —
research returns a ResearchReport, adapt returns (path, code), and only the orchestrator
knows about user prompts and job launch.
"""
from __future__ import annotations

import json
import sys

import typer

from . import adapt_agent, research_agent
from .schemas import ResearchReport
from .tools.executor_tools import execute_quantization


def format_report(report: ResearchReport) -> str:
    lines: list[str] = []
    lines.append(f"Model:    {report.resolved_model_id}")
    if report.params_b is not None:
        lines.append(f"Params:   {report.params_b}B")
    if report.instance_type or report.vram_gb:
        parts = []
        if report.instance_type:
            parts.append(report.instance_type)
        if report.vram_gb is not None:
            parts.append(f"{report.vram_gb:g} GB VRAM")
        lines.append(f"Hardware: {' / '.join(parts)}")
    lines.append("")
    lines.append("Candidate methods:")
    for i, m in enumerate(report.methods, 1):
        lines.append(
            f"  {i}. {m.name}  ({m.id}, {m.bits}-bit, ~{m.est_vram_gb:g} GB, "
            f"Q={m.quality_score}/5 S={m.speed_score}/5"
            f"{', needs calibration' if m.needs_calibration else ''})"
        )
        lines.append(f"     {m.summary}")
    lines.append("")
    lines.append(f"Tradeoffs: {report.tradeoffs}")
    return "\n".join(lines)


def prompt_selection(n: int) -> int | None:
    """Read a 1..n selection from stdin. Returns None on 'q' or EOF."""
    while True:
        typer.echo(f"\nPick 1-{n} or q to abort: ", nl=False, err=True)
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            return None
        if not line:
            return None
        raw = line.strip().lower()
        if raw in {"q", "quit", "exit"}:
            return None
        if raw.isdigit():
            i = int(raw)
            if 1 <= i <= n:
                return i
        typer.echo(f"Invalid choice: {raw!r}", err=True)


def _format_handoff(script_path: str, job_payload: str | None) -> str:
    out = [f"Script written: {script_path}"]
    if job_payload is None:
        out.append("(--dry): skipped execute_quantization. Run the script manually when ready.")
        return "\n".join(out)
    try:
        job = json.loads(job_payload)
    except json.JSONDecodeError:
        out.append(job_payload)
        return "\n".join(out)
    if "error" in job:
        out.append(f"execute_quantization failed: {job['error']}")
        return "\n".join(out)
    out.append(
        f"Job launched: {job['job_id']} (pid={job['pid']}, status={job['status']})"
    )
    out.append(
        f"Monitor: quant-agent jobs logs {job['job_id']} -n 200   "
        f"|   quant-agent jobs status {job['job_id']}"
    )
    return "\n".join(out)


def run(user_input: str, dry: bool = False) -> str:
    report = research_agent.run(user_input)
    typer.echo(format_report(report), err=True)

    idx = prompt_selection(len(report.methods))
    if idx is None:
        return "aborted"
    chosen = report.methods[idx - 1]

    script_path, script_code = adapt_agent.run(
        model_id=report.resolved_model_id,
        method=chosen,
    )

    if dry:
        return _format_handoff(script_path, None)

    job_payload = execute_quantization.invoke(
        {
            "method_id": chosen.id,
            "model_id": report.resolved_model_id,
            "script_code": script_code,
        }
    )
    return _format_handoff(script_path, job_payload)
