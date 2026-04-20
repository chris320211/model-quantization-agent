"""Top-level flow: Research → user selection → Adapt → (optional) execute.

The orchestrator owns the stdin/stderr handshake so the agents themselves stay pure —
research returns a ResearchReport, adapt returns (path, code), and only the orchestrator
knows about user prompts and job launch.
"""
from __future__ import annotations

import json
import sys

import typer

from . import adapt_agent, executor, fix_agent, research_agent
from .executor import JobMeta
from .schemas import MethodCandidate, ResearchReport
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
        if report.gpu_arch:
            cc = f" sm_{int(report.compute_capability * 10)}" if report.compute_capability else ""
            parts.append(f"{report.gpu_arch}{cc}")
        lines.append(f"Hardware: {' / '.join(parts)}")
    lines.append("")
    lines.append(f"Considered methods ({len(report.considered)}):")
    for c in report.considered:
        tag = "INCLUDE" if c.verdict == "include" else "reject "
        lines.append(f"  [{tag}] {c.id:<24} {c.reason}")
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


def _supervise(
    initial_job_id: str,
    chosen: MethodCandidate,
    model_id: str,
    max_repairs: int,
) -> tuple[JobMeta, list[JobMeta]]:
    """Wait on the initial job; on runtime failure, invoke fix_agent up to max_repairs times.

    Returns (final_meta, chain) where chain is the ordered list of all job metas
    involved (initial + any relaunches). ``final_meta`` is the last meta seen —
    completed, failed after exhausting repairs, killed, or non-retryable.
    """
    meta = executor.wait_for_job(initial_job_id)
    chain: list[JobMeta] = [meta]

    for attempt in range(1, max_repairs + 1):
        if meta.status == "completed":
            return meta, chain
        if meta.status != "failed":
            # killed or any other non-failed terminal state: don't retry
            return meta, chain

        typer.echo(
            f"\n[fix] attempt {attempt}/{max_repairs}: job {meta.job_id} failed "
            f"(exit {meta.exit_code}). Invoking fix agent...",
            err=True,
        )
        try:
            new_job_id = fix_agent.run(
                job_id=meta.job_id,
                method=chosen,
                model_id=model_id,
                attempt=attempt,
                max_attempts=max_repairs,
            )
        except Exception as e:  # noqa: BLE001 — any fix failure stops the loop
            typer.echo(f"[fix] fix agent errored: {e}", err=True)
            return meta, chain

        if new_job_id is None:
            typer.echo(
                "[fix] fix agent classified failure as non-retryable; stopping.",
                err=True,
            )
            return meta, chain

        meta = executor.wait_for_job(new_job_id)
        chain.append(meta)

    return meta, chain


def _format_supervise_trail(chain: list[JobMeta]) -> str:
    lines: list[str] = []
    for m in chain:
        tag = f"attempt {m.attempt}"
        status_line = f"Job {m.job_id} ({tag}): {m.status}"
        if m.exit_code is not None:
            status_line += f" (exit {m.exit_code})"
        lines.append(status_line)
    final = chain[-1]
    lines.append(
        f"Final status: {final.status}"
        + (f" (exit {final.exit_code})" if final.exit_code is not None else "")
    )
    lines.append(
        f"Monitor: quant-agent jobs logs {final.job_id} -n 200   "
        f"|   quant-agent jobs status {final.job_id}"
    )
    return "\n".join(lines)


def run(user_input: str, dry: bool = False, max_repairs: int = 3) -> str:
    report = research_agent.run(user_input)
    typer.echo(format_report(report), err=True)

    idx = prompt_selection(len(report.methods))
    if idx is None:
        return "aborted"

    # Try the user's pick first, then fall back through the remaining candidates
    # in report order if Adapt fails (clone/install/write errors). User chose
    # auto-fallback over abort or re-prompt.
    order = [idx - 1] + [i for i in range(len(report.methods)) if i != idx - 1]

    last_error: Exception | None = None
    for pos, i in enumerate(order):
        chosen = report.methods[i]
        if pos > 0:
            typer.echo(
                f"\n[fallback] Previous candidate failed: {last_error}. "
                f"Trying next candidate: {chosen.name} ({chosen.id}).",
                err=True,
            )
        try:
            script_path, script_code = adapt_agent.run(
                model_id=report.resolved_model_id,
                method=chosen,
            )
        except Exception as e:  # noqa: BLE001 — any adapt failure triggers fallback
            last_error = e
            continue

        if dry:
            return _format_handoff(script_path, None)

        job_payload = execute_quantization.invoke(
            {
                "method_id": chosen.id,
                "model_id": report.resolved_model_id,
                "script_code": script_code,
            }
        )

        handoff = _format_handoff(script_path, job_payload)
        try:
            job = json.loads(job_payload)
        except json.JSONDecodeError:
            return handoff
        if "error" in job or "job_id" not in job:
            return handoff

        if max_repairs <= 0:
            return handoff

        final, chain = _supervise(
            initial_job_id=job["job_id"],
            chosen=chosen,
            model_id=report.resolved_model_id,
            max_repairs=max_repairs,
        )
        return handoff + "\n\n" + _format_supervise_trail(chain)

    return f"All {len(order)} candidates failed. Last error: {last_error}"
