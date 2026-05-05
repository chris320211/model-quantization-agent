"""Top-level flow: Research → user selection → Adapt → execute → (optional) tune loop.

The orchestrator owns the stdin/stderr handshake so the agents themselves stay pure —
research returns a ResearchReport, adapt returns (path, code), tune_agent returns a
single decision, and only the orchestrator knows about user prompts, job launch, and
disk pruning between iterations.

Tune loop (when ``--tune`` is set):
  1. After the baseline iteration succeeds, run the measurement script to capture
     prefill / decode latency, peak VRAM, and WikiText-2 perplexity.
  2. Optionally measure the fp16 reference (cached per (model, instance)) so the
     final report can show "did we beat fp16".
  3. Repeatedly: ask tune_agent for the next config → adapt re-emits the script
     with that config baked in → executor.launch + supervise (fix_agent stays
     nested) → measurement → record in pareto history + tune_history.jsonl.
  4. Terminate on stagnation (``stagnate_after`` consecutive non-improvements),
     when tune_agent returns Stop, or when ``max_tune_iter`` is reached.
  5. Disk policy: keep the latest iteration + the running Pareto-best; prune
     others' jobs/<id>/ artifacts only AFTER metrics persist.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys

import typer

from . import adapt_agent, baseline, executor, fix_agent, research_agent
from .executor import JOBS_ROOT, JobMeta
from .hyperparam_inference import default_config, infer_ranges
from .measurement import metrics_summary, run_measurement
from .pareto import Metrics, best_so_far, detect_stagnation, is_pareto_improvement
from .schemas import MethodCandidate, ResearchReport
from .tools.executor_tools import execute_quantization
from . import tune_agent, tune_history

log = logging.getLogger(__name__)


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


def _prompt_yes_no(question: str, default_no: bool = True) -> bool:
    """Read y/N from stdin. EOF or empty answer → default."""
    suffix = " [y/N]: " if default_no else " [Y/n]: "
    typer.echo(question + suffix, nl=False, err=True)
    try:
        line = sys.stdin.readline()
    except KeyboardInterrupt:
        return False
    if not line:
        return not default_no
    raw = line.strip().lower()
    if raw in {"y", "yes"}:
        return True
    if raw in {"n", "no"}:
        return False
    return not default_no


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


def _run_adapt_with_retry(
    model_id: str,
    method: MethodCandidate,
    max_adapt_retries: int,
    *,
    hyperparameters: dict | None = None,
    script_suffix: str | None = None,
) -> tuple[str, str]:
    """Invoke adapt_agent up to ``max_adapt_retries`` times on the same method.

    On each retry the previous error is fed back into the agent's user message so
    it can adjust (different install_steps, different entry point). Raises the
    final exception if every attempt fails.
    """
    last_err: Exception | None = None
    total = max(1, max_adapt_retries)
    for attempt in range(1, total + 1):
        if attempt > 1:
            typer.echo(
                f"\n[adapt-retry] attempt {attempt}/{total} on {method.id} "
                f"(previous error: {last_err})",
                err=True,
            )
        try:
            return adapt_agent.run(
                model_id=model_id,
                method=method,
                previous_error=last_err,
                hyperparameters=hyperparameters,
                script_suffix=script_suffix,
            )
        except Exception as e:  # noqa: BLE001 — any adapt failure retries
            last_err = e
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# Measurement / tune helpers
# ---------------------------------------------------------------------------


def _measure_job(meta: JobMeta) -> Metrics | None:
    """Run the measurement script against a completed job's quantized output.

    ``meta.output_dir`` points at the on-disk model. Returns None if the
    measurement script can't run (no GPU, missing deps in the method venv).
    The caller treats None as "skip pareto comparison this iteration".
    """
    job_dir = JOBS_ROOT / meta.job_id
    venv_py = executor.venv_python(meta.method_id)
    if not venv_py.exists():
        log.warning("venv missing for measure: %s", venv_py)
        return None
    try:
        return run_measurement(
            job_dir=job_dir,
            model_path=meta.output_dir,
            venv_python=venv_py,
        )
    except Exception as e:  # noqa: BLE001 — measurement is best-effort
        log.warning("measurement failed for %s: %s", meta.job_id, e)
        return None


def _persist_metrics(meta: JobMeta, metrics: Metrics, hyperparameters: dict | None) -> None:
    """Write metrics into the JobMeta on disk so jobs cli can show them."""
    meta.metrics = metrics.to_dict()
    if hyperparameters is not None:
        meta.hyperparameters = hyperparameters
    (JOBS_ROOT / meta.job_id / "meta.json").write_text(meta.to_json())


def _prune_iteration(meta: JobMeta) -> None:
    """Remove a tune iteration's job artifacts. Caller enforces the keep-set policy."""
    job_dir = JOBS_ROOT / meta.job_id
    if not job_dir.exists():
        return
    try:
        shutil.rmtree(job_dir)
    except OSError as e:
        log.warning("disk prune failed for %s: %s", meta.job_id, e)
    out_dir = meta.output_dir
    try:
        if out_dir and out_dir.startswith("./quantized/"):
            from pathlib import Path
            p = Path(out_dir)
            if p.exists():
                shutil.rmtree(p)
    except OSError as e:
        log.warning("output prune failed for %s: %s", out_dir, e)


def _format_pareto_summary(
    *,
    method: MethodCandidate,
    history: list[tune_agent.IterationRecord],
    fp16: Metrics | None,
    best_iter: int | None,
) -> str:
    lines = [""]
    lines.append(f"=== Tune summary: {method.name} ({method.id}) ===")
    if fp16 is not None:
        lines.append(f"fp16 baseline:  {metrics_summary(fp16)}")
    for i, r in enumerate(history, start=1):
        marker = "*" if best_iter == i else " "
        if r.metrics is None:
            lines.append(f" {marker} iter {i}: crashed   hp={r.hyperparameters}")
        else:
            lines.append(f" {marker} iter {i}: {metrics_summary(r.metrics)}   hp={r.hyperparameters}")
    if best_iter is not None and history[best_iter - 1].metrics is not None:
        winner = history[best_iter - 1]
        lines.append(f"Best config: iter {best_iter}  {winner.hyperparameters}")
    return "\n".join(lines)


def _tune_loop(
    *,
    method: MethodCandidate,
    model_id: str,
    instance_type: str | None,
    baseline_meta: JobMeta,
    baseline_metrics: Metrics,
    fp16: Metrics | None,
    max_tune_iter: int,
    stagnate_after: int,
    max_repairs: int,
    max_adapt_retries: int,
) -> str:
    """Drive the closed-loop tuner until termination. Returns a summary string.

    Iteration N writes its script with suffix ``_iterN`` so prior iterations'
    scripts on disk aren't overwritten. Each call to adapt_agent re-builds the
    same venv (idempotent) and re-emits the script with the new hyperparameters
    baked into the TUNE-LOCKED header.
    """
    job_dir = JOBS_ROOT / baseline_meta.job_id
    ranges = infer_ranges(method, job_dir=job_dir)

    initial_hp = method.hyperparameters or default_config(ranges)
    history: list[tune_agent.IterationRecord] = [
        tune_agent.IterationRecord(
            hyperparameters=dict(initial_hp),
            metrics=baseline_metrics,
            note="baseline iteration (chosen method defaults or research-agent picks)",
        )
    ]

    # Track which iteration each metric came from so we can prune the rest later.
    iter_metas: list[JobMeta] = [baseline_meta]
    prior_wins = tune_history.query_prior_wins(
        model_id=model_id, instance_type=instance_type, method_id=method.id
    )

    if not ranges.specs:
        typer.echo(
            f"[tune] {method.id} has no tunable ranges available; "
            "running a single iteration only.",
            err=True,
        )
        return _finalize_loop(history, iter_metas, method=method, fp16=fp16,
                              model_id=model_id, instance_type=instance_type)

    for iteration in range(2, max_tune_iter + 1):
        metrics_history = [r.metrics for r in history if r.metrics is not None]
        running_best = best_so_far(metrics_history)

        decision = tune_agent.propose(
            method=method,
            ranges=ranges,
            history=history,
            best_so_far=running_best,
            fp16_baseline=fp16,
            prior_wins=prior_wins,
        )
        if decision.decision == "stop":
            typer.echo(f"\n[tune] tune_agent stopped: {decision.reason}", err=True)
            break

        next_hp = decision.hyperparameters or {}
        typer.echo(
            f"\n[tune] iter {iteration}/{max_tune_iter}: trying {next_hp}\n"
            f"        rationale: {decision.rationale}",
            err=True,
        )

        # Adapt with new config; iter suffix keeps prior scripts on disk.
        try:
            script_path, script_code = _run_adapt_with_retry(
                model_id=model_id,
                method=method,
                max_adapt_retries=max_adapt_retries,
                hyperparameters=next_hp,
                script_suffix=f"iter{iteration}",
            )
        except Exception as e:  # noqa: BLE001
            typer.echo(f"[tune] iter {iteration} adapt failed: {e}", err=True)
            history.append(tune_agent.IterationRecord(
                hyperparameters=next_hp, metrics=None, note=f"adapt failed: {e}",
            ))
            continue

        job_payload = execute_quantization.invoke({
            "method_id": method.id,
            "model_id": model_id,
            "script_code": script_code,
        })
        try:
            job = json.loads(job_payload)
        except json.JSONDecodeError:
            history.append(tune_agent.IterationRecord(
                hyperparameters=next_hp, metrics=None,
                note=f"executor returned non-JSON: {job_payload[:200]}",
            ))
            continue
        if "error" in job or "job_id" not in job:
            history.append(tune_agent.IterationRecord(
                hyperparameters=next_hp, metrics=None,
                note=f"launch failed: {job.get('error', 'unknown')}",
            ))
            continue

        # Mark the JobMeta as a tune iteration before we wait, so jobs cli reflects it.
        iter_meta = executor.refresh_status(job["job_id"])
        iter_meta.tune_iter = iteration
        iter_meta.hyperparameters = dict(next_hp)
        (JOBS_ROOT / iter_meta.job_id / "meta.json").write_text(iter_meta.to_json())

        final, chain = _supervise(
            initial_job_id=job["job_id"],
            chosen=method,
            model_id=model_id,
            max_repairs=max_repairs,
        )
        if final.status != "completed":
            typer.echo(
                f"[tune] iter {iteration} failed (status={final.status}); "
                "recording crash and continuing.",
                err=True,
            )
            history.append(tune_agent.IterationRecord(
                hyperparameters=next_hp, metrics=None,
                note=f"job ended {final.status} after {len(chain)} attempts",
            ))
            iter_metas.append(final)
            _prune_intermediate_jobs(history, iter_metas)
            continue

        metrics = _measure_job(final)
        if metrics is None:
            typer.echo(f"[tune] iter {iteration} measurement failed", err=True)
            history.append(tune_agent.IterationRecord(
                hyperparameters=next_hp, metrics=None,
                note="measurement script returned no result",
            ))
            iter_metas.append(final)
            _prune_intermediate_jobs(history, iter_metas)
            continue

        _persist_metrics(final, metrics, next_hp)
        history.append(tune_agent.IterationRecord(
            hyperparameters=next_hp, metrics=metrics,
        ))
        iter_metas.append(final)
        typer.echo(f"[tune] iter {iteration} measured: {metrics_summary(metrics)}", err=True)

        # Persist this iteration to cross-run history before pruning so the
        # data survives even if disk-prune races with anything else.
        tune_history.append(
            model_id=model_id, instance_type=instance_type, method_id=method.id,
            hyperparameters=next_hp, metrics=metrics.to_dict(),
            note=f"iter{iteration}",
        )

        _prune_intermediate_jobs(history, iter_metas)

        metrics_history = [r.metrics for r in history if r.metrics is not None]
        if detect_stagnation(metrics_history, n=stagnate_after):
            typer.echo(
                f"[tune] stagnation detected after {stagnate_after} non-improving iterations; "
                "stopping.",
                err=True,
            )
            break

    return _finalize_loop(
        history, iter_metas, method=method, fp16=fp16,
        model_id=model_id, instance_type=instance_type,
    )


def _prune_intermediate_jobs(
    history: list[tune_agent.IterationRecord],
    iter_metas: list[JobMeta],
) -> None:
    """Keep latest + Pareto-best; delete the rest. Called AFTER metrics persist."""
    if len(iter_metas) <= 1:
        return
    metrics_only = [r.metrics for r in history if r.metrics is not None]
    best = best_so_far(metrics_only)
    keep_idx: set[int] = {len(iter_metas) - 1}  # latest

    if best is not None:
        for i, r in enumerate(history):
            if r.metrics is best:
                # Map history index → iter_metas index. They run in lockstep
                # because we append to both per iteration (history may carry an
                # extra leading "baseline" record but iter_metas[0] is that
                # baseline too, so the indices align).
                if i < len(iter_metas):
                    keep_idx.add(i)
                break

    for i, m in enumerate(iter_metas):
        if i in keep_idx:
            continue
        _prune_iteration(m)


def _finalize_loop(
    history: list[tune_agent.IterationRecord],
    iter_metas: list[JobMeta],
    *,
    method: MethodCandidate,
    fp16: Metrics | None,
    model_id: str,
    instance_type: str | None,
) -> str:
    metrics_only = [r.metrics for r in history if r.metrics is not None]
    best = best_so_far(metrics_only)
    best_iter: int | None = None
    if best is not None:
        for i, r in enumerate(history, start=1):
            if r.metrics is best:
                best_iter = i
                break

    summary = _format_pareto_summary(
        method=method, history=history, fp16=fp16, best_iter=best_iter,
    )

    # If the running best beat fp16 across all metrics, surface that prominently.
    if best is not None and fp16 is not None and is_pareto_improvement(fp16, best):
        summary += "\nResult: Pareto-improves over fp16 baseline."
    elif best is not None and fp16 is not None:
        summary += "\nResult: did not Pareto-improve over fp16."

    return summary


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run(
    user_input: str,
    dry: bool = False,
    max_repairs: int = 3,
    max_adapt_retries: int = 2,
    *,
    tune: bool = False,
    auto_tune: bool = False,
    max_tune_iter: int = 5,
    stagnate_after: int = 2,
) -> str:
    """Research → pick → Adapt → execute → (optional) tune.

    ``tune`` enables the closed-loop tuner; ``auto_tune`` implies ``tune`` and
    skips the interactive "Tune further? [y/N]" prompt. Both are no-ops in
    ``--dry`` mode (no execution → no metrics → nothing to tune).
    """
    if auto_tune:
        tune = True

    report = research_agent.run(user_input)
    typer.echo(format_report(report), err=True)

    idx = prompt_selection(len(report.methods))
    if idx is None:
        return "aborted"

    order = [idx - 1] + [i for i in range(len(report.methods)) if i != idx - 1]

    last_error: Exception | None = None
    for pos, i in enumerate(order):
        chosen = report.methods[i]
        if pos > 0:
            typer.echo(
                f"\n[fallback] Candidate {report.methods[order[pos-1]].id} exhausted "
                f"retries ({last_error}). Trying next candidate: {chosen.name} ({chosen.id}).",
                err=True,
            )
        try:
            script_path, script_code = _run_adapt_with_retry(
                model_id=report.resolved_model_id,
                method=chosen,
                max_adapt_retries=max_adapt_retries,
                hyperparameters=chosen.hyperparameters,
            )
        except Exception as e:  # noqa: BLE001 — same-method retries exhausted
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
        trail = _format_supervise_trail(chain)

        if final.status != "completed" or not tune:
            return handoff + "\n\n" + trail

        # Baseline iteration succeeded and the user asked for tuning.
        baseline_metrics = _measure_job(final)
        if baseline_metrics is None:
            return handoff + "\n\n" + trail + "\n\n[tune] baseline measurement failed; skipping tune loop."
        _persist_metrics(final, baseline_metrics, chosen.hyperparameters)

        baseline_summary = (
            f"\n\nBaseline metrics: {metrics_summary(baseline_metrics)}"
        )

        if not auto_tune:
            if not _prompt_yes_no("\nTune further?"):
                return handoff + "\n\n" + trail + baseline_summary

        # fp16 reference (cached). Best-effort: if the venv build or measurement
        # fails we proceed without a baseline — the iteration-vs-iteration Pareto
        # check still works.
        fp16: Metrics | None = None
        try:
            fp16_dir = JOBS_ROOT / final.job_id / "fp16_baseline"
            fp16 = baseline.measure_fp16_baseline(
                model_id=report.resolved_model_id,
                instance_type=report.instance_type,
                job_dir=fp16_dir,
            )
            typer.echo(f"[tune] fp16 reference: {metrics_summary(fp16)}", err=True)
        except Exception as e:  # noqa: BLE001
            typer.echo(f"[tune] fp16 reference unavailable: {e}", err=True)

        tune_summary = _tune_loop(
            method=chosen,
            model_id=report.resolved_model_id,
            instance_type=report.instance_type,
            baseline_meta=final,
            baseline_metrics=baseline_metrics,
            fp16=fp16,
            max_tune_iter=max_tune_iter,
            stagnate_after=stagnate_after,
            max_repairs=max_repairs,
            max_adapt_retries=max_adapt_retries,
        )
        return handoff + "\n\n" + trail + baseline_summary + "\n" + tune_summary

    return f"All {len(order)} candidates failed. Last error: {last_error}"
