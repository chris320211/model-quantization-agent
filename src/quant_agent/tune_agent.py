"""Tune agent: propose the next hyperparameter configuration to try.

Structurally simpler than fix_agent — no tools, no ReAct loop. The LLM has
all the information it needs in a single structured-output call:

  - the method's tunable ranges (HyperparamRanges)
  - what the user has already tried this run (history + metrics)
  - the running Pareto-best
  - prior wins from cross-run tune_history (warm-start hints)
  - the fp16 baseline metrics for context

Output is a discriminated union: either ``Proposal(hyperparameters=...)`` to
relaunch with, or ``Stop(reason=...)`` to terminate the loop. The orchestrator
takes the proposal, re-adapts the script with the new config, relaunches, and
measures. fix_agent stays nested inside the relaunch in case the new config
crashes the script.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .config import load_settings
from .llm import AgentStage, create_chat_model
from .hyperparam_inference import HyperparamRanges
from .pareto import Metrics
from .schemas import MethodCandidate
from .tune_history import HistoryEntry

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IterationRecord:
    """One past iteration in the current tune session."""
    hyperparameters: dict
    metrics: Metrics | None  # None when the iteration crashed even after fix_agent
    note: str | None = None


class Proposal(BaseModel):
    decision: Literal["propose"] = "propose"
    hyperparameters: dict[str, Any] = Field(
        ..., description="Flat name->value dict for the next iteration."
    )
    rationale: str = Field(..., max_length=500)


class Stop(BaseModel):
    decision: Literal["stop"] = "stop"
    reason: str = Field(..., max_length=300)


class TuneDecision(BaseModel):
    """Discriminated union: propose a config OR stop the loop."""
    decision: Literal["propose", "stop"]
    hyperparameters: dict[str, Any] | None = None
    rationale: str | None = None
    reason: str | None = None


_PROMPT = """You are the Tune agent. The user is running a closed-loop hyperparameter
search to find the Pareto-best quantization config for {method_name} on this hardware.
Pick the NEXT configuration to try, or STOP if further iterations won't help.

Method: {method_id} ({method_name})
Tunable ranges (YOU MUST pick values from these `values` lists — no novel values):
{ranges_json}

fp16 baseline (reference for "better than baseline" — lower is better on every metric):
{fp16_json}

Iterations completed in THIS session (most recent last):
{history_json}

Pareto-best so far this session:
{best_json}

Full non-dominated Pareto frontier this session:
{frontier_json}

Prior wins on the same (model, instance, method) from past sessions (warm-start hints —
use as inspiration, not gospel; hardware/dataset state may have shifted):
{prior_wins_json}

Decision rules:
  - decision="propose" with hyperparameters={{name: value, ...}}: a flat dict where every
    key is a name from the ranges and every value is from that knob's `values` list.
    Do NOT propose a configuration already in the history (exact dict match) — repeats
    waste budget.
  - decision="stop" with a one-sentence reason: choose this when (a) every reasonable
    Pareto-improving move has been tried, (b) the search space is exhausted, (c) the
    last 2 iterations regressed clearly, or (d) the running best already strictly
    dominates the fp16 baseline on speed AND VRAM with negligible ppl drift.

Optimize for Pareto improvement: any metric better by >2% AND no metric worse by >2%
(stricter for ppl: >0.5%). Avoid fads — small changes to one knob beat large changes
to several at once.
"""


def _serialize_metrics(m: Metrics | None) -> dict | None:
    return m.to_dict() if m is not None else None


def _serialize_history(history: list[IterationRecord]) -> str:
    rows = [
        {
            "iter": i,
            "hyperparameters": r.hyperparameters,
            "metrics": _serialize_metrics(r.metrics),
            "note": r.note,
        }
        for i, r in enumerate(history, start=1)
    ]
    return json.dumps(rows, indent=2)


def _serialize_prior_wins(wins: list[HistoryEntry]) -> str:
    rows = [
        {
            "hyperparameters": e.hyperparameters,
            "metrics": e.metrics,
            "timestamp": e.timestamp,
        }
        for e in wins
    ]
    return json.dumps(rows, indent=2)


def propose(
    *,
    method: MethodCandidate,
    ranges: HyperparamRanges,
    history: list[IterationRecord],
    best_so_far: Metrics | None,
    fp16_baseline: Metrics | None,
    prior_wins: list[HistoryEntry],
    pareto_frontier: list[Metrics] | None = None,
) -> TuneDecision:
    """Single LLM call. Returns a TuneDecision (propose or stop).

    Falls back to ``Stop`` with reason="schema_validation_failed" if the LLM
    returns malformed output twice — the caller treats stop the same way
    regardless of cause.
    """
    if not ranges.specs:
        return TuneDecision(decision="stop", reason="No tunable ranges available for this method.")

    s = load_settings()
    llm = create_chat_model(AgentStage.TUNE, s)
    structured = llm.with_structured_output(TuneDecision)

    prompt = _PROMPT.format(
        method_id=method.id,
        method_name=method.name,
        ranges_json=json.dumps(ranges.model_dump(), indent=2),
        fp16_json=json.dumps(_serialize_metrics(fp16_baseline), indent=2),
        history_json=_serialize_history(history),
        best_json=json.dumps(_serialize_metrics(best_so_far), indent=2),
        frontier_json=json.dumps(
            [_serialize_metrics(m) for m in (pareto_frontier or [])], indent=2
        ),
        prior_wins_json=_serialize_prior_wins(prior_wins),
    )

    try:
        decision: TuneDecision = structured.invoke(prompt)
    except ValidationError as e:
        log.warning("tune_agent first call failed validation: %s", e)
        try:
            decision = structured.invoke(
                prompt + f"\n\nPrior call rejected for: {e}\nRe-emit valid TuneDecision."
            )
        except ValidationError as e2:
            log.warning("tune_agent second call also failed: %s", e2)
            return TuneDecision(decision="stop", reason="schema_validation_failed")

    return _enforce_constraints(decision, ranges, history)


def _enforce_constraints(
    decision: TuneDecision,
    ranges: HyperparamRanges,
    history: list[IterationRecord],
) -> TuneDecision:
    """Validate that proposed values are in-range and not a duplicate.

    Re-runs of identical configs waste budget; out-of-range values would crash
    the next quantization run. Both get converted to a Stop so the orchestrator
    handles it cleanly.
    """
    if decision.decision == "stop":
        return decision
    if not decision.hyperparameters:
        return TuneDecision(decision="stop", reason="empty hyperparameter proposal")

    by_name = {s.name: s for s in ranges.specs}
    for name, value in decision.hyperparameters.items():
        spec = by_name.get(name)
        if spec is None:
            return TuneDecision(
                decision="stop",
                reason=f"proposed unknown knob {name!r}",
            )
        type_ok = {
            "bool": type(value) is bool,
            "int": type(value) is int,
            "float": type(value) in {int, float} and type(value) is not bool,
            "categorical": True,
        }[spec.type]
        if not type_ok:
            return TuneDecision(
                decision="stop",
                reason=f"proposed {name} has wrong type for {spec.type}: {type(value).__name__}",
            )
        if value not in spec.values:
            return TuneDecision(
                decision="stop",
                reason=f"proposed {name}={value!r} not in allowed values {spec.values!r}",
            )

    # Materialize a complete effective configuration so history, generated scripts,
    # repair metadata, and duplicate detection all describe the same settings.
    full = {name: spec.default for name, spec in by_name.items()}
    full.update(decision.hyperparameters)
    decision = decision.model_copy(update={"hyperparameters": full})

    for prior in history:
        if prior.hyperparameters == decision.hyperparameters:
            return TuneDecision(
                decision="stop",
                reason="proposal duplicates a prior iteration",
            )
    return decision
