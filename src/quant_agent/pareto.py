"""Pure Pareto dominance + stagnation logic for the tune loop.

All four tracked metrics are minimized: prefill latency, per-token decode latency,
peak VRAM, and perplexity. The tuner retains the complete epsilon-aware
non-dominated frontier. Stagnation means N consecutive iterations failed to change
that frontier.

Asymmetric tolerances reflect quality being more sensitive than speed:
ppl tolerated drift is half a percent, latency drift is two percent.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Iterable

# Relative tolerances; each metric is "worse" only if it regresses by more
# than this fraction of the prior value. ppl is the tightest because a 1%
# perplexity bump is a meaningful quality cliff.
EPSILONS: dict[str, float] = {
    "prefill_ms": 0.02,
    "decode_ms": 0.02,
    "vram_gb": 0.01,
    "ppl": 0.005,
}

_METRIC_NAMES = tuple(EPSILONS.keys())


@dataclass(frozen=True)
class Metrics:
    prefill_ms: float
    decode_ms: float
    vram_gb: float
    ppl: float
    prefill_std_ms: float | None = None
    decode_std_ms: float | None = None
    samples: int = 1

    def __post_init__(self) -> None:
        for name in _METRIC_NAMES:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
        if self.ppl <= 0:
            raise ValueError("ppl must be greater than zero")
        for name in ("prefill_std_ms", "decode_std_ms"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.samples < 1:
            raise ValueError("samples must be at least one")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Metrics":
        return cls(
            **{k: float(d[k]) for k in _METRIC_NAMES},
            prefill_std_ms=(
                float(d["prefill_std_ms"]) if d.get("prefill_std_ms") is not None else None
            ),
            decode_std_ms=(
                float(d["decode_std_ms"]) if d.get("decode_std_ms") is not None else None
            ),
            samples=int(d.get("samples", 1)),
        )


def _delta(prev: float, curr: float) -> float:
    """Signed relative change. Positive = curr is larger than prev."""
    if prev == 0:
        return 0.0 if curr == 0 else float("inf")
    return (curr - prev) / abs(prev)


def is_pareto_improvement(prev: Metrics, curr: Metrics) -> bool:
    """True when curr Pareto-dominates prev with epsilon-tolerance.

    Lower is better for every tracked metric. A regression of any metric beyond
    its epsilon disqualifies, even if other metrics improve.
    """
    any_better = False
    for name in _METRIC_NAMES:
        eps = EPSILONS[name]
        d = _delta(getattr(prev, name), getattr(curr, name))
        if d > eps:
            return False
        if d < -eps:
            any_better = True
    return any_better


def best_so_far(history: Iterable[Metrics]) -> Metrics | None:
    """Return one deterministic representative from the Pareto frontier.

    Kept for compatibility with callers that need a single prompt anchor. It must not
    be used for pruning; use :func:`pareto_frontier` for that.
    """
    frontier = pareto_frontier(history)
    return frontier[-1] if frontier else None


def _epsilon_equivalent(a: Metrics, b: Metrics) -> bool:
    return all(
        abs(_delta(getattr(a, name), getattr(b, name))) <= EPSILONS[name]
        for name in _METRIC_NAMES
    )


def pareto_frontier(history: Iterable[Metrics]) -> list[Metrics]:
    """Return every epsilon-aware non-dominated point, preserving input order.

    Equivalent points collapse to the latest observation so repeated noisy runs do
    not inflate the frontier. Incomparable speed/quality tradeoffs are all retained.
    """
    frontier: list[Metrics] = []
    for candidate in history:
        if any(is_pareto_improvement(candidate, existing) for existing in frontier):
            # Existing dominates candidate.
            continue
        frontier = [
            existing
            for existing in frontier
            if not is_pareto_improvement(existing, candidate)
            and not _epsilon_equivalent(existing, candidate)
        ]
        frontier.append(candidate)
    return frontier


def detect_stagnation(history: list[Metrics], n: int = 2) -> bool:
    """True when the last ``n`` entries failed to change the Pareto frontier.

    Adding an incomparable tradeoff counts as progress; duplicates and dominated
    points do not. This makes stagnation independent of any scalar winner policy.
    """
    if len(history) <= n:
        return False
    before = pareto_frontier(history[:-n])
    current = list(before)
    changed = False
    for point in history[-n:]:
        updated = pareto_frontier([*current, point])
        equivalent = len(updated) == len(current) and all(
            any(_epsilon_equivalent(left, right) for right in current)
            for left in updated
        )
        if not equivalent:
            changed = True
        current = updated
    return not changed
