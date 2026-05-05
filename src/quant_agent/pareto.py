"""Pure Pareto dominance + stagnation logic for the tune loop.

All four tracked metrics are minimized: prefill latency, decode latency,
peak VRAM, perplexity. An iteration is an "improvement" over the running
best when at least one metric is strictly better (beyond epsilon) AND no
metric is strictly worse (beyond epsilon). Stagnation = N consecutive
iterations failed to improve over best-so-far.

Asymmetric tolerances reflect quality being more sensitive than speed:
ppl tolerated drift is half a percent, latency drift is two percent.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Metrics":
        return cls(**{k: float(d[k]) for k in _METRIC_NAMES})


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
    """Return the current Pareto-best from a history (latest improvement wins ties).

    Iterates in order; an entry replaces the running best whenever it Pareto-improves.
    Equivalent to the running best the tune loop tracks.
    """
    best: Metrics | None = None
    for m in history:
        if best is None or is_pareto_improvement(best, m):
            best = m
    return best


def detect_stagnation(history: list[Metrics], n: int = 2) -> bool:
    """True when the last n entries failed to Pareto-improve over the best-so-far.

    The check anchors on best_so_far over the prefix history[:-n]; that way
    a long plateau after an early peak still terminates correctly.
    """
    if len(history) <= n:
        return False
    anchor = best_so_far(history[:-n])
    if anchor is None:
        return False
    tail = history[-n:]
    return not any(is_pareto_improvement(anchor, m) for m in tail)
