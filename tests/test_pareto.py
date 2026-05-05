"""Pareto dominance + stagnation: pure-logic table-driven tests."""
from __future__ import annotations

import pytest

from quant_agent.pareto import (
    EPSILONS,
    Metrics,
    best_so_far,
    detect_stagnation,
    is_pareto_improvement,
)


def _m(prefill=100.0, decode=200.0, vram=10.0, ppl=8.0) -> Metrics:
    return Metrics(prefill_ms=prefill, decode_ms=decode, vram_gb=vram, ppl=ppl)


# is_pareto_improvement -------------------------------------------------------


def test_strict_dominance_is_improvement():
    prev = _m(100, 200, 10, 8)
    curr = _m(80, 180, 9, 7.9)  # all four better well past epsilon
    assert is_pareto_improvement(prev, curr)


def test_no_change_is_not_improvement():
    """Zero delta on every metric means no axis improved → not an improvement."""
    prev = _m()
    curr = _m()
    assert not is_pareto_improvement(prev, curr)


def test_one_axis_better_others_unchanged_is_improvement():
    prev = _m(100, 200, 10, 8)
    # decode improves by >2%, others unchanged
    curr = _m(100, 195, 10, 8)
    assert is_pareto_improvement(prev, curr)


def test_one_axis_regresses_disqualifies_even_if_others_improve():
    prev = _m(100, 200, 10, 8)
    # prefill improved -10% but ppl regressed +2% (>0.5% epsilon)
    curr = _m(90, 200, 10, 8.16)
    assert not is_pareto_improvement(prev, curr)


def test_within_epsilon_is_treated_as_unchanged():
    """A 1% change in latency falls under the 2% epsilon — not better, not worse."""
    prev = _m(100, 200, 10, 8)
    curr = _m(99, 199, 10, 8)  # 1% better — within eps, NOT a real improvement
    assert not is_pareto_improvement(prev, curr)


def test_ppl_epsilon_tighter_than_latency():
    """0.6% ppl drift counts as worse, while 1% latency drift does not."""
    prev = _m(100, 200, 10, 8)
    # latency improves 5%, but ppl regresses 0.6% > 0.5% eps → disqualified
    curr = _m(95, 190, 10, 8.05)
    assert not is_pareto_improvement(prev, curr)


def test_vram_epsilon():
    """vram is at 1% — 2% regression should disqualify."""
    prev = _m(100, 200, 10, 8)
    curr = _m(80, 180, 10.21, 8)  # vram up 2.1%, latency wins
    assert not is_pareto_improvement(prev, curr)


# best_so_far -----------------------------------------------------------------


def test_best_so_far_returns_first_when_no_improvement():
    history = [_m(100, 200, 10, 8), _m(100, 200, 10, 8), _m(100, 200, 10, 8)]
    assert best_so_far(history) == history[0]


def test_best_so_far_advances_on_pareto_winner():
    h = [_m(100, 200, 10, 8), _m(80, 180, 9, 7.9)]
    assert best_so_far(h) == h[1]


def test_best_so_far_keeps_prior_when_later_regresses():
    h = [_m(100, 200, 10, 8), _m(80, 180, 9, 7.9), _m(120, 200, 10, 8)]
    assert best_so_far(h) == h[1]


def test_best_so_far_empty_returns_none():
    assert best_so_far([]) is None


# detect_stagnation -----------------------------------------------------------


def test_stagnation_short_history_is_false():
    assert not detect_stagnation([_m(), _m()], n=2)


def test_stagnation_when_last_n_dont_improve():
    h = [
        _m(100, 200, 10, 8),
        _m(80, 180, 9, 7.9),  # wins
        _m(81, 180, 9, 7.9),  # no improvement (tie)
        _m(80, 181, 9, 7.9),  # no improvement (tie)
    ]
    assert detect_stagnation(h, n=2)


def test_no_stagnation_when_recent_iter_improved():
    h = [
        _m(100, 200, 10, 8),
        _m(95, 195, 10, 8),  # within eps; not an improvement
        _m(80, 180, 9, 7.9),  # wins
    ]
    assert not detect_stagnation(h, n=2)


def test_stagnation_long_plateau_after_early_peak():
    """Improvement at iter 2 then 4 stagnant iters — stagnation triggers."""
    h = [
        _m(100, 200, 10, 8),
        _m(80, 180, 9, 7.9),  # peak
        _m(80, 180, 9, 7.9),
        _m(80, 180, 9, 7.9),
        _m(80, 180, 9, 7.9),
    ]
    assert detect_stagnation(h, n=2)


def test_metrics_round_trip():
    m = _m(123.4, 567.8, 9.01, 8.234)
    d = m.to_dict()
    assert Metrics.from_dict(d) == m


def test_epsilons_are_relative_not_absolute():
    """Sanity: epsilon is a fraction, not a raw float — 100ms ± 1ms is still 1%."""
    assert all(0 < v < 1 for v in EPSILONS.values())
