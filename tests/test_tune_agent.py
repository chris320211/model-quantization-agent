"""tune_agent.propose: structured-output LLM call + post-hoc constraint enforcement."""
from __future__ import annotations

from unittest.mock import MagicMock

from quant_agent import tune_agent
from quant_agent.hyperparam_inference import HyperparamRanges, HyperparamSpec
from quant_agent.pareto import Metrics
from quant_agent.schemas import MethodCandidate
from quant_agent.tune_agent import IterationRecord, TuneDecision


def _method() -> MethodCandidate:
    return MethodCandidate(
        id="awq",
        name="AWQ",
        repo_url="https://github.com/casper-hansen/AutoAWQ",
        bits=4,
        est_vram_gb=4.7,
        quality_score=5,
        speed_score=4,
        needs_calibration=True,
        summary="x",
    )


def _ranges() -> HyperparamRanges:
    return HyperparamRanges(
        method_id="awq",
        specs=[
            HyperparamSpec(name="group_size", type="int",
                           values=[32, 64, 128, -1], default=128),
            HyperparamSpec(name="zero_point", type="bool",
                           values=[True, False], default=True),
        ],
    )


def _metrics() -> Metrics:
    return Metrics(prefill_ms=100, decode_ms=200, vram_gb=10, ppl=8)


def _patch_llm(monkeypatch, decision: TuneDecision):
    """Replace ChatAnthropic + with_structured_output to return a canned decision."""
    fake_structured = MagicMock()
    fake_structured.invoke = MagicMock(return_value=decision)

    fake_llm = MagicMock()
    fake_llm.with_structured_output = MagicMock(return_value=fake_structured)

    monkeypatch.setattr(tune_agent, "ChatAnthropic", lambda **kw: fake_llm)
    return fake_structured


# Happy path ------------------------------------------------------------------


def test_propose_returns_valid_decision(monkeypatch):
    decision = TuneDecision(
        decision="propose",
        hyperparameters={"group_size": 64, "zero_point": False},
        rationale="trying smaller group size",
    )
    _patch_llm(monkeypatch, decision)

    result = tune_agent.propose(
        method=_method(),
        ranges=_ranges(),
        history=[IterationRecord(hyperparameters={"group_size": 128, "zero_point": True}, metrics=_metrics())],
        best_so_far=_metrics(),
        fp16_baseline=_metrics(),
        prior_wins=[],
    )

    assert result.decision == "propose"
    assert result.hyperparameters == {"group_size": 64, "zero_point": False}


def test_propose_stop_passes_through(monkeypatch):
    decision = TuneDecision(decision="stop", reason="search exhausted")
    _patch_llm(monkeypatch, decision)

    result = tune_agent.propose(
        method=_method(), ranges=_ranges(), history=[],
        best_so_far=None, fp16_baseline=None, prior_wins=[],
    )
    assert result.decision == "stop"
    assert "exhausted" in result.reason


# Constraint enforcement ------------------------------------------------------


def test_constraint_rejects_unknown_knob(monkeypatch):
    decision = TuneDecision(
        decision="propose",
        hyperparameters={"made_up_knob": 5},
        rationale="invented a knob",
    )
    _patch_llm(monkeypatch, decision)

    result = tune_agent.propose(
        method=_method(), ranges=_ranges(), history=[],
        best_so_far=None, fp16_baseline=None, prior_wins=[],
    )
    assert result.decision == "stop"
    assert "made_up_knob" in result.reason


def test_constraint_rejects_out_of_range_value(monkeypatch):
    decision = TuneDecision(
        decision="propose",
        hyperparameters={"group_size": 96},  # not in [32, 64, 128, -1]
        rationale="splitting the difference",
    )
    _patch_llm(monkeypatch, decision)

    result = tune_agent.propose(
        method=_method(), ranges=_ranges(), history=[],
        best_so_far=None, fp16_baseline=None, prior_wins=[],
    )
    assert result.decision == "stop"
    assert "96" in result.reason


def test_constraint_rejects_duplicate_proposal(monkeypatch):
    repeat_hp = {"group_size": 64, "zero_point": True}
    decision = TuneDecision(
        decision="propose",
        hyperparameters=repeat_hp,
        rationale="trying again",
    )
    _patch_llm(monkeypatch, decision)

    result = tune_agent.propose(
        method=_method(),
        ranges=_ranges(),
        history=[IterationRecord(hyperparameters=dict(repeat_hp), metrics=_metrics())],
        best_so_far=_metrics(),
        fp16_baseline=None,
        prior_wins=[],
    )
    assert result.decision == "stop"
    assert "duplicates" in result.reason


def test_propose_with_no_ranges_immediately_stops(monkeypatch):
    """When the method has nothing to tune, skip the LLM and stop."""
    fake_llm = MagicMock()
    monkeypatch.setattr(tune_agent, "ChatAnthropic", lambda **kw: fake_llm)

    result = tune_agent.propose(
        method=_method(),
        ranges=HyperparamRanges(method_id="awq", specs=[]),
        history=[],
        best_so_far=None,
        fp16_baseline=None,
        prior_wins=[],
    )
    assert result.decision == "stop"
    fake_llm.with_structured_output.assert_not_called()


def test_empty_proposal_converts_to_stop(monkeypatch):
    decision = TuneDecision(decision="propose", hyperparameters={}, rationale="nothing")
    _patch_llm(monkeypatch, decision)

    result = tune_agent.propose(
        method=_method(), ranges=_ranges(), history=[],
        best_so_far=None, fp16_baseline=None, prior_wins=[],
    )
    assert result.decision == "stop"
