from quant_agent import research_agent
from quant_agent.compatibility import CompatibilityDecision, ConstraintReason
from quant_agent.research_agent import _parse_input
from quant_agent.schemas import ConsideredMethod, MethodCandidate, ResearchReport


def test_parse_input_extracts_constraints_without_polluting_model_phrase():
    parsed = _parse_input(
        "quantize llama2 7b to g5.xlarge 4-bit for vllm speed priority no calibration"
    )
    assert parsed.instance_phrase == "g5.xlarge"
    assert parsed.target_bits == 4
    assert parsed.backend == "vllm"
    assert parsed.priority == "speed"
    assert parsed.have_calibration_data is False
    assert parsed.model_phrase == "llama2 7b"


def test_parse_input_extracts_qat_and_kv_cache_flags():
    parsed = _parse_input("quantize org/model for p5.48xlarge using QAT KV-cache")
    assert parsed.allow_qat is True
    assert parsed.need_kv_cache_quant is True
    assert parsed.model_phrase == "org/model"


def _minimal_report() -> ResearchReport:
    from quant_agent.tools.recommender import load_catalog

    catalog = load_catalog()
    candidates = []
    for method_id in ("awq", "gptq", "bnb_nf4"):
        row = next(row for row in catalog if row["id"] == method_id)
        candidates.append(MethodCandidate(
            id=method_id,
            name=row["name"],
            repo_url=row["repos"][0],
            bits=4,
            est_vram_gb=3.5,
            quality_score=row["quality"],
            speed_score=row["speedup"],
            needs_calibration=row["needs_calibration"],
            summary="fixture",
        ))
    return ResearchReport(
        resolved_model_id="org/model",
        params_b=5,
        considered=[
            ConsideredMethod(id=row["id"], verdict="include", reason="fixture")
            for row in catalog
        ],
        methods=candidates,
        tradeoffs="fixture",
    )


def test_deterministic_block_cannot_be_a_finalist():
    report = _minimal_report()
    decision = CompatibilityDecision(
        method_id="awq", status="blocked",
        reasons=[ConstraintReason(
            code="compute_capability_too_low", message="requires sm_90", source="capability"
        )],
    )
    try:
        research_agent._require_no_blocked_finalists(report, [decision])
    except ValueError as exc:
        assert "awq" in str(exc)
    else:
        raise AssertionError("deterministically blocked finalist was accepted")


def test_deterministic_block_canonicalizes_considered_reason():
    report = _minimal_report()
    decision = CompatibilityDecision(
        method_id="fp8", status="blocked",
        reasons=[ConstraintReason(
            code="compute_capability_too_low", message="requires sm_90", source="capability"
        )],
    )
    updated = research_agent._canonicalize_blocked_verdicts(report, [decision])
    row = next(item for item in updated.considered if item.id == "fp8")
    assert row.verdict == "reject"
    assert "requires sm_90" in row.reason


def test_port_required_finalist_is_annotated_not_rejected():
    report = _minimal_report()
    decision = CompatibilityDecision(
        method_id="awq", status="port_required", model_family="qwen2",
        reasons=[ConstraintReason(
            code="undocumented_model_family",
            message="no documented qwen2 support; attempt a separate overlay port",
            source="capability",
        )],
    )
    research_agent._require_no_blocked_finalists(report, [decision])
    updated = research_agent._annotate_port_candidates(report, [decision])
    candidate = next(method for method in updated.methods if method.id == "awq")
    assert candidate.requires_port is True
    assert "overlay port" in candidate.port_reason


def test_port_required_considered_method_is_not_rejected_for_missing_support():
    report = _minimal_report()
    report = report.model_copy(update={
        "considered": [
            row.model_copy(update={"verdict": "reject"}) if row.id == "awq" else row
            for row in report.considered
        ]
    })
    decision = CompatibilityDecision(
        method_id="awq", status="port_required", model_family="qwen2",
        reasons=[ConstraintReason(
            code="undocumented_model_family",
            message="no documented qwen2 support; attempt a separate overlay port",
            source="capability",
        )],
    )
    updated = research_agent._canonicalize_port_verdicts(report, [decision])
    row = next(item for item in updated.considered if item.id == "awq")
    assert row.verdict == "include"
    assert row.reason.startswith("overlay port path:")
