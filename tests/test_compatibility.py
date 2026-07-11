from quant_agent.compatibility import (
    CompatibilityRequest,
    evaluate_method,
    infer_model_family,
    load_capabilities,
)


def _method(**overrides):
    data = {
        "id": "x",
        "bits": [4, 8],
        "quantizes": ["weights"],
        "needs_calibration": False,
        "qat": False,
        "inference_backends": ["transformers", "vllm"],
    }
    data.update(overrides)
    return data


def test_infer_model_family_from_transformers_architecture():
    assert infer_model_family(["Qwen2ForCausalLM"]) == "qwen2"
    assert infer_model_family(["LlamaForCausalLM"]) == "llama"
    assert infer_model_family(["UnknownForCausalLM"]) is None


def test_hard_constraints_block_backend_bits_and_qat():
    decision = evaluate_method(
        _method(bits=[4], qat=True),
        CompatibilityRequest(
            target_bits=8, backend="tgi", allow_qat=False,
        ),
    )
    assert decision.status == "blocked"
    assert {r.code for r in decision.reasons} == {
        "backend_mismatch", "qat_not_allowed", "bit_width_mismatch"
    }


def test_vram_estimate_selects_first_fitting_preferred_width():
    decision = evaluate_method(
        _method(bits=[4, 8]),
        CompatibilityRequest(params_b=7, vram_gb=6),
    )
    assert decision.status == "eligible"
    assert decision.chosen_bits == 4


def test_vram_estimate_blocks_when_no_width_fits():
    decision = evaluate_method(
        _method(bits=[4, 8]),
        CompatibilityRequest(params_b=70, vram_gb=24),
    )
    assert decision.status == "blocked"
    assert "estimated_vram_exceeded" in {r.code for r in decision.reasons}


def test_capability_hardware_and_family_constraints_are_enforced():
    cap = {
        "supported_families": ["llama"],
        "family_policy": "allowlist",
        "min_compute_capability": 9.0,
    }
    decision = evaluate_method(
        _method(),
        CompatibilityRequest(
            compute_capability=8.6, architectures=["Qwen2ForCausalLM"]
        ),
        cap,
    )
    assert decision.status == "blocked"
    codes = {r.code for r in decision.reasons}
    assert "compute_capability_too_low" in codes
    assert "model_family_not_supported" in codes


def test_partial_documented_family_list_does_not_become_a_false_denylist():
    decision = evaluate_method(
        _method(),
        CompatibilityRequest(architectures=["Qwen2ForCausalLM"]),
        {"supported_families": ["llama"]},
    )
    assert decision.status == "unknown"
    assert "model_family_not_supported" not in {r.code for r in decision.reasons}


def test_kv_only_is_blocked_unless_requested():
    method = _method(quantizes=["kv_cache"])
    blocked = evaluate_method(method, CompatibilityRequest())
    requested = evaluate_method(
        method, CompatibilityRequest(need_kv_cache_quant=True)
    )
    assert blocked.status == "blocked"
    assert requested.status == "eligible"


def test_documented_capability_turns_unknown_into_eligible():
    decision = evaluate_method(
        _method(),
        CompatibilityRequest(architectures=["LlamaForCausalLM"]),
        {"supported_families": ["llama"]},
    )
    assert decision.status == "eligible"


def test_packaged_capability_dataset_covers_catalog_with_pinned_sources():
    from quant_agent.tools.recommender import load_catalog

    load_capabilities.cache_clear()
    capabilities = load_capabilities()
    assert set(capabilities) == {row["id"] for row in load_catalog()}
    assert capabilities["fp8"]["min_compute_capability"] == 8.9
    assert capabilities["smoothquant"]["support_tier"] == "verified"
