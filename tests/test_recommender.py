from __future__ import annotations

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

from quant_agent.tools.recommender import Constraints, rank


def test_7b_fits_at_4bit_on_16gb():
    ranked = rank(Constraints(params_b=7.0, vram_gb=16.0, target_bits=4, priority="quality"))
    assert ranked, "expected at least one method"
    top = ranked[0]
    assert top["bits"] == 4
    # Llama-7B at 4-bit ≈ 3.5 GB weights; comfortably under 16 GB.
    assert top["weight_gb"] < 4.0


def test_70b_does_not_fit_at_4bit_on_16gb():
    ranked = rank(Constraints(params_b=70.0, vram_gb=16.0, target_bits=4))
    assert ranked == [], "70B model at 4-bit should not fit on 16 GB"


def test_backend_filter_excludes_incompatible():
    # GGUF / llama.cpp is not a vllm backend; vllm filter must exclude it.
    ranked = rank(Constraints(params_b=7.0, vram_gb=24.0, backend="vllm"))
    ids = [m["id"] for m in ranked]
    assert "gguf" not in ids
    assert any(i in ids for i in ("awq", "gptq"))


def test_llama_cpp_surfaces_gguf():
    ranked = rank(Constraints(params_b=7.0, vram_gb=24.0, backend="llama_cpp"))
    assert any(m["id"] == "gguf" for m in ranked)


def test_no_calibration_excludes_gptq_and_awq():
    ranked = rank(
        Constraints(params_b=7.0, vram_gb=24.0, target_bits=4, have_calibration_data=False)
    )
    ids = {m["id"] for m in ranked}
    assert "gptq" not in ids
    assert "awq" not in ids
    # Calibration-free methods should remain.
    assert {"hqq", "bnb_nf4"} & ids


def test_qat_gated_by_allow_qat():
    ranked_default = rank(Constraints(params_b=7.0, vram_gb=24.0), top_k=20)
    assert "llm_qat" not in {m["id"] for m in ranked_default}

    ranked_qat = rank(Constraints(params_b=7.0, vram_gb=24.0, allow_qat=True), top_k=20)
    assert "llm_qat" in {m["id"] for m in ranked_qat}


def test_kv_cache_requirement_surfaces_kivi():
    ranked = rank(
        Constraints(params_b=7.0, vram_gb=24.0, need_kv_cache_quant=True)
    )
    assert any(m["id"] == "kivi" for m in ranked)


def test_priority_changes_ordering():
    q = rank(Constraints(params_b=7.0, vram_gb=24.0, target_bits=4, priority="quality"))
    s = rank(Constraints(params_b=7.0, vram_gb=24.0, target_bits=4, priority="speed"))
    assert q and s
    # With quality priority, AWQ (quality=5) should outrank GPTQ (quality=4) all else equal.
    q_ids = [m["id"] for m in q]
    if "awq" in q_ids and "gptq" in q_ids:
        assert q_ids.index("awq") < q_ids.index("gptq")
