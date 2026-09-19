"""Offline tests for the quant-benchmark helper."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agents" / "skills" / "_shared" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchmark as benchmark_mod  # noqa: E402


def test_compare_metrics_quality_and_efficiency():
    quantized = {
        "perplexity": 12.0,
        "peak_vram_gb": 4.0,
        "tokens_per_s": 80.0,
    }
    baseline = {
        "perplexity": 10.0,
        "peak_vram_gb": 8.0,
        "tokens_per_s": 40.0,
    }
    cmp_ = benchmark_mod.compare_metrics(quantized, baseline)
    assert cmp_["ppl_delta"] == pytest.approx(2.0)
    assert cmp_["ppl_ratio"] == pytest.approx(1.2)
    assert cmp_["quality_ok"] is True
    assert cmp_["improved_vram"] is True
    assert cmp_["improved_throughput"] is True
    assert cmp_["efficiency_improved"] is True


def test_compare_metrics_quality_not_ok_when_ppl_explodes():
    quantized = {"perplexity": 40.0, "peak_vram_gb": 3.0, "tokens_per_s": 10.0}
    baseline = {"perplexity": 10.0, "peak_vram_gb": 3.0, "tokens_per_s": 10.0}
    cmp_ = benchmark_mod.compare_metrics(quantized, baseline)
    assert cmp_["quality_ok"] is False
    assert cmp_["ppl_ratio"] == pytest.approx(4.0)
    assert cmp_["improved_vram"] is False
    assert cmp_["efficiency_improved"] is False


def test_compare_metrics_rejects_nonfinite_ppl():
    with pytest.raises(ValueError):
        benchmark_mod.compare_metrics(
            {"perplexity": float("nan")},
            {"perplexity": 10.0},
        )


def test_quant_benchmark_skill_requires_wikitext_and_both_sides():
    text = (ROOT / "agents" / "skills" / "quant-benchmark" / "SKILL.md").read_text()
    assert "benchmark.py" in text
    assert "WikiText-2" in text
    assert "fp16" in text
    assert "command -v python || command -v python3" in text
    assert "warmup" in benchmark_mod.BENCHMARK_SCRIPT or "compile" in benchmark_mod.BENCHMARK_SCRIPT.lower()
    assert "compare.py" in text or "compare/" in text


def test_parent_pipeline_always_launches_benchmark():
    quant = (ROOT / "agents" / "skills" / "quant" / "SKILL.md").read_text()
    contract = (ROOT / "agents" / "skills" / "_shared" / "pipeline_contract.md").read_text()
    assert "quant-benchmark" in quant
    assert "always" in quant.lower()
    assert "quant-benchmark" in contract
    verify = (ROOT / "agents" / "skills" / "quant-verify" / "SKILL.md").read_text()
    assert "quant-benchmark" in verify
