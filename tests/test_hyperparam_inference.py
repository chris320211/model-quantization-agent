"""hyperparam_inference: catalog tier + LLM tier + validation."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from quant_agent import hyperparam_inference as hpi
from quant_agent.hyperparam_inference import (
    HyperparamRanges,
    HyperparamSpec,
    default_config,
    infer_ranges,
)
from quant_agent.schemas import MethodCandidate


def _method(mid: str = "awq") -> MethodCandidate:
    return MethodCandidate(
        id=mid,
        name=mid.upper(),
        repo_url=f"https://github.com/example/{mid}",
        bits=4,
        est_vram_gb=4.7,
        quality_score=5,
        speed_score=4,
        needs_calibration=True,
        summary="x",
    )


# Tier 1: catalog defaults ----------------------------------------------------


def test_infer_ranges_uses_catalog_defaults_for_known_method(tmp_path, monkeypatch):
    """awq has hyperparameters_default in seed/methods.yaml — no LLM call needed."""
    # block any cache hit so we exercise the catalog path explicitly
    monkeypatch.setattr(hpi, "_load_global_cache", lambda: {})

    llm_called = MagicMock()
    monkeypatch.setattr(hpi, "_query_llm", llm_called)

    ranges = infer_ranges(_method("awq"), job_dir=tmp_path)
    assert ranges.method_id == "awq"
    assert any(s.name == "group_size" for s in ranges.specs)
    llm_called.assert_not_called()
    assert (tmp_path / "hyperparams.yaml").exists()


# Tier 2: LLM fallback --------------------------------------------------------


def test_infer_ranges_falls_back_to_llm_when_catalog_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(hpi, "_catalog_defaults", lambda mid: None)
    monkeypatch.setattr(hpi, "_load_global_cache", lambda: {})
    saved: dict = {}
    monkeypatch.setattr(hpi, "_save_global_cache", lambda c: saved.update(c))
    monkeypatch.setattr(hpi, "_fetch_readme", lambda m: "FAKE README")

    llm_output = HyperparamRanges(
        method_id="custom",
        specs=[
            HyperparamSpec(name="alpha", type="float",
                           values=[0.5, 0.7, 0.9], default=0.7),
        ],
    )
    monkeypatch.setattr(hpi, "_query_llm", lambda m, r: llm_output)

    ranges = infer_ranges(_method("custom"), job_dir=tmp_path)
    assert ranges.specs[0].name == "alpha"
    assert any(saved)


def test_llm_validation_failure_falls_back_to_empty(tmp_path, monkeypatch):
    """Two LLM attempts both fail Pydantic → empty ranges, no crash."""
    monkeypatch.setattr(hpi, "_catalog_defaults", lambda mid: None)
    monkeypatch.setattr(hpi, "_load_global_cache", lambda: {})
    monkeypatch.setattr(hpi, "_save_global_cache", lambda c: None)
    monkeypatch.setattr(hpi, "_fetch_readme", lambda m: "FAKE README")

    def bad_llm(method, readme):
        # Manually raise ValidationError without going through Pydantic's
        # structured-output happy path: simulate the wrapper rejecting output.
        raise ValidationError.from_exception_data("HyperparamRanges", [])

    monkeypatch.setattr(hpi, "_query_llm", bad_llm)

    ranges = infer_ranges(_method("custom"), job_dir=tmp_path)
    assert ranges.specs == []


def test_validate_specs_rejects_default_not_in_values():
    bad = HyperparamRanges(
        method_id="x",
        specs=[
            HyperparamSpec(name="a", type="int", values=[1, 2, 3], default=99),
        ],
    )
    ok, err = hpi._validate_specs(bad)
    assert not ok
    assert "default" in err


def test_validate_specs_rejects_duplicate_names():
    bad = HyperparamRanges(
        method_id="x",
        specs=[
            HyperparamSpec(name="a", type="int", values=[1, 2], default=1),
            HyperparamSpec(name="a", type="int", values=[3, 4], default=3),
        ],
    )
    ok, err = hpi._validate_specs(bad)
    assert not ok
    assert "duplicate" in err


def test_default_config_returns_flat_name_value_dict():
    ranges = HyperparamRanges(
        method_id="awq",
        specs=[
            HyperparamSpec(name="group_size", type="int", values=[32, 64, 128], default=128),
            HyperparamSpec(name="sym", type="bool", values=[True, False], default=True),
        ],
    )
    assert default_config(ranges) == {"group_size": 128, "sym": True}


# Cache behavior --------------------------------------------------------------


def test_global_cache_short_circuits_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(hpi, "_catalog_defaults", lambda mid: None)
    cached = HyperparamRanges(
        method_id="custom",
        specs=[HyperparamSpec(name="x", type="int", values=[1, 2], default=1)],
    )
    monkeypatch.setattr(hpi, "_load_global_cache", lambda: {
        "custom@unknown": cached.model_dump()
    })

    llm_called = MagicMock()
    monkeypatch.setattr(hpi, "_query_llm", llm_called)

    ranges = infer_ranges(_method("custom"), job_dir=tmp_path)
    assert ranges.specs[0].name == "x"
    llm_called.assert_not_called()


def test_infer_type_classifies_correctly():
    assert hpi._infer_type([True, False]) == "bool"
    assert hpi._infer_type([1, 2, 3]) == "int"
    assert hpi._infer_type([0.1, 0.5, 1.0]) == "float"
    assert hpi._infer_type(["GEMM", "GEMV"]) == "categorical"
