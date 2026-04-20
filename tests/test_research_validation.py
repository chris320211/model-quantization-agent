from __future__ import annotations

import pytest

from quant_agent.schemas import ConsideredMethod, MethodCandidate, ResearchReport
from quant_agent.tools.recommender import load_catalog


def _all_catalog_ids() -> list[str]:
    return [m["id"] for m in load_catalog()]


def _full_considered(include_ids: set[str]) -> list[ConsideredMethod]:
    out = []
    for mid in _all_catalog_ids():
        verdict = "include" if mid in include_ids else "reject"
        out.append(ConsideredMethod(id=mid, verdict=verdict, reason="ok"))
    return out


def _candidate(mid: str) -> MethodCandidate:
    catalog = {m["id"]: m for m in load_catalog()}
    entry = catalog[mid]
    return MethodCandidate(
        id=mid,
        name=entry["name"],
        repo_url=(entry.get("repos") or ["https://github.com/x/y"])[0],
        bits=(entry.get("bits") or [4])[0],
        est_vram_gb=4.0,
        quality_score=entry.get("quality", 3),
        speed_score=entry.get("speedup", 3),
        needs_calibration=bool(entry.get("needs_calibration", False)),
        summary="ok",
    )


def _base_kwargs() -> dict:
    return {
        "resolved_model_id": "meta-llama/Llama-2-7b-hf",
        "tradeoffs": "paragraph",
    }


def test_happy_path_validates():
    includes = {"awq", "gptq", "bnb_nf4"}
    r = ResearchReport(
        considered=_full_considered(includes),
        methods=[_candidate(i) for i in sorted(includes)],
        **_base_kwargs(),
    )
    assert len(r.considered) == len(_all_catalog_ids())


def test_missing_catalog_id_raises():
    includes = {"awq", "gptq", "bnb_nf4"}
    considered = [c for c in _full_considered(includes) if c.id != "fp8"]
    with pytest.raises(Exception) as ei:
        ResearchReport(
            considered=considered,
            methods=[_candidate(i) for i in sorted(includes)],
            **_base_kwargs(),
        )
    assert "missing catalog ids" in str(ei.value)
    assert "fp8" in str(ei.value)


def test_finalist_without_include_verdict_raises():
    includes = {"awq", "gptq"}  # bnb_nf4 is reject here
    with pytest.raises(Exception) as ei:
        ResearchReport(
            considered=_full_considered(includes),
            methods=[_candidate("awq"), _candidate("gptq"), _candidate("bnb_nf4")],
            **_base_kwargs(),
        )
    assert "without an 'include' verdict" in str(ei.value)
    assert "bnb_nf4" in str(ei.value)


def test_duplicate_considered_id_raises():
    includes = {"awq", "gptq", "bnb_nf4"}
    considered = _full_considered(includes)
    considered.append(ConsideredMethod(id="awq", verdict="include", reason="dup"))
    with pytest.raises(Exception) as ei:
        ResearchReport(
            considered=considered,
            methods=[_candidate(i) for i in sorted(includes)],
            **_base_kwargs(),
        )
    assert "duplicate" in str(ei.value)


def test_unknown_considered_id_raises():
    includes = {"awq", "gptq", "bnb_nf4"}
    considered = _full_considered(includes)
    considered.append(
        ConsideredMethod(id="totally_made_up", verdict="reject", reason="x")
    )
    with pytest.raises(Exception) as ei:
        ResearchReport(
            considered=considered,
            methods=[_candidate(i) for i in sorted(includes)],
            **_base_kwargs(),
        )
    assert "not in the catalog" in str(ei.value)
