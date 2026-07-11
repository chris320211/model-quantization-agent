from __future__ import annotations

from typing import Any, Literal
import math

from pydantic import BaseModel, Field, model_validator


class MethodCandidate(BaseModel):
    id: str = Field(..., description="Packaged catalog id, e.g. 'awq'.")
    name: str
    repo_url: str = Field(..., description="Canonical repo URL (methods.yaml repos[0]).")
    bits: int = Field(..., gt=0, description="Target weight bit width for this candidate.")
    est_vram_gb: float = Field(
        ..., ge=0, allow_inf_nan=False, description="Estimated weight-memory footprint in GB."
    )
    quality_score: int = Field(..., ge=0, le=5, description="Perplexity retention 0-5.")
    speed_score: int = Field(..., ge=0, le=5, description="Kernel-level speedup 0-5.")
    needs_calibration: bool
    summary: str = Field(..., description="2-3 sentence why/when for this method.")
    hyperparameters: dict[str, Any] | None = Field(
        None,
        description="Method-specific hyperparameter values "
        "(e.g. {'group_size': 128, 'sym': true}). Populated by the tune loop or "
        "by the research agent when ranges are known. None means use method defaults.",
    )


class ConsideredMethod(BaseModel):
    """One row in the 34-way walk the Research agent performs over the catalog."""
    id: str = Field(..., description="Packaged catalog id.")
    verdict: Literal["include", "reject"]
    reason: str = Field(
        ...,
        description="One line citing architecture, GPU/compute-capability, VRAM, "
        "or bit-width fit — grounded in the catalog fields + hf_info.",
    )


class ResolvedInputs(BaseModel):
    resolved_model_id: str
    params_b: float | None = None
    instance_type: str | None = None
    vram_gb: float | None = None
    compute_capability: float | None = None
    gpu_arch: str | None = None
    memory_bandwidth_gb_s: float | None = None
    peak_fp16_tflops: float | None = None
    int8_tops: float | None = None


class ResearchReport(ResolvedInputs):
    considered: list[ConsideredMethod] = Field(
        ...,
        description="Per-catalog-method verdict (include or reject) with a one-line reason. "
        "One entry per catalog id — the full walk.",
    )
    methods: list[MethodCandidate] = Field(
        ...,
        min_length=3,
        max_length=8,
        description="3-8 finalists drawn from the 'include' verdicts above; do NOT pick a winner.",
    )
    tradeoffs: str = Field(
        ...,
        description="One paragraph comparing the finalists, grounded in the catalog + model/GPU facts.",
    )

    @model_validator(mode="after")
    def _check_catalog_walk(self) -> "ResearchReport":
        # Local import to avoid schemas → tools → config → schemas cycles at import time.
        from .tools.recommender import load_catalog

        catalog = load_catalog()
        catalog_ids = [m["id"] for m in catalog]
        catalog_by_id = {m["id"]: m for m in catalog}
        catalog_set = set(catalog_ids)

        considered_ids = [c.id for c in self.considered]
        considered_set = set(considered_ids)

        missing = catalog_set - considered_set
        extra = considered_set - catalog_set
        dup_count = len(considered_ids) - len(considered_set)

        problems: list[str] = []
        if missing:
            problems.append(f"considered is missing catalog ids: {sorted(missing)}")
        if extra:
            problems.append(f"considered has ids not in the catalog: {sorted(extra)}")
        if dup_count:
            problems.append(f"considered has {dup_count} duplicate id(s)")

        include_set = {c.id for c in self.considered if c.verdict == "include"}
        bad_finalists = [m.id for m in self.methods if m.id not in include_set]
        if bad_finalists:
            problems.append(
                f"methods contain ids without an 'include' verdict: {bad_finalists}"
            )

        not_in_catalog = [m.id for m in self.methods if m.id not in catalog_set]
        if not_in_catalog:
            problems.append(f"methods contain ids not in the catalog: {not_in_catalog}")

        method_ids = [m.id for m in self.methods]
        if len(method_ids) != len(set(method_ids)):
            problems.append("methods contain duplicate finalist ids")

        for candidate in self.methods:
            entry = catalog_by_id.get(candidate.id)
            if entry is None:
                continue
            canonical_repo = (entry.get("repos") or [None])[0]
            expected = {
                "name": entry.get("name"),
                "repo_url": canonical_repo,
                "quality_score": entry.get("quality", 0),
                "speed_score": entry.get("speedup", 0),
                "needs_calibration": bool(entry.get("needs_calibration", False)),
            }
            for field, value in expected.items():
                if getattr(candidate, field) != value:
                    problems.append(
                        f"candidate {candidate.id!r} has non-catalog {field}: "
                        f"{getattr(candidate, field)!r} != {value!r}"
                    )
            if candidate.bits not in (entry.get("bits") or []):
                problems.append(
                    f"candidate {candidate.id!r} uses unsupported bits={candidate.bits}"
                )
            if not math.isfinite(candidate.est_vram_gb):
                problems.append(f"candidate {candidate.id!r} has non-finite est_vram_gb")
            if self.params_b is not None:
                expected_vram = self.params_b * candidate.bits / 8 * 1.4
                if not math.isclose(candidate.est_vram_gb, expected_vram, abs_tol=0.05):
                    problems.append(
                        f"candidate {candidate.id!r} est_vram_gb={candidate.est_vram_gb:g} "
                        f"does not match {expected_vram:g}"
                    )
            if candidate.hyperparameters is not None:
                specs = entry.get("hyperparameters_default") or {}
                for name, value in candidate.hyperparameters.items():
                    if name not in specs:
                        problems.append(
                            f"candidate {candidate.id!r} has unknown hyperparameter {name!r}"
                        )
                    elif value not in specs[name].get("values", []):
                        problems.append(
                            f"candidate {candidate.id!r} has invalid {name}={value!r}"
                        )

        if problems:
            raise ValueError("; ".join(problems))
        return self
