from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MethodCandidate(BaseModel):
    id: str = Field(..., description="Catalog id from seed/methods.yaml, e.g. 'awq'.")
    name: str
    repo_url: str = Field(..., description="Canonical repo URL (methods.yaml repos[0]).")
    bits: int = Field(..., description="Target weight bit width for this candidate.")
    est_vram_gb: float = Field(..., description="Estimated weight-memory footprint in GB.")
    quality_score: int = Field(..., ge=0, le=5, description="Perplexity retention 0-5.")
    speed_score: int = Field(..., ge=0, le=5, description="Kernel-level speedup 0-5.")
    needs_calibration: bool
    summary: str = Field(..., description="2-3 sentence why/when for this method.")


class ConsideredMethod(BaseModel):
    """One row in the 34-way walk the Research agent performs over the catalog."""
    id: str = Field(..., description="Catalog id from seed/methods.yaml.")
    verdict: Literal["include", "reject"]
    reason: str = Field(
        ...,
        description="One line citing architecture, GPU/compute-capability, VRAM, "
        "or bit-width fit — grounded in the retrieved literature.",
    )


class ResolvedInputs(BaseModel):
    resolved_model_id: str
    params_b: float | None = None
    instance_type: str | None = None
    vram_gb: float | None = None
    compute_capability: float | None = None
    gpu_arch: str | None = None


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
        description="One paragraph comparing the finalists, grounded in retrieved chunks.",
    )
