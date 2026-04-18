from __future__ import annotations

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


class ResolvedInputs(BaseModel):
    resolved_model_id: str
    params_b: float | None = None
    instance_type: str | None = None
    vram_gb: float | None = None


class ResearchReport(ResolvedInputs):
    methods: list[MethodCandidate] = Field(
        ...,
        min_length=3,
        max_length=8,
        description="3-8 candidate methods; do NOT pick a winner.",
    )
    tradeoffs: str = Field(
        ...,
        description="One paragraph comparing the surfaced methods.",
    )
