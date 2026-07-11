"""Typed state and bounded artifacts for the staged Adapt pipeline."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from .io_utils import atomic_write_text


class AdaptPlan(BaseModel):
    """Repository-derived plan produced before anything is installed."""

    install_steps: list[str] = Field(default_factory=list, max_length=12)
    entrypoint: str | None = None
    script_style: Literal["standalone", "wrapper"]
    evidence_files: list[str] = Field(default_factory=list, max_length=12)
    notes: str = ""

    @field_validator("install_steps")
    @classmethod
    def _nonempty_steps(cls, value: list[str]) -> list[str]:
        if any(not isinstance(step, str) or not step.strip() for step in value):
            raise ValueError("install_steps must contain non-empty strings")
        return value

    @field_validator("entrypoint")
    @classmethod
    def _relative_entrypoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        p = Path(value)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("entrypoint must be relative to the cloned repository")
        return value


class AdaptStageRecord(BaseModel):
    name: Literal[
        "prepare", "acquire", "plan", "environment", "architecture",
        "generate", "validate", "promote",
    ]
    status: Literal["started", "completed", "failed"]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    detail: dict = Field(default_factory=dict)


class AdaptTrace(BaseModel):
    schema_version: int = 1
    model_id: str
    method_id: str
    stages: list[AdaptStageRecord] = Field(default_factory=list)

    def record(self, name: str, status: str, **detail) -> None:
        self.stages.append(AdaptStageRecord(name=name, status=status, detail=detail))

    def persist(self, path: Path) -> None:
        atomic_write_text(path, self.model_dump_json(indent=2))


class AdaptPlanSession:
    """Single-assignment sink used by the repository-planning agent."""

    def __init__(self) -> None:
        self.plan: AdaptPlan | None = None

    def write(self, **payload) -> dict:
        if self.plan is not None:
            return {"status": "error", "message": "adapt plan is already finalized"}
        try:
            self.plan = AdaptPlan(**payload)
        except Exception as exc:  # Pydantic exposes useful field-specific errors
            return {"status": "error", "message": str(exc)}
        return {"status": "ok", "plan": self.plan.model_dump()}


def make_write_adapt_plan_tool(session: AdaptPlanSession):
    @tool
    def write_adapt_plan(
        install_steps: list[str],
        script_style: Literal["standalone", "wrapper"],
        entrypoint: str | None = None,
        evidence_files: list[str] | None = None,
        notes: str = "",
    ) -> str:
        """Finalize the repository-derived Adapt plan.

        install_steps must be commands accepted by install_method_venv; entrypoint
        is relative to the cloned repository. Call exactly once after inspecting
        the README and relevant source files.
        """
        return json.dumps(session.write(
            install_steps=install_steps,
            script_style=script_style,
            entrypoint=entrypoint,
            evidence_files=evidence_files or [],
            notes=notes,
        ), indent=2)

    return write_adapt_plan
