"""Central OpenAI model policy for each agent stage.

All model-backed stages use the Responses API and explicitly disable response
storage.  Stage defaults keep routine structured work on the balanced model and
reserve the flagship model for porting and runtime repair, where deeper reasoning
has the highest leverage.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .config import Settings, load_settings


class AgentStage(str, Enum):
    RESEARCH = "research"
    PLAN = "plan"
    AUTHOR = "author"
    PORT = "port"
    FIX = "fix"
    TUNE = "tune"
    HYPERPARAM = "hyperparam"


@dataclass(frozen=True)
class StageModelPolicy:
    model: str
    reasoning_effort: str


_DEFAULT_POLICIES: dict[AgentStage, StageModelPolicy] = {
    AgentStage.RESEARCH: StageModelPolicy("gpt-5.6-terra", "medium"),
    AgentStage.PLAN: StageModelPolicy("gpt-5.6-terra", "medium"),
    AgentStage.AUTHOR: StageModelPolicy("gpt-5.6-terra", "medium"),
    AgentStage.PORT: StageModelPolicy("gpt-5.6-sol", "high"),
    AgentStage.FIX: StageModelPolicy("gpt-5.6-sol", "high"),
    AgentStage.TUNE: StageModelPolicy("gpt-5.6-terra", "low"),
    AgentStage.HYPERPARAM: StageModelPolicy("gpt-5.6-terra", "low"),
}
_VALID_EFFORTS = {"low", "medium", "high"}


def resolve_model_policy(
    stage: AgentStage, settings: Settings | None = None
) -> StageModelPolicy:
    """Resolve stage-specific settings, with explicit environment overrides.

    Precedence is stage override, global override, then the checked-in default.
    The same precedence applies to reasoning effort.
    """
    s = settings or load_settings()
    prefix = f"QUANT_AGENT_{stage.value.upper()}"
    default = _DEFAULT_POLICIES[stage]
    model = (
        os.environ.get(f"{prefix}_MODEL")
        or s.model_override
        or default.model
    ).strip()
    effort = (
        os.environ.get(f"{prefix}_REASONING_EFFORT")
        or os.environ.get("QUANT_AGENT_REASONING_EFFORT")
        or default.reasoning_effort
    ).strip().lower()
    if not model:
        raise ValueError(f"empty model configured for {stage.value}")
    if effort not in _VALID_EFFORTS:
        raise ValueError(
            f"invalid reasoning effort {effort!r} for {stage.value}; "
            f"expected one of {sorted(_VALID_EFFORTS)}"
        )
    return StageModelPolicy(model=model, reasoning_effort=effort)


def _chat_openai_class() -> type[Any]:
    # Lazy import keeps catalog-only and deterministic commands usable before the
    # optional provider integration is imported by an LLM-backed stage.
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI model integration is not installed. Run "
            "`pip install -c constraints.txt -e .` in this checkout."
        ) from exc

    return ChatOpenAI


def create_chat_model(stage: AgentStage, settings: Settings | None = None) -> Any:
    """Build a LangChain OpenAI model pinned to Responses API semantics."""
    s = settings or load_settings()
    policy = resolve_model_policy(stage, s)
    return _chat_openai_class()(
        model=policy.model,
        api_key=s.openai_api_key,
        use_responses_api=True,
        output_version="responses/v1",
        reasoning={"effort": policy.reasoning_effort},
        store=False,
    )
