from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_agent import llm
from quant_agent.llm import AgentStage


def _settings(model_override: str | None = None):
    return SimpleNamespace(openai_api_key="sk-test", model_override=model_override)


def test_stage_defaults_reserve_flagship_for_port_and_fix(monkeypatch):
    for name in (
        "QUANT_AGENT_MODEL", "QUANT_AGENT_PORT_MODEL", "QUANT_AGENT_FIX_MODEL",
        "QUANT_AGENT_REASONING_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)
    assert llm.resolve_model_policy(AgentStage.RESEARCH, _settings()).model == "gpt-5.6-terra"
    assert llm.resolve_model_policy(AgentStage.PORT, _settings()).model == "gpt-5.6-sol"
    assert llm.resolve_model_policy(AgentStage.FIX, _settings()).reasoning_effort == "high"
    assert llm.resolve_model_policy(AgentStage.TUNE, _settings()).reasoning_effort == "low"


def test_stage_override_beats_global_override(monkeypatch):
    monkeypatch.setenv("QUANT_AGENT_MODEL", "global-model")
    monkeypatch.setenv("QUANT_AGENT_PORT_MODEL", "port-model")
    policy = llm.resolve_model_policy(AgentStage.PORT, _settings("settings-model"))
    assert policy.model == "port-model"


def test_invalid_reasoning_effort_fails_closed(monkeypatch):
    monkeypatch.setenv("QUANT_AGENT_FIX_REASONING_EFFORT", "maximum-ish")
    with pytest.raises(ValueError, match="invalid reasoning effort"):
        llm.resolve_model_policy(AgentStage.FIX, _settings())


def test_chat_model_uses_responses_api_without_storage(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm, "_chat_openai_class", lambda: FakeChatOpenAI)
    model = llm.create_chat_model(AgentStage.RESEARCH, _settings())
    assert isinstance(model, FakeChatOpenAI)
    assert captured["use_responses_api"] is True
    assert captured["output_version"] == "responses/v1"
    assert captured["store"] is False
    assert captured["reasoning"] == {"effort": "medium"}
