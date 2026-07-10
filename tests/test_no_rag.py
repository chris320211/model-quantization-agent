"""Invariants for the RAG removal + Adapt refactor.

Guards against regressions that re-introduce the vector-RAG stack or the cloud
credentials it required.
"""
from __future__ import annotations

import importlib

import pytest

from quant_agent import adapt_agent, config, research_agent


def test_settings_has_no_rag_credential_fields():
    fields = set(config.Settings.__dataclass_fields__)
    for gone in (
        "voyage_api_key",
        "qdrant_url",
        "qdrant_api_key",
        "r2_account_id",
        "r2_access_key_id",
        "r2_secret_access_key",
        "r2_bucket_name",
    ):
        assert gone not in fields, f"{gone} should have been removed from Settings"
    assert {"anthropic_api_key", "model", "seed_path", "output_dir", "github_token", "hf_token"} <= fields


def test_load_settings_requires_only_anthropic(monkeypatch):
    config.load_settings.cache_clear()
    for k in (
        "VOYAGE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY",
        "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    try:
        s = config.load_settings()
        assert s.anthropic_api_key == "sk-ant-test-key"
    finally:
        config.load_settings.cache_clear()


def test_rag_modules_are_gone():
    for mod in ("quant_agent.tools.rag", "quant_agent.retrieval", "quant_agent.ingest"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)


def test_research_prompt_is_catalog_only():
    assert "{rag}" not in research_agent._PROMPT
    assert "Retrieved literature" not in research_agent._PROMPT
    assert not hasattr(research_agent, "_fan_out_rag")


def test_adapt_arxiv_id_resolver():
    # Unknown id -> None; a known catalog id -> str or None, never raises.
    assert adapt_agent._catalog_arxiv_id("definitely-not-a-real-method-id") is None
    assert adapt_agent._catalog_arxiv_id("awq") is None or isinstance(
        adapt_agent._catalog_arxiv_id("awq"), str
    )
