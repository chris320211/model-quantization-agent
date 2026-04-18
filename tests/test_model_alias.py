from __future__ import annotations

from unittest.mock import MagicMock, patch

from quant_agent.tools import model_alias


def test_alias_hit_exact():
    r = model_alias.resolve("llama2 7b")
    assert r.source == "alias"
    assert r.model_id == "meta-llama/Llama-2-7b-hf"
    assert r.candidates == ["meta-llama/Llama-2-7b-hf"]


def test_alias_hit_with_punctuation_and_case():
    r = model_alias.resolve("Llama-2  7B!!")
    assert r.source == "alias"
    assert r.model_id == "meta-llama/Llama-2-7b-hf"


def test_hf_search_fallback_on_unknown_name():
    fake_models = [MagicMock(modelId=f"org/Some-{i}") for i in range(3)]
    with patch.object(model_alias, "_hf_search", return_value=[m.modelId for m in fake_models]):
        r = model_alias.resolve("definitely-not-a-real-alias-xyz-123")
    assert r.source == "hf_search"
    assert len(r.candidates) == 3
    # Ambiguous → model_id stays None until user picks
    assert r.model_id is None


def test_hf_search_single_hit_auto_picks():
    with patch.object(model_alias, "_hf_search", return_value=["org/Only-One"]):
        r = model_alias.resolve("obscure-model-name")
    assert r.source == "hf_search"
    assert r.model_id == "org/Only-One"


def test_unresolved_when_alias_miss_and_empty_search():
    with patch.object(model_alias, "_hf_search", return_value=[]):
        r = model_alias.resolve("nonsense-xyz-9999")
    assert r.source == "unresolved"
    assert r.model_id is None
    assert r.candidates == []
