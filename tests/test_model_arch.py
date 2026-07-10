from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from quant_agent.tools import model_arch


def test_fetch_config_detects_auto_map(monkeypatch):
    cfg = {"architectures": ["FooForCausalLM"], "model_type": "foo", "auto_map": {"AutoModel": "m.Foo"}}
    monkeypatch.setattr(model_arch, "_load_config", lambda mid: cfg)
    out = model_arch.fetch_model_config_dict("org/foo")
    assert out["trust_remote_code_required"] is True
    assert out["config"] == cfg


def test_fetch_config_standard_arch(monkeypatch):
    monkeypatch.setattr(model_arch, "_load_config", lambda mid: {"architectures": ["LlamaForCausalLM"]})
    out = model_arch.fetch_model_config_dict("org/llama")
    assert out["trust_remote_code_required"] is False


def test_fetch_config_error_is_captured(monkeypatch):
    def boom(mid):
        raise RuntimeError("404 not found")

    monkeypatch.setattr(model_arch, "_load_config", boom)
    out = model_arch.fetch_model_config_dict("x")
    assert "error" in out


def test_inspect_config_only_when_trust_required(monkeypatch):
    monkeypatch.setattr(
        model_arch,
        "fetch_model_config_dict",
        lambda mid: {"model_id": mid, "architectures": ["X"], "config": {}, "trust_remote_code_required": True},
    )
    out = json.loads(model_arch.inspect_architecture_core("m", "awq", trust_remote_code=False))
    assert out["status"] == "config_only"
    assert "trust_remote_code" in out["reason"]


def test_inspect_config_only_when_venv_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        model_arch,
        "fetch_model_config_dict",
        lambda mid: {"model_id": mid, "architectures": ["X"], "config": {}, "trust_remote_code_required": False},
    )
    monkeypatch.setattr(model_arch, "venv_python", lambda mid: tmp_path / "nope")
    out = json.loads(model_arch.inspect_architecture_core("m", "awq"))
    assert out["status"] == "config_only"
    assert "venv not built" in out["reason"]


def test_inspect_ok_collapses_layers(monkeypatch, tmp_path):
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n")
    monkeypatch.setattr(
        model_arch,
        "fetch_model_config_dict",
        lambda mid: {"model_id": mid, "architectures": ["X"], "config": {}, "trust_remote_code_required": False},
    )
    monkeypatch.setattr(model_arch, "venv_python", lambda mid: py)
    rows = [
        {"name": "model.layers.0.self_attn.q_proj", "cls": "Linear", "in": 4096, "out": 4096},
        {"name": "model.layers.1.self_attn.q_proj", "cls": "Linear", "in": 4096, "out": 4096},
        {"name": "lm_head", "cls": "Linear", "in": 4096, "out": 32000},
    ]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="INTROSPECT_RESULT=" + json.dumps(rows), stderr="", returncode=0),
    )
    out = json.loads(model_arch.inspect_architecture_core("m", "awq"))
    assert out["status"] == "ok"
    assert out["num_leaf_linear_embedding"] == 3
    patterns = {m["pattern"]: m for m in out["modules"]}
    assert "model.layers.{i}.self_attn.q_proj" in patterns
    assert patterns["model.layers.{i}.self_attn.q_proj"]["count"] == 2
    assert "lm_head" in patterns


def test_inspect_meta_load_failure_falls_back(monkeypatch, tmp_path):
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n")
    monkeypatch.setattr(
        model_arch,
        "fetch_model_config_dict",
        lambda mid: {"model_id": mid, "architectures": ["X"], "config": {}, "trust_remote_code_required": False},
    )
    monkeypatch.setattr(model_arch, "venv_python", lambda mid: py)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="", stderr="ImportError: no module named x", returncode=1),
    )
    out = json.loads(model_arch.inspect_architecture_core("m", "awq"))
    assert out["status"] == "config_only"
    assert "stderr_tail" in out


def test_collapse_dedups_indexed_blocks():
    rows = [
        {"name": "a.0.w", "cls": "Linear", "in": 1, "out": 2},
        {"name": "a.1.w", "cls": "Linear", "in": 1, "out": 2},
        {"name": "head", "cls": "Linear", "in": 2, "out": 3},
    ]
    collapsed = model_arch._collapse(rows)
    assert len(collapsed) == 2
    assert collapsed[0]["pattern"] == "a.{i}.w"
    assert collapsed[0]["count"] == 2
