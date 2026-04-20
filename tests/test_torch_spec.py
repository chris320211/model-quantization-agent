from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_agent.tools import torch_spec


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("QUANT_AGENT_TORCH_SPEC", raising=False)


def _fake_smi(cc: str):
    def _run(*a, **kw):
        return SimpleNamespace(returncode=0, stdout=f"{cc}\n", stderr="")
    return _run


def test_hopper_picks_cu124(monkeypatch):
    monkeypatch.setattr(torch_spec.shutil, "which", lambda cmd: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(torch_spec.subprocess, "run", _fake_smi("9.0"))
    spec = torch_spec.detect_torch_spec()
    assert spec.cuda_tag == "cu124"
    assert spec.torch_pin == "torch==2.4.1"
    assert "cu124" in spec.index_url


def test_ampere_stays_on_cu121(monkeypatch):
    monkeypatch.setattr(torch_spec.shutil, "which", lambda cmd: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(torch_spec.subprocess, "run", _fake_smi("8.6"))
    spec = torch_spec.detect_torch_spec()
    assert spec.cuda_tag == "cu121"
    assert spec.torch_pin == "torch==2.3.1"


def test_no_nvidia_smi_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(torch_spec.shutil, "which", lambda cmd: None)
    spec = torch_spec.detect_torch_spec()
    assert spec.cuda_tag == "cu121"


def test_env_override_wins_over_detection(monkeypatch):
    monkeypatch.setenv("QUANT_AGENT_TORCH_SPEC", "torch==2.5.0|cu125")
    monkeypatch.setattr(torch_spec.shutil, "which", lambda cmd: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(torch_spec.subprocess, "run", _fake_smi("9.0"))
    spec = torch_spec.detect_torch_spec()
    assert spec.torch_pin == "torch==2.5.0"
    assert spec.cuda_tag == "cu125"


def test_malformed_env_override_is_ignored(monkeypatch):
    monkeypatch.setenv("QUANT_AGENT_TORCH_SPEC", "garbage-no-pipe")
    monkeypatch.setattr(torch_spec.shutil, "which", lambda cmd: None)
    spec = torch_spec.detect_torch_spec()
    assert spec.cuda_tag == "cu121"


def test_pip_install_string_includes_index_url():
    spec = torch_spec.TorchSpec(torch_pin="torch==2.3.1", cuda_tag="cu121")
    cmd = spec.pip_install()
    assert "--index-url https://download.pytorch.org/whl/cu121" in cmd
    assert "torch==2.3.1" in cmd
