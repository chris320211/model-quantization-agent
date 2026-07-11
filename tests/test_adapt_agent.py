from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_agent import adapt_agent
from quant_agent.schemas import MethodCandidate


def _method() -> MethodCandidate:
    return MethodCandidate(
        id="awq",
        name="Activation-aware Weight Quantization",
        repo_url="https://github.com/mit-han-lab/llm-awq",
        bits=4,
        est_vram_gb=1,
        quality_score=5,
        speed_score=4,
        needs_calibration=True,
        summary="x",
    )


def _settings(tmp_path):
    return SimpleNamespace(output_dir=tmp_path, model="test", anthropic_api_key="test")


def test_adapt_atomically_promotes_current_validated_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(adapt_agent, "load_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(adapt_agent, "ChatAnthropic", lambda **kwargs: object())

    def factory(llm, tools, prompt):
        match = re.search(r"Output script path: (.+)", prompt)
        assert match
        target = match.group(1).strip()
        writer = next(tool for tool in tools if tool.name == "write_script")

        class Agent:
            def invoke(self, *args, **kwargs):
                writer.invoke({"path": target, "code": "print('validated')\n"})

        return Agent()

    monkeypatch.setattr(adapt_agent, "create_react_agent", factory)
    path, code = adapt_agent.run("org/model", _method())
    assert code == "print('validated')\n"
    assert Path(path).exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_adapt_cannot_reuse_stale_stable_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(adapt_agent, "load_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(adapt_agent, "ChatAnthropic", lambda **kwargs: object())
    stable = tmp_path / "quantize_org_model_awq.py"
    stable.write_text("print('stale')\n")

    class Agent:
        def invoke(self, *args, **kwargs):
            return None

    monkeypatch.setattr(adapt_agent, "create_react_agent", lambda *a, **k: Agent())
    with pytest.raises(RuntimeError, match="validated artifact"):
        adapt_agent.run("org/model", _method())
    assert stable.read_text() == "print('stale')\n"
