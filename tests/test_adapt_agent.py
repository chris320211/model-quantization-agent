from __future__ import annotations

import re
import json
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


def _mock_stages(monkeypatch):
    monkeypatch.setattr(
        adapt_agent, "clone_method_repo",
        SimpleNamespace(invoke=lambda _: json.dumps({
            "status": "ok", "commit_sha": "abc123", "already_present": False,
        })),
    )
    monkeypatch.setattr(
        adapt_agent, "install_method_venv",
        SimpleNamespace(invoke=lambda _: json.dumps({
            "status": "ok", "python": "/tmp/venv/bin/python",
        })),
    )
    monkeypatch.setattr(
        adapt_agent, "fetch_model_config",
        SimpleNamespace(invoke=lambda _: json.dumps({
            "model_id": "org/model", "architectures": ["LlamaForCausalLM"],
            "trust_remote_code_required": False,
        })),
    )
    monkeypatch.setattr(
        adapt_agent, "inspect_architecture_core",
        lambda *a, **k: json.dumps({"status": "ok", "modules": []}),
    )


def test_adapt_atomically_promotes_current_validated_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(adapt_agent, "load_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(adapt_agent, "ChatAnthropic", lambda **kwargs: object())
    _mock_stages(monkeypatch)

    def factory(llm, tools, prompt):
        names = {tool.name for tool in tools}
        if "write_adapt_plan" in names:
            planner = next(tool for tool in tools if tool.name == "write_adapt_plan")

            class PlanAgent:
                def invoke(self, *args, **kwargs):
                    planner.invoke({
                        "install_steps": ["pip install -e ."],
                        "script_style": "standalone",
                        "entrypoint": None,
                        "evidence_files": ["README.md"],
                    })

            return PlanAgent()

        match = re.search(r"Output script: (.+)", prompt)
        assert match
        target = match.group(1).strip()
        writer = next(tool for tool in tools if tool.name == "write_script")

        class Agent:
            def invoke(self, *args, **kwargs):
                writer.invoke({
                    "path": target,
                    "code": "MODEL_ID = 'org/model'\nOUTPUT_DIR = './quantized/awq-org__model'\nprint('validated')\n",
                })

        return Agent()

    monkeypatch.setattr(adapt_agent, "create_react_agent", factory)
    path, code = adapt_agent.run("org/model", _method())
    assert "print('validated')" in code
    assert Path(path).exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert Path(path).with_suffix(".adapt.json").exists()


def test_adapt_cannot_reuse_stale_stable_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(adapt_agent, "load_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(adapt_agent, "ChatAnthropic", lambda **kwargs: object())
    _mock_stages(monkeypatch)
    stable = tmp_path / "quantize_org_model_awq.py"
    stable.write_text("print('stale')\n")

    def factory(llm, tools, prompt):
        names = {tool.name for tool in tools}
        if "write_adapt_plan" in names:
            planner = next(tool for tool in tools if tool.name == "write_adapt_plan")

            class PlanAgent:
                def invoke(self, *args, **kwargs):
                    planner.invoke({"install_steps": [], "script_style": "standalone"})

            return PlanAgent()

        class AuthorAgent:
            def invoke(self, *args, **kwargs):
                return None

        return AuthorAgent()

    monkeypatch.setattr(adapt_agent, "create_react_agent", factory)
    with pytest.raises(RuntimeError, match="validated artifact"):
        adapt_agent.run("org/model", _method())
    assert stable.read_text() == "print('stale')\n"
