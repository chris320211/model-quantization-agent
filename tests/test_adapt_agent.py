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
    return SimpleNamespace(
        output_dir=tmp_path, model_override=None, openai_api_key="test"
    )


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
    monkeypatch.setattr(adapt_agent, "create_chat_model", lambda *args, **kwargs: object())
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
    monkeypatch.setattr(adapt_agent, "create_chat_model", lambda *args, **kwargs: object())
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


def test_port_required_method_creates_separate_overlay_and_contract_script(tmp_path, monkeypatch):
    monkeypatch.setattr(adapt_agent, "load_settings", lambda: _settings(tmp_path))
    model_stages = []
    monkeypatch.setattr(
        adapt_agent, "create_chat_model",
        lambda stage, *args, **kwargs: model_stages.append(stage) or object(),
    )
    _mock_stages(monkeypatch)
    method = _method().model_copy(update={
        "requires_port": True,
        "port_reason": "Qwen2 is not documented upstream",
    })
    patch = (
        "diff --git a/awq/modeling.py b/awq/modeling.py\n"
        "--- a/awq/modeling.py\n+++ b/awq/modeling.py\n"
        "@@ -1 +1,2 @@\n SUPPORTED = ['llama']\n+SUPPORTED.append('qwen2')\n"
    )

    def factory(llm, tools, prompt):
        names = {tool.name for tool in tools}
        if "write_adapt_plan" in names:
            writer = next(tool for tool in tools if tool.name == "write_adapt_plan")

            class PlanAgent:
                def invoke(self, *args, **kwargs):
                    writer.invoke({
                        "install_steps": [], "script_style": "wrapper",
                        "entrypoint": "examples/quantize.py",
                        "evidence_files": ["README.md", "awq/modeling.py"],
                    })
            return PlanAgent()

        if "write_port_overlay" in names:
            writer = next(tool for tool in tools if tool.name == "write_port_overlay")

            class PortAgent:
                def invoke(self, *args, **kwargs):
                    writer.invoke({
                        "patch": patch,
                        "rationale": "Add Qwen2 architecture dispatch.",
                        "evidence_files": ["awq/modeling.py"],
                        "target_modules": ["model.layers.*.self_attn.q_proj"],
                    })
            return PortAgent()

        writer = next(tool for tool in tools if tool.name == "write_script")
        target = re.search(r"Output script: (.+)", prompt).group(1).strip()
        overlay = re.search(r"Overlay directory: (.+)", prompt).group(1).strip()
        code = (
            f"# QUANT_AGENT_OVERLAY_DIR={overlay}\n"
            "import os\n"
            "MODEL_ID = 'org/model'\n"
            "OUTPUT_DIR = './quantized/awq-org__model'\n"
            f"OVERLAY = os.environ.get('QUANT_AGENT_OVERLAY_DIR', '{overlay}')\n"
            "REPO = os.environ['QUANT_AGENT_METHOD_REPO']\n"
        )

        class AuthorAgent:
            def invoke(self, *args, **kwargs):
                writer.invoke({"path": target, "code": code})
        return AuthorAgent()

    monkeypatch.setattr(adapt_agent, "create_react_agent", factory)
    path, code = adapt_agent.run("org/model", method)
    match = re.search(r"^# QUANT_AGENT_OVERLAY_DIR=(.+)$", code, re.MULTILINE)
    assert match
    overlay = Path(match.group(1))
    assert overlay.is_dir()
    assert (overlay / "overlay.patch").read_text() == patch
    assert model_stages == [
        adapt_agent.AgentStage.PLAN,
        adapt_agent.AgentStage.PORT,
        adapt_agent.AgentStage.AUTHOR,
    ]
    trace = json.loads(Path(path).with_suffix(".adapt.json").read_text())
    assert any(stage["name"] == "port" and stage["status"] == "completed" for stage in trace["stages"])
