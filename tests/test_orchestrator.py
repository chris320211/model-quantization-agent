from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from quant_agent import orchestrator
from quant_agent.schemas import MethodCandidate, ResearchReport


def _fixture_report() -> ResearchReport:
    return ResearchReport(
        resolved_model_id="meta-llama/Llama-2-7b-hf",
        params_b=6.74,
        instance_type="g5.xlarge",
        vram_gb=24.0,
        methods=[
            MethodCandidate(
                id="awq",
                name="AWQ",
                repo_url="https://github.com/casper-hansen/AutoAWQ",
                bits=4,
                est_vram_gb=4.7,
                quality_score=5,
                speed_score=4,
                needs_calibration=True,
                summary="AWQ summary",
            ),
            MethodCandidate(
                id="gptq",
                name="GPTQ",
                repo_url="https://github.com/ModelCloud/GPTQModel",
                bits=4,
                est_vram_gb=4.7,
                quality_score=4,
                speed_score=4,
                needs_calibration=True,
                summary="GPTQ summary",
            ),
            MethodCandidate(
                id="hqq",
                name="HQQ",
                repo_url="https://github.com/mobiusml/hqq",
                bits=4,
                est_vram_gb=4.9,
                quality_score=4,
                speed_score=3,
                needs_calibration=False,
                summary="HQQ summary",
            ),
        ],
        tradeoffs="AWQ vs GPTQ vs HQQ paragraph",
    )


def test_format_report_includes_methods_and_tradeoffs():
    r = _fixture_report()
    out = orchestrator.format_report(r)
    assert "Llama-2-7b" in out
    assert "g5.xlarge" in out
    assert "24 GB" in out
    assert "1. AWQ" in out
    assert "2. GPTQ" in out
    assert "3. HQQ" in out
    assert "Tradeoffs:" in out


def test_run_invokes_adapt_and_execute_with_second_choice(monkeypatch):
    report = _fixture_report()
    seen: dict = {}

    def fake_research_run(user_input: str):
        seen["input"] = user_input
        return report

    def fake_adapt_run(model_id, method):
        seen["adapt_model_id"] = model_id
        seen["adapt_method"] = method
        return ("/tmp/out/quantize_x_gptq.py", "import gptqmodel\n")

    fake_stdin = io.StringIO("2\n")
    monkeypatch.setattr(orchestrator.sys, "stdin", fake_stdin)
    monkeypatch.setattr(orchestrator.research_agent, "run", fake_research_run)
    monkeypatch.setattr(orchestrator.adapt_agent, "run", fake_adapt_run)

    execute_payload = json.dumps(
        {"job_id": "JOB1", "pid": 1234, "status": "running"}
    )
    exec_invoke = MagicMock(return_value=execute_payload)
    fake_tool = SimpleNamespace(invoke=exec_invoke)
    monkeypatch.setattr(orchestrator, "execute_quantization", fake_tool)

    result = orchestrator.run("port llama2 7b to g5.xlarge")

    assert seen["input"] == "port llama2 7b to g5.xlarge"
    assert seen["adapt_method"].id == "gptq"
    exec_invoke.assert_called_once()
    kwargs = exec_invoke.call_args.args[0]
    assert kwargs["method_id"] == "gptq"
    assert kwargs["model_id"] == "meta-llama/Llama-2-7b-hf"
    assert kwargs["script_code"] == "import gptqmodel\n"
    assert "Job launched: JOB1" in result
    assert "/tmp/out/quantize_x_gptq.py" in result


def test_run_dry_skips_execute(monkeypatch):
    report = _fixture_report()

    monkeypatch.setattr(orchestrator.sys, "stdin", io.StringIO("1\n"))
    monkeypatch.setattr(orchestrator.research_agent, "run", lambda _: report)
    monkeypatch.setattr(
        orchestrator.adapt_agent,
        "run",
        lambda model_id, method: ("/tmp/out/script.py", "code"),
    )

    exec_invoke = MagicMock()
    monkeypatch.setattr(
        orchestrator, "execute_quantization", SimpleNamespace(invoke=exec_invoke)
    )

    result = orchestrator.run("whatever", dry=True)

    exec_invoke.assert_not_called()
    assert "--dry" in result
    assert "/tmp/out/script.py" in result


def test_run_aborts_on_q(monkeypatch):
    report = _fixture_report()
    monkeypatch.setattr(orchestrator.sys, "stdin", io.StringIO("q\n"))
    monkeypatch.setattr(orchestrator.research_agent, "run", lambda _: report)

    adapt_mock = MagicMock()
    monkeypatch.setattr(orchestrator.adapt_agent, "run", adapt_mock)

    result = orchestrator.run("whatever")

    adapt_mock.assert_not_called()
    assert result == "aborted"


def test_research_fixture_ids_are_in_catalog():
    """Guard: every fixture MethodCandidate.id must exist in the real catalog."""
    from quant_agent.tools.recommender import load_catalog

    ids = {m["id"] for m in load_catalog()}
    for mc in _fixture_report().methods:
        assert mc.id in ids
