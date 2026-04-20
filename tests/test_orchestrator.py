from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from quant_agent import orchestrator
from quant_agent.executor import JobMeta
from quant_agent.schemas import ConsideredMethod, MethodCandidate, ResearchReport


def _make_meta(job_id: str, status: str, exit_code: int | None = None, attempt: int = 1, parent: str | None = None) -> JobMeta:
    return JobMeta(
        job_id=job_id,
        method_id="gptq",
        model_id="meta-llama/Llama-2-7b-hf",
        venv="gptq",
        script_path=f"/tmp/jobs/{job_id}/script.py",
        output_dir="/tmp/out",
        pid=1234,
        started_at="2026-04-19T00:00:00+00:00",
        status=status,
        exit_code=exit_code,
        attempt=attempt,
        parent_job_id=parent,
    )


def _fixture_report() -> ResearchReport:
    from quant_agent.tools.recommender import load_catalog

    include_ids = {"awq", "gptq", "bnb_nf4"}
    considered = []
    for m in load_catalog():
        mid = m["id"]
        if mid in include_ids:
            considered.append(
                ConsideredMethod(id=mid, verdict="include", reason="Llama supported; sm_86 OK.")
            )
        else:
            considered.append(
                ConsideredMethod(id=mid, verdict="reject", reason="skipped for fixture simplicity.")
            )
    return ResearchReport(
        resolved_model_id="meta-llama/Llama-2-7b-hf",
        params_b=6.74,
        instance_type="g5.xlarge",
        vram_gb=24.0,
        compute_capability=8.6,
        gpu_arch="Ampere",
        considered=considered,
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
                id="bnb_nf4",
                name="bitsandbytes NF4 (QLoRA)",
                repo_url="https://github.com/bitsandbytes-foundation/bitsandbytes",
                bits=4,
                est_vram_gb=4.9,
                quality_score=4,
                speed_score=3,
                needs_calibration=False,
                summary="NF4 summary",
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
    assert "Ampere" in out
    assert "sm_86" in out
    assert "Considered methods" in out
    assert "[INCLUDE]" in out
    assert "[reject " in out
    assert "fp8" in out
    assert "1. AWQ" in out
    assert "2. GPTQ" in out
    assert "bnb_nf4" in out
    assert "Tradeoffs:" in out


def test_run_invokes_adapt_and_execute_with_second_choice(monkeypatch):
    report = _fixture_report()
    seen: dict = {}

    def fake_research_run(user_input: str):
        seen["input"] = user_input
        return report

    def fake_adapt_run(model_id, method, previous_error=None):
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
    monkeypatch.setattr(
        orchestrator.executor, "wait_for_job", lambda jid: _make_meta(jid, "completed", 0)
    )

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
        lambda model_id, method, previous_error=None: ("/tmp/out/script.py", "code"),
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


def test_supervise_retries_until_success(monkeypatch):
    report = _fixture_report()
    chosen = report.methods[0]  # awq

    wait_calls: list[str] = []
    fake_metas = iter(
        [
            _make_meta("JOB1", "failed", exit_code=1, attempt=1),
            _make_meta("JOB2", "completed", exit_code=0, attempt=2, parent="JOB1"),
        ]
    )

    def fake_wait(job_id):
        wait_calls.append(job_id)
        return next(fake_metas)

    fix_calls: list[dict] = []

    def fake_fix(**kwargs):
        fix_calls.append(kwargs)
        return "JOB2"

    monkeypatch.setattr(orchestrator.executor, "wait_for_job", fake_wait)
    monkeypatch.setattr(orchestrator.fix_agent, "run", fake_fix)

    final, chain = orchestrator._supervise("JOB1", chosen, "meta-llama/Llama-2-7b-hf", max_repairs=3)

    assert [m.job_id for m in chain] == ["JOB1", "JOB2"]
    assert final.status == "completed"
    assert len(fix_calls) == 1
    assert fix_calls[0]["job_id"] == "JOB1"
    assert fix_calls[0]["attempt"] == 1


def test_supervise_gives_up_after_max_repairs(monkeypatch):
    report = _fixture_report()
    chosen = report.methods[0]

    counter = {"n": 0}

    def fake_wait(job_id):
        counter["n"] += 1
        return _make_meta(job_id, "failed", exit_code=1, attempt=counter["n"])

    def fake_fix(**kwargs):
        return f"JOB{kwargs['attempt'] + 1}"

    monkeypatch.setattr(orchestrator.executor, "wait_for_job", fake_wait)
    monkeypatch.setattr(orchestrator.fix_agent, "run", fake_fix)

    final, chain = orchestrator._supervise("JOB1", chosen, "model", max_repairs=3)

    # Initial job + 3 repair attempts = 4 metas in the chain.
    assert len(chain) == 4
    assert final.status == "failed"


def test_supervise_stops_on_non_retryable(monkeypatch):
    report = _fixture_report()
    chosen = report.methods[0]

    monkeypatch.setattr(
        orchestrator.executor, "wait_for_job",
        lambda jid: _make_meta(jid, "failed", exit_code=1),
    )
    monkeypatch.setattr(orchestrator.fix_agent, "run", lambda **kw: None)

    final, chain = orchestrator._supervise("JOB1", chosen, "model", max_repairs=3)

    assert len(chain) == 1
    assert final.status == "failed"


def test_supervise_skips_retry_on_killed(monkeypatch):
    report = _fixture_report()
    chosen = report.methods[0]

    monkeypatch.setattr(
        orchestrator.executor, "wait_for_job",
        lambda jid: _make_meta(jid, "killed"),
    )
    fix_mock = MagicMock()
    monkeypatch.setattr(orchestrator.fix_agent, "run", fix_mock)

    final, chain = orchestrator._supervise("JOB1", chosen, "model", max_repairs=3)

    fix_mock.assert_not_called()
    assert final.status == "killed"
    assert len(chain) == 1


def test_adapt_retry_succeeds_on_second_attempt(monkeypatch):
    report = _fixture_report()

    calls = {"n": 0, "seen_errors": []}

    def flaky_adapt(model_id, method, previous_error=None):
        calls["n"] += 1
        calls["seen_errors"].append(previous_error)
        if calls["n"] == 1:
            raise RuntimeError("install_method_venv failed")
        return ("/tmp/out/script.py", "code")

    monkeypatch.setattr(orchestrator.sys, "stdin", io.StringIO("1\n"))
    monkeypatch.setattr(orchestrator.research_agent, "run", lambda _: report)
    monkeypatch.setattr(orchestrator.adapt_agent, "run", flaky_adapt)

    monkeypatch.setattr(
        orchestrator,
        "execute_quantization",
        SimpleNamespace(
            invoke=MagicMock(return_value=json.dumps({"job_id": "JOB1", "pid": 1, "status": "running"}))
        ),
    )
    monkeypatch.setattr(
        orchestrator.executor, "wait_for_job", lambda jid: _make_meta(jid, "completed", 0)
    )

    result = orchestrator.run("whatever", max_adapt_retries=2)

    assert calls["n"] == 2
    assert calls["seen_errors"][0] is None
    assert isinstance(calls["seen_errors"][1], RuntimeError)
    assert "Job launched: JOB1" in result


def test_adapt_retry_exhausted_falls_back_to_next_candidate(monkeypatch):
    report = _fixture_report()

    per_method_calls: dict[str, int] = {}

    def always_fails_on_first_method(model_id, method, previous_error=None):
        per_method_calls[method.id] = per_method_calls.get(method.id, 0) + 1
        if method.id == report.methods[0].id:
            raise RuntimeError(f"{method.id} broken")
        return ("/tmp/out/script.py", "code")

    monkeypatch.setattr(orchestrator.sys, "stdin", io.StringIO("1\n"))
    monkeypatch.setattr(orchestrator.research_agent, "run", lambda _: report)
    monkeypatch.setattr(orchestrator.adapt_agent, "run", always_fails_on_first_method)

    monkeypatch.setattr(
        orchestrator,
        "execute_quantization",
        SimpleNamespace(
            invoke=MagicMock(return_value=json.dumps({"job_id": "JOB2", "pid": 1, "status": "running"}))
        ),
    )
    monkeypatch.setattr(
        orchestrator.executor, "wait_for_job", lambda jid: _make_meta(jid, "completed", 0)
    )

    result = orchestrator.run("whatever", max_adapt_retries=2)

    # First method got 2 adapt attempts, then fell through to the second method.
    assert per_method_calls[report.methods[0].id] == 2
    assert per_method_calls[report.methods[1].id] == 1
    assert "Job launched: JOB2" in result


def test_run_with_max_repairs_zero_skips_supervise(monkeypatch):
    report = _fixture_report()

    monkeypatch.setattr(orchestrator.sys, "stdin", io.StringIO("1\n"))
    monkeypatch.setattr(orchestrator.research_agent, "run", lambda _: report)
    monkeypatch.setattr(
        orchestrator.adapt_agent,
        "run",
        lambda model_id, method, previous_error=None: ("/tmp/out/script.py", "code"),
    )

    execute_payload = json.dumps({"job_id": "JOB1", "pid": 1, "status": "running"})
    monkeypatch.setattr(
        orchestrator,
        "execute_quantization",
        SimpleNamespace(invoke=MagicMock(return_value=execute_payload)),
    )
    wait_mock = MagicMock()
    monkeypatch.setattr(orchestrator.executor, "wait_for_job", wait_mock)

    result = orchestrator.run("whatever", max_repairs=0)

    wait_mock.assert_not_called()
    assert "Job launched: JOB1" in result
