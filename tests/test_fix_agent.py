from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from quant_agent import fix_agent
from quant_agent.executor import JobMeta
from quant_agent.schemas import MethodCandidate


def _method() -> MethodCandidate:
    return MethodCandidate(
        id="awq",
        name="AWQ",
        repo_url="https://github.com/casper-hansen/AutoAWQ",
        bits=4,
        est_vram_gb=4.7,
        quality_score=5,
        speed_score=4,
        needs_calibration=True,
        summary="x",
    )


def _meta(job_id: str = "JOB1") -> JobMeta:
    return JobMeta(
        job_id=job_id,
        method_id="awq",
        model_id="meta-llama/Llama-3.2-3B",
        venv="awq",
        script_path=f"/tmp/jobs/{job_id}/script.py",
        output_dir="/tmp/out",
        pid=1,
        started_at="2026-04-19T00:00:00+00:00",
        status="failed",
        exit_code=1,
    )


def test_run_returns_new_job_id_on_successful_relaunch(monkeypatch):
    """When the ReAct loop emits a relaunch_job tool message with status=ok, run() returns the new id."""
    fake_tool_message = SimpleNamespace(
        name="relaunch_job",
        content=json.dumps({"status": "ok", "new_job_id": "JOB2", "parent_job_id": "JOB1"}),
    )
    final_state = {"messages": [fake_tool_message]}

    fake_agent = SimpleNamespace(invoke=MagicMock(return_value=final_state))
    monkeypatch.setattr(fix_agent, "create_react_agent", lambda *a, **kw: fake_agent)
    monkeypatch.setattr(fix_agent, "ChatAnthropic", lambda **kw: MagicMock())
    monkeypatch.setattr(fix_agent.executor, "refresh_status", lambda jid: _meta(jid))

    new_id = fix_agent.run(
        job_id="JOB1",
        method=_method(),
        model_id="meta-llama/Llama-3.2-3B",
        attempt=1,
        max_attempts=3,
    )
    assert new_id == "JOB2"


def test_run_returns_none_when_agent_does_not_relaunch(monkeypatch):
    """Non-retryable errors: the agent reasons and stops without calling relaunch_job."""
    final_state = {"messages": [SimpleNamespace(name=None, content="I give up")]}
    fake_agent = SimpleNamespace(invoke=MagicMock(return_value=final_state))
    monkeypatch.setattr(fix_agent, "create_react_agent", lambda *a, **kw: fake_agent)
    monkeypatch.setattr(fix_agent, "ChatAnthropic", lambda **kw: MagicMock())
    monkeypatch.setattr(fix_agent.executor, "refresh_status", lambda jid: _meta(jid))

    new_id = fix_agent.run(
        job_id="JOB1",
        method=_method(),
        model_id="m",
        attempt=1,
        max_attempts=3,
    )
    assert new_id is None


def test_run_returns_none_when_relaunch_tool_errored(monkeypatch):
    """relaunch_job returned status=error — don't treat it as a successful relaunch."""
    fake_tool_message = SimpleNamespace(
        name="relaunch_job",
        content=json.dumps({"status": "error", "error": "boom"}),
    )
    final_state = {"messages": [fake_tool_message]}
    fake_agent = SimpleNamespace(invoke=MagicMock(return_value=final_state))
    monkeypatch.setattr(fix_agent, "create_react_agent", lambda *a, **kw: fake_agent)
    monkeypatch.setattr(fix_agent, "ChatAnthropic", lambda **kw: MagicMock())
    monkeypatch.setattr(fix_agent.executor, "refresh_status", lambda jid: _meta(jid))

    new_id = fix_agent.run(
        job_id="JOB1",
        method=_method(),
        model_id="m",
        attempt=1,
        max_attempts=3,
    )
    assert new_id is None


def test_extract_new_job_id_picks_latest_relaunch(monkeypatch):
    """If the agent called relaunch_job twice (second succeeded), pick the latest ok payload."""
    messages = [
        SimpleNamespace(
            name="relaunch_job",
            content=json.dumps({"status": "error", "error": "first try failed"}),
        ),
        SimpleNamespace(
            name="relaunch_job",
            content=json.dumps({"status": "ok", "new_job_id": "JOB3"}),
        ),
    ]
    assert fix_agent._extract_new_job_id({"messages": messages}) == "JOB3"
