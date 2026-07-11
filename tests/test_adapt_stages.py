import json

from quant_agent.adapt_stages import AdaptPlan, AdaptPlanSession, AdaptTrace, make_write_adapt_plan_tool


def test_adapt_plan_rejects_escaping_entrypoint():
    try:
        AdaptPlan(install_steps=[], entrypoint="../evil.py", script_style="wrapper")
    except ValueError as exc:
        assert "relative" in str(exc)
    else:
        raise AssertionError("escaping entrypoint was accepted")


def test_plan_session_is_single_assignment():
    session = AdaptPlanSession()
    tool = make_write_adapt_plan_tool(session)
    first = json.loads(tool.invoke({
        "install_steps": ["pip install -e ."],
        "script_style": "standalone",
        "entrypoint": None,
    }))
    second = json.loads(tool.invoke({
        "install_steps": [], "script_style": "wrapper", "entrypoint": "x.py",
    }))
    assert first["status"] == "ok"
    assert second["status"] == "error"
    assert session.plan is not None


def test_trace_persists_typed_stage_records(tmp_path):
    trace = AdaptTrace(model_id="org/model", method_id="awq")
    trace.record("prepare", "completed", output="x.py")
    out = tmp_path / "trace.json"
    trace.persist(out)
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == 1
    assert payload["stages"][0]["name"] == "prepare"
