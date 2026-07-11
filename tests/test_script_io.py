from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quant_agent.tools import script_io


def _fake_run_ok(*args, **kwargs):
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _fake_run_import_err(*args, **kwargs):
    return SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="Traceback (most recent call last):\nModuleNotFoundError: No module named 'autowaq'\n",
    )


def _stub_venv(tmp_path, monkeypatch):
    fake_py = tmp_path / "fake_python"
    fake_py.write_text("#!/bin/sh\nexit 0\n")
    fake_py.chmod(0o755)
    monkeypatch.setattr(script_io, "venv_python", lambda method_id: fake_py)


def test_parse_error_returns_false(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    ok, stage, msg = script_io.validate("def broken(:\n  pass", "awq")
    assert ok is False
    assert stage == "parse"
    assert "line" in msg


def test_dry_import_failure_returns_false(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    with patch.object(subprocess, "run", _fake_run_import_err):
        ok, stage, msg = script_io.validate("import autowaq\n", "awq")
    assert ok is False
    assert stage == "dry-import"
    assert "autowaq" in msg


def test_successful_validation(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    with patch.object(subprocess, "run", _fake_run_ok):
        ok, stage, msg = script_io.validate("import os\nimport json\n", "awq")
    assert ok is True
    assert stage == "ok"


def test_typed_staged_result_records_passed_and_skipped_stages(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    report = script_io.validate_staged("MODEL = 'org/model'\n", "awq")
    assert isinstance(report, script_io.ValidationResult)
    assert report.ok is True
    assert report.stage is script_io.ValidationStage.OK
    assert [check.stage for check in report.checks] == [
        script_io.ValidationStage.PARSE,
        script_io.ValidationStage.DRY_IMPORT,
        script_io.ValidationStage.STATIC_SEMANTICS,
        script_io.ValidationStage.SMOKE,
    ]
    assert report.checks[2].status is script_io.CheckStatus.SKIPPED
    assert report.checks[3].status is script_io.CheckStatus.SKIPPED


def test_staged_validation_stops_after_syntax_failure(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    report = script_io.validate_staged("def broken(:\n", "awq")
    assert report.ok is False
    assert report.stage is script_io.ValidationStage.PARSE
    assert len(report.checks) == 1


def test_static_semantics_accept_exact_required_values(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    code = """
MODEL_ID = "org/model"
OUTPUT_DIR = "/runs/exact"
config = {"bits": 4, "group_size": 128, "symmetric": True}
"""
    report = script_io.validate_staged(
        code,
        "awq",
        expected_model_id="org/model",
        expected_output_dir=Path("/runs/exact"),
        locked_hyperparameters={"bits": 4, "group_size": 128, "symmetric": True},
    )
    assert report.ok is True
    semantic = report.checks[2]
    assert semantic.stage is script_io.ValidationStage.STATIC_SEMANTICS
    assert semantic.status is script_io.CheckStatus.PASSED


def test_static_semantics_accept_kwargs_and_assignments(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    code = """
model_id = "org/model"
output_dir = "/runs/exact"
group_size = 128
quantize(bits=4)
settings["damp_percent"] = 0.01
"""
    report = script_io.validate_staged(
        code,
        "awq",
        expected_model_id="org/model",
        expected_output_dir="/runs/exact",
        locked_hyperparameters={"bits": 4, "group_size": 128, "damp_percent": 0.01},
    )
    assert report.ok is True


def test_static_semantics_fail_closed_on_missing_or_wrong_values(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    report = script_io.validate_staged(
        'MODEL_ID = "other/model"\nOUTPUT_DIR = "/runs/not-exact"\nbits = 8\n',
        "awq",
        expected_model_id="org/model",
        expected_output_dir="/runs/exact",
        locked_hyperparameters={"bits": 4, "group_size": 128},
    )
    assert report.ok is False
    assert report.stage is script_io.ValidationStage.STATIC_SEMANTICS
    assert "model_id='org/model'" in report.message
    assert "output_dir='/runs/exact'" in report.message
    assert "locked bits=4" in report.message
    assert "locked group_size=128" in report.message
    assert all(check.stage is not script_io.ValidationStage.SMOKE for check in report.checks)


def test_locked_literal_comparison_does_not_confuse_bool_and_int(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    report = script_io.validate_staged(
        "bits = True\n",
        "awq",
        locked_hyperparameters={"bits": 1},
    )
    assert report.ok is False
    assert report.stage is script_io.ValidationStage.STATIC_SEMANTICS


def test_missing_venv_skips_dry_import(tmp_path, monkeypatch):
    # When the method venv hasn't been built, validate() trusts ast.parse and
    # skips dry-import instead of failing — the Adapt agent may be writing a
    # stdlib-only wrapper, and install_method_venv may not have run yet.
    monkeypatch.setattr(script_io, "venv_python", lambda method_id: tmp_path / "does_not_exist")
    ok, stage, msg = script_io.validate("import os\n", "awq")
    assert ok is True
    assert stage == "ok"
    assert "skipped" in msg


def test_session_writes_on_success(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    sess = script_io.ValidationSession(method_id="awq", allowed_root=tmp_path)
    out = tmp_path / "sub" / "script.py"
    with patch.object(subprocess, "run", _fake_run_ok):
        result = sess.write(str(out), "import os\n")
    assert result["status"] == "ok"
    assert result["attempts_left"] == 3  # unchanged on success
    assert out.read_text() == "import os\n"


def test_session_retry_on_error(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    sess = script_io.ValidationSession(method_id="awq", max_attempts=3, allowed_root=tmp_path)
    out = tmp_path / "script.py"
    with patch.object(subprocess, "run", _fake_run_import_err):
        r1 = sess.write(str(out), "import autowaq\n")
    assert r1["status"] == "error"
    assert r1["stage"] == "dry-import"
    assert r1["attempts_left"] == 2
    assert not out.exists()  # nothing written on recoverable failure


def test_session_static_failure_does_not_write(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    sess = script_io.ValidationSession(
        method_id="awq",
        max_attempts=1,
        allowed_root=tmp_path,
        expected_model_id="org/model",
        expected_output_dir=tmp_path / "quantized",
        locked_hyperparameters={"bits": 4},
    )
    out = tmp_path / "script.py"
    result = sess.write(
        str(out),
        f'MODEL_ID = "org/model"\nOUTPUT_DIR = "{tmp_path / "quantized"}"\nbits = 8\n',
    )
    assert result["status"] == "error-exhausted"
    assert result["stage"] == "static-semantics"
    assert result["validation"]["checks"][-1]["status"] == "failed"
    assert not out.exists()


def test_explicit_smoke_command_runs_temp_script_and_removes_it(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    seen: dict[str, object] = {}

    def smoke_ok(argv, **kwargs):
        candidate = Path(argv[1])
        seen["argv"] = argv
        seen["candidate"] = candidate
        seen["code"] = candidate.read_text()
        seen["timeout"] = kwargs["timeout"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    smoke = script_io.SmokeCommand(("tiny-model-smoke", "{script}"), timeout_seconds=7)
    sess = script_io.ValidationSession(
        method_id="awq",
        allowed_root=tmp_path,
        smoke_command=smoke,
    )
    out = tmp_path / "script.py"
    with patch.object(subprocess, "run", smoke_ok):
        result = sess.write(str(out), "answer = 42\n")
    assert result["status"] == "ok"
    assert result["validation"]["checks"][-1]["stage"] == "smoke"
    assert result["validation"]["checks"][-1]["status"] == "passed"
    assert seen["code"] == "answer = 42\n"
    assert seen["timeout"] == 7
    assert not Path(seen["candidate"]).exists()
    assert out.read_text() == "answer = 42\n"


def test_smoke_failure_and_timeout_fail_closed(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    smoke = script_io.SmokeCommand(("tiny-model-smoke", "{script}"), timeout_seconds=2)
    out = tmp_path / "script.py"

    sess = script_io.ValidationSession(
        method_id="awq", max_attempts=1, allowed_root=tmp_path, smoke_command=smoke
    )
    failed = SimpleNamespace(returncode=2, stdout="", stderr="tiny model failed\n")
    with patch.object(subprocess, "run", return_value=failed):
        result = sess.write(str(out), "answer = 42\n")
    assert result["status"] == "error-exhausted"
    assert result["stage"] == "smoke"
    assert "tiny model failed" in result["message"]
    assert not out.exists()

    sess = script_io.ValidationSession(
        method_id="awq", max_attempts=1, allowed_root=tmp_path, smoke_command=smoke
    )
    with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("smoke", 2)):
        result = sess.write(str(out), "answer = 42\n")
    assert result["status"] == "error-exhausted"
    assert result["stage"] == "smoke"
    assert "timed out after 2s" in result["message"]
    assert not out.exists()


def test_smoke_command_requires_placeholder_and_bounded_timeout():
    invalid = [
        (("smoke",), 10),
        (("smoke", "{script}"), 0),
        (("smoke", "{script}"), 301),
    ]
    for argv, timeout in invalid:
        try:
            script_io.SmokeCommand(argv, timeout_seconds=timeout)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid smoke configuration should fail closed")

    try:
        script_io.ValidationSession(method_id="awq", smoke_command="python {script}")
    except TypeError:
        pass
    else:
        raise AssertionError("a shell-like smoke command string must be rejected")


def test_session_exhaustion_fails_closed_without_writing(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    sess = script_io.ValidationSession(method_id="awq", max_attempts=1, allowed_root=tmp_path)
    out = tmp_path / "script.py"
    with patch.object(subprocess, "run", _fake_run_import_err):
        r = sess.write(str(out), "import autowaq\n")
    assert r["status"] == "error-exhausted"
    assert r["attempts_left"] == 0
    assert not out.exists()
    assert sess.validated_path is None


def test_session_tracks_only_current_validated_path(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    stale = tmp_path / "stale.py"
    stale.write_text("print('old')\n")
    out = tmp_path / "current.py"
    sess = script_io.ValidationSession(method_id="awq", allowed_root=tmp_path)
    with patch.object(subprocess, "run", _fake_run_ok):
        sess.write(str(out), "import os\n")
    assert sess.validated_path == out.resolve()
    assert stale.read_text() == "print('old')\n"


def test_make_write_script_tool_returns_json(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    sess = script_io.ValidationSession(method_id="awq", allowed_root=tmp_path)
    tool_fn = script_io.make_write_script_tool(sess)
    with patch.object(subprocess, "run", _fake_run_ok):
        raw = tool_fn.invoke({"path": str(tmp_path / "s.py"), "code": "import os\n"})
    payload = json.loads(raw)
    assert payload["status"] == "ok"


def test_make_write_script_tool_accepts_semantic_overrides(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    sess = script_io.ValidationSession(method_id="awq", allowed_root=tmp_path)
    tool_fn = script_io.make_write_script_tool(
        sess,
        expected_model_id="org/model",
        expected_output_dir=tmp_path / "artifact",
        locked_hyperparameters={"bits": 4},
    )
    code = (
        'MODEL_ID = "org/model"\n'
        f'OUTPUT_DIR = "{tmp_path / "artifact"}"\n'
        "bits = 4\n"
    )
    raw = tool_fn.invoke({"path": str(tmp_path / "s.py"), "code": code})
    payload = json.loads(raw)
    assert payload["status"] == "ok"
    assert payload["validation"]["checks"][2]["status"] == "passed"
