from __future__ import annotations

import json
import subprocess
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
    monkeypatch.setattr(script_io, "venv_python", lambda venv: fake_py)
    monkeypatch.setattr(script_io, "METHOD_TO_VENV", {"awq": "awq"})


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


def test_unknown_method_venv(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    ok, stage, msg = script_io.validate("import os\n", "unknown_method")
    assert ok is False
    assert stage == "dry-import"
    assert "No venv mapping" in msg


def test_missing_venv_python(tmp_path, monkeypatch):
    monkeypatch.setattr(script_io, "venv_python", lambda venv: tmp_path / "does_not_exist")
    monkeypatch.setattr(script_io, "METHOD_TO_VENV", {"awq": "awq"})
    ok, stage, msg = script_io.validate("import os\n", "awq")
    assert ok is False
    assert stage == "dry-import"
    assert "Venv python not found" in msg


def test_session_writes_on_success(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    sess = script_io.ValidationSession(method_id="awq")
    out = tmp_path / "sub" / "script.py"
    with patch.object(subprocess, "run", _fake_run_ok):
        result = sess.write(str(out), "import os\n")
    assert result["status"] == "ok"
    assert result["attempts_left"] == 3  # unchanged on success
    assert out.read_text() == "import os\n"


def test_session_retry_on_error(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    sess = script_io.ValidationSession(method_id="awq", max_attempts=3)
    out = tmp_path / "script.py"
    with patch.object(subprocess, "run", _fake_run_import_err):
        r1 = sess.write(str(out), "import autowaq\n")
    assert r1["status"] == "error"
    assert r1["stage"] == "dry-import"
    assert r1["attempts_left"] == 2
    assert not out.exists()  # nothing written on recoverable failure


def test_session_exhaustion_writes_with_warning(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    sess = script_io.ValidationSession(method_id="awq", max_attempts=1)
    out = tmp_path / "script.py"
    with patch.object(subprocess, "run", _fake_run_import_err):
        r = sess.write(str(out), "import autowaq\n")
    assert r["status"] == "error-exhausted"
    assert r["attempts_left"] == 0
    body = out.read_text()
    assert body.startswith("# WARNING: failed validation")


def test_make_write_script_tool_returns_json(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    sess = script_io.ValidationSession(method_id="awq")
    tool_fn = script_io.make_write_script_tool(sess)
    with patch.object(subprocess, "run", _fake_run_ok):
        raw = tool_fn.invoke({"path": str(tmp_path / "s.py"), "code": "import os\n"})
    payload = json.loads(raw)
    assert payload["status"] == "ok"
