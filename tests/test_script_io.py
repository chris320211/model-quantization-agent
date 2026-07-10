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


def test_session_exhaustion_writes_with_warning(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    sess = script_io.ValidationSession(method_id="awq", max_attempts=1, allowed_root=tmp_path)
    out = tmp_path / "script.py"
    with patch.object(subprocess, "run", _fake_run_import_err):
        r = sess.write(str(out), "import autowaq\n")
    assert r["status"] == "error-exhausted"
    assert r["attempts_left"] == 0
    body = out.read_text()
    assert body.startswith("# WARNING: failed validation")


def test_make_write_script_tool_returns_json(tmp_path, monkeypatch):
    _stub_venv(tmp_path, monkeypatch)
    sess = script_io.ValidationSession(method_id="awq", allowed_root=tmp_path)
    tool_fn = script_io.make_write_script_tool(sess)
    with patch.object(subprocess, "run", _fake_run_ok):
        raw = tool_fn.invoke({"path": str(tmp_path / "s.py"), "code": "import os\n"})
    payload = json.loads(raw)
    assert payload["status"] == "ok"
