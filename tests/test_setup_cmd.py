from __future__ import annotations

import os

import pytest

from quant_agent import setup_cmd


def test_write_env_file_is_mode_600_from_atomic_replacement(tmp_path):
    target = tmp_path / "credentials"
    setup_cmd._write_env_file(target, "TOKEN=value\n")
    assert target.read_text() == "TOKEN=value\n"
    assert target.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".credentials.*.tmp"))


def test_write_env_file_replaces_regular_file(tmp_path):
    target = tmp_path / "credentials"
    target.write_text("old\n")
    setup_cmd._write_env_file(target, "new\n")
    assert target.read_text() == "new\n"
    assert target.stat().st_mode & 0o777 == 0o600


def test_write_env_file_refuses_symlink(tmp_path):
    real = tmp_path / "real"
    real.write_text("keep\n")
    link = tmp_path / "credentials"
    os.symlink(real, link)
    with pytest.raises(RuntimeError, match="symlink"):
        setup_cmd._write_env_file(link, "secret\n")
    assert real.read_text() == "keep\n"


def test_format_env_rejects_assignment_injection():
    with pytest.raises(ValueError, match="forbidden"):
        setup_cmd._format_env({
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "QUANT_AGENT_MODEL": "safe\nGITHUB_TOKEN=attacker",
        })
