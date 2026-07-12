from __future__ import annotations

import json
import hashlib
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from quant_agent.executor import JobMeta


def _git(*args, cwd=None):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_jobmeta_round_trip_with_retry_fields():
    meta = JobMeta(
        job_id="JOB1",
        method_id="awq",
        model_id="meta-llama/Llama-3.2-3B",
        venv="awq",
        script_path="/tmp/jobs/JOB1/script.py",
        output_dir="/tmp/out/JOB1",
        pid=12345,
        started_at="2026-04-19T12:00:00+00:00",
        parent_job_id="JOB0",
        attempt=2,
    )
    payload = json.loads(meta.to_json())
    assert payload["parent_job_id"] == "JOB0"
    assert payload["attempt"] == 2

    reloaded = JobMeta(**payload)
    assert asdict(reloaded) == asdict(meta)


def test_jobmeta_defaults_are_backwards_compatible():
    """A meta.json written before the retry fields existed still loads."""
    legacy_payload = {
        "job_id": "OLD",
        "method_id": "gptq",
        "model_id": "m",
        "venv": "gptq",
        "script_path": "/tmp/old/script.py",
        "output_dir": "/tmp/old",
        "pid": 1,
        "started_at": "2026-04-01T00:00:00+00:00",
    }
    meta = JobMeta(**legacy_payload)
    assert meta.parent_job_id is None
    assert meta.attempt == 1
    assert meta.status == "running"
    assert meta.fix_note is None
    assert meta.manifest_path is None
    assert meta.execution_mode == "host"


def test_jobmeta_fix_note_round_trip():
    meta = JobMeta(
        job_id="JOB2",
        method_id="awq",
        model_id="m",
        venv="awq",
        script_path="/tmp/jobs/JOB2/script.py",
        output_dir="/tmp/out",
        pid=1,
        started_at="2026-07-10T00:00:00+00:00",
        parent_job_id="JOB1",
        attempt=2,
        fix_note="pinned transformers==4.46.3 in the awq venv",
    )
    payload = json.loads(meta.to_json())
    assert payload["fix_note"] == "pinned transformers==4.46.3 in the awq venv"
    assert JobMeta(**payload).fix_note == meta.fix_note


def test_error_signature_picks_last_error_line(tmp_path, monkeypatch):
    from quant_agent import executor

    monkeypatch.setattr(executor, "JOBS_ROOT", tmp_path)
    job_id = "20260101T000000Z-abc123"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "stderr.log").write_text(
        "Traceback (most recent call last):\n"
        '  File "script.py", line 3, in <module>\n'
        "    import awq\n"
        "ModuleNotFoundError: No module named 'awq'\n"
    )
    (job_dir / "stdout.log").write_text("loading model\n")

    assert (
        executor.error_signature(job_id) == "ModuleNotFoundError: No module named 'awq'"
    )


def test_error_signature_prefers_stdout_error_over_stderr_noise(tmp_path, monkeypatch):
    """tqdm/download progress on stderr must not shadow a real error printed to stdout."""
    from quant_agent import executor

    monkeypatch.setattr(executor, "JOBS_ROOT", tmp_path)
    job_id = "20260101T000000Z-aaa111"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "stderr.log").write_text("Downloading: 37%|###       | 3/8 [00:12<00:20]\n")
    (job_dir / "stdout.log").write_text("loading model\nValueError: bad group_size\n")

    assert executor.error_signature(job_id) == "ValueError: bad group_size"


def test_error_signature_empty_stderr_falls_through_to_stdout(tmp_path, monkeypatch):
    from quant_agent import executor

    monkeypatch.setattr(executor, "JOBS_ROOT", tmp_path)
    job_id = "20260101T000000Z-bbb222"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "stderr.log").write_text("")
    (job_dir / "stdout.log").write_text("RuntimeError: CUDA error\n")

    assert executor.error_signature(job_id) == "RuntimeError: CUDA error"


def test_error_signature_falls_back_to_last_stderr_line(tmp_path, monkeypatch):
    from quant_agent import executor

    monkeypatch.setattr(executor, "JOBS_ROOT", tmp_path)
    job_id = "20260101T000000Z-def456"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "stderr.log").write_text("Killed\n")

    assert executor.error_signature(job_id) == "Killed"


def test_error_signature_missing_job_returns_none():
    from quant_agent import executor

    assert executor.error_signature("not-a-real-job-id") is None


def test_relaunch_job_records_fix_note(tmp_path, monkeypatch):
    from quant_agent.tools import executor_tools

    script = tmp_path / "script.py"
    script.write_text("print('hi')\n")
    parent = JobMeta(
        job_id="20260101T000000Z-abc123",
        method_id="awq",
        model_id="m",
        venv="awq",
        script_path=str(script),
        output_dir="/tmp/out",
        pid=1,
        started_at="2026-07-10T00:00:00+00:00",
        status="failed",
        exit_code=1,
        tune_iter=2,
        hyperparameters={"group_size": 128},
    )
    monkeypatch.setattr(executor_tools.executor, "refresh_status", lambda jid: parent)

    captured: dict = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return JobMeta(
            job_id="20260101T000001Z-abc124",
            method_id="awq",
            model_id="m",
            venv="awq",
            script_path=str(script),
            output_dir="/tmp/out",
            pid=2,
            started_at="2026-07-10T00:00:01+00:00",
            parent_job_id=parent.job_id,
            attempt=2,
            fix_note=kwargs.get("fix_note"),
        )

    monkeypatch.setattr(executor_tools.executor, "launch", fake_launch)

    out = json.loads(
        executor_tools.relaunch_job.invoke(
            {"job_id": parent.job_id, "fix_description": "  pinned foo==1.2  "}
        )
    )
    assert out["status"] == "ok"
    assert captured["fix_note"] == "pinned foo==1.2"
    assert captured["tune_iter"] == 2
    assert captured["hyperparameters"] == {"group_size": 128}
    assert out["fix_note"] == "pinned foo==1.2"


def test_edit_script_cannot_change_tune_locked_block(tmp_path, monkeypatch):
    from quant_agent.tools import executor_tools

    jid = "20260101T000000Z-abc999"
    job_dir = tmp_path / jid
    job_dir.mkdir()
    script = job_dir / "script.py"
    script.write_text(
        "# TUNE-LOCKED HYPERPARAMETERS (do not modify in fix_agent):\n"
        "# group_size=128\n"
        "print('ok')\n"
    )
    monkeypatch.setattr(executor_tools.executor, "JOBS_ROOT", tmp_path)
    result = json.loads(
        executor_tools.edit_script.invoke(
            {"job_id": jid, "old": "# group_size=128", "new": "# group_size=64"}
        )
    )
    assert result["status"] == "error"
    assert "TUNE-LOCKED" in result["error"]
    assert "group_size=128" in script.read_text()


def test_edit_script_cannot_remove_port_overlay_contract(tmp_path, monkeypatch):
    from quant_agent import executor
    from quant_agent.tools.executor_tools import edit_script

    job_id = "20260711T000000Z-abc123"
    job = tmp_path / job_id
    job.mkdir()
    overlay = tmp_path / "out" / "overlays" / "awq" / "model" / "hash"
    text = (
        f"# QUANT_AGENT_OVERLAY_DIR={overlay}\n"
        "import os\nOVERLAY = os.environ['QUANT_AGENT_OVERLAY_DIR']\n"
        "REPO = os.environ['QUANT_AGENT_METHOD_REPO']\n"
    )
    (job / "script.py").write_text(text)
    monkeypatch.setattr(executor, "JOBS_ROOT", tmp_path)
    payload = json.loads(edit_script.invoke({
        "job_id": job_id,
        "old": "REPO = os.environ['QUANT_AGENT_METHOD_REPO']",
        "new": "REPO = '/canonical/repo'",
    }))
    assert payload["status"] == "error"
    assert "port-overlay contract" in payload["error"]
    assert (job / "script.py").read_text() == text


def test_container_command_plan_renders_without_shell_interpolation():
    from quant_agent.execution_policy import ContainerCommandPlan, ExecutionPolicy

    policy = ExecutionPolicy.containerized(
        ContainerCommandPlan(
            ("docker", "run", "--rm", "-v", "{job_dir}:/job", "image", "python", "/job/script.py")
        )
    )
    argv = policy.command_argv(
        host_python=Path("/unused/python"),
        script_path=Path("/tmp/job with spaces/script.py"),
        job_dir=Path("/tmp/job with spaces"),
        output_dir="/tmp/output",
        repo_root=Path("/workspace"),
        job_id="id",
        method_id="awq",
        model_id="org/model",
    )
    assert argv[4] == "/tmp/job with spaces:/job"
    assert argv[-2:] == ["python", "/job/script.py"]


def test_execution_policy_rejects_incomplete_or_malformed_container_plan():
    from quant_agent.execution_policy import ContainerCommandPlan, ExecutionMode, ExecutionPolicy

    with pytest.raises(ValueError, match="requires a container command"):
        ExecutionPolicy(mode=ExecutionMode.CONTAINER)
    with pytest.raises(ValueError, match="unsupported container runtime"):
        ContainerCommandPlan(("bash", "-c", "python script.py"))
    with pytest.raises(ValueError, match="credential environment names"):
        ContainerCommandPlan(("docker", "run", "--env=HF_TOKEN=secret-value", "image"))
    plan = ContainerCommandPlan(("docker", "{unknown}"))
    with pytest.raises(ValueError, match="placeholder"):
        plan.render({key: "x" for key in (
            "job_dir", "script_path", "output_dir", "repo_root", "job_id", "method_id", "model_id"
        )})


def test_build_manifest_collects_required_reproducibility_fields(tmp_path, monkeypatch):
    from quant_agent import reproducibility

    monkeypatch.setattr(reproducibility, "_runtime_versions", lambda: {"torch": "2.test"})
    monkeypatch.setattr(
        reproducibility,
        "_gpu_cuda_info",
        lambda: {"gpus": [{"name": "Test GPU"}], "cuda_toolkit": "12.test"},
    )
    monkeypatch.setattr(reproducibility, "_method_repo_commit", lambda path: "b" * 40)
    manifest = reproducibility.build_manifest(
        method_id="awq",
        model_id="org/model",
        script_code="pass\n",
        output_dir="/output",
        execution_mode="container",
        method_repo_dir=tmp_path / "repo",
        created_at="2026-07-11T00:00:00+00:00",
    )

    payload = json.loads(manifest.to_json())
    assert payload["schema_version"] == "1.0"
    assert payload["created_at"] == "2026-07-11T00:00:00+00:00"
    assert payload["script_sha256"] == hashlib.sha256(b"pass\n").hexdigest()
    assert payload["runtime_versions"] == {"torch": "2.test"}
    assert payload["gpu_cuda"]["cuda_toolkit"] == "12.test"
    assert payload["method_repo_commit"] == "b" * 40
    assert payload["execution_command"] == []
    assert payload["python"]["version"]
    assert payload["platform"]["system"]


def test_launch_writes_reproducibility_manifest(tmp_path, monkeypatch):
    from quant_agent import executor
    from quant_agent.reproducibility import ReproducibilityManifest

    jobs = tmp_path / "jobs"
    venvs = tmp_path / ".venvs"
    python = venvs / "awq" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    monkeypatch.setattr(executor, "JOBS_ROOT", jobs)
    monkeypatch.setattr(executor, "VENV_ROOT", venvs)
    monkeypatch.setattr(executor, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(executor, "_new_job_id", lambda: "20260101T000000Z-abc123")

    class Proc:
        pid = 43210

    monkeypatch.setattr(executor.subprocess, "Popen", lambda *args, **kwargs: Proc())
    monkeypatch.setattr(executor.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        executor,
        "build_manifest",
        lambda **kwargs: ReproducibilityManifest(
            schema_version="1.0",
            created_at=kwargs["created_at"],
            model_id=kwargs["model_id"],
            method_id=kwargs["method_id"],
            script_sha256=hashlib.sha256(kwargs["script_code"].encode()).hexdigest(),
            output_dir=kwargs["output_dir"],
            execution_mode=kwargs["execution_mode"],
            python={"version": "3.test"},
            platform={"system": "test"},
            runtime_versions={},
            gpu_cuda={},
            method_repo_commit=None,
        ),
    )

    meta = executor.launch("awq", "org/model", "print('ok')\n", "/tmp/out")
    manifest_path = jobs / meta.job_id / "reproducibility.json"
    payload = json.loads(manifest_path.read_text())
    assert meta.manifest_path == str(manifest_path)
    assert meta.execution_mode == "host"
    assert payload["model_id"] == "org/model"
    assert payload["method_id"] == "awq"
    assert payload["output_dir"] == "/tmp/out"
    assert payload["execution_mode"] == "host"
    assert payload["script_sha256"] == hashlib.sha256(b"print('ok')\n").hexdigest()
    assert not list(manifest_path.parent.glob(".reproducibility.json.*.tmp"))


def test_container_launch_uses_plan_without_host_venv_or_acknowledgement(tmp_path, monkeypatch):
    from quant_agent import executor
    from quant_agent.config import host_execution_policy
    from quant_agent.execution_policy import ContainerCommandPlan, ExecutionPolicy
    from quant_agent.reproducibility import ReproducibilityManifest

    monkeypatch.setattr(executor, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(executor, "VENV_ROOT", tmp_path / "missing-venvs")
    monkeypatch.setattr(executor, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(executor, "_new_job_id", lambda: "20260101T000000Z-abc124")
    captured = {}

    class Proc:
        pid = 43211

    def fake_popen(args, **kwargs):
        captured["args"] = args
        return Proc()

    monkeypatch.setattr(executor.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(executor.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        executor,
        "build_manifest",
        lambda **kwargs: ReproducibilityManifest(
            schema_version="1.0", created_at=kwargs["created_at"], model_id=kwargs["model_id"],
            method_id=kwargs["method_id"], script_sha256="a" * 64,
            output_dir=kwargs["output_dir"], execution_mode=kwargs["execution_mode"],
            python={}, platform={}, runtime_versions={}, gpu_cuda={}, method_repo_commit=None,
        ),
    )
    policy = ExecutionPolicy.containerized(
        ContainerCommandPlan(("podman", "run", "--rm", "image", "python", "/job/script.py"))
    )
    with host_execution_policy(False):
        meta = executor.launch("awq", "org/model", "pass\n", "/tmp/out", execution_policy=policy)

    assert meta.execution_mode == "container"
    assert "podman run --rm image python /job/script.py" in captured["args"][2]


def test_relaunch_refuses_to_downgrade_container_job_to_host(tmp_path, monkeypatch):
    from quant_agent import executor
    from quant_agent.tools.executor_tools import relaunch_job

    parent = JobMeta(
        job_id="20260711T000000Z-abcdef", method_id="awq", model_id="org/model",
        venv="awq", script_path=str(tmp_path / "script.py"), output_dir="/out",
        pid=1, started_at="now", status="failed", execution_mode="container",
    )
    (tmp_path / "script.py").write_text("pass\n")
    monkeypatch.setattr(executor, "refresh_status", lambda _: parent)
    payload = json.loads(relaunch_job.invoke({
        "job_id": parent.job_id, "fix_description": "test",
    }))
    assert payload["status"] == "error"
    assert "refusing to downgrade" in payload["error"]


def test_launch_snapshots_port_overlay_and_passes_only_snapshot_to_job(tmp_path, monkeypatch):
    from quant_agent import executor
    from quant_agent.port_overlay import PortOverlaySession
    from quant_agent.reproducibility import ReproducibilityManifest

    jobs = tmp_path / "jobs"
    venvs = tmp_path / ".venvs"
    python = venvs / "awq" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    overlay_session = PortOverlaySession(
        root=tmp_path / "out" / "overlays" / "awq" / "org_model",
        method_id="awq", model_id="org/model", base_commit="a" * 40,
    )
    result = overlay_session.write(
        patch=(
            "diff --git a/awq/model.py b/awq/model.py\n"
            "--- a/awq/model.py\n+++ b/awq/model.py\n"
            "@@ -1 +1,2 @@\n X = 1\n+Y = 2\n"
        ),
        rationale="Add target architecture dispatch.",
    )
    overlay = Path(result["overlay_dir"])
    script = (
        f"# QUANT_AGENT_OVERLAY_DIR={overlay}\n"
        "import os\nOVERLAY = os.environ['QUANT_AGENT_OVERLAY_DIR']\n"
        "REPO = os.environ['QUANT_AGENT_METHOD_REPO']\n"
    )
    monkeypatch.setattr(executor, "JOBS_ROOT", jobs)
    monkeypatch.setattr(executor, "VENV_ROOT", venvs)
    monkeypatch.setattr(executor, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(executor, "_new_job_id", lambda: "20260101T000000Z-aabbcc")
    def fake_prepare(method_id, manifest, snapshot, job_dir):
        worktree = job_dir / "method-repo"
        worktree.mkdir()
        return worktree
    monkeypatch.setattr(executor, "_prepare_overlay_worktree", fake_prepare)
    monkeypatch.setattr(
        executor,
        "build_manifest",
        lambda **kwargs: ReproducibilityManifest(
            schema_version="1.0", created_at=kwargs["created_at"],
            model_id=kwargs["model_id"], method_id=kwargs["method_id"],
            script_sha256="a" * 64, output_dir=kwargs["output_dir"],
            execution_mode=kwargs["execution_mode"], python={}, platform={},
            runtime_versions={}, gpu_cuda={}, method_repo_commit=None,
            overlay_sha256=kwargs["overlay_sha256"],
            overlay_snapshot_path=kwargs["overlay_snapshot_path"],
        ),
    )
    captured = {}

    class Proc:
        pid = 43212

    def fake_popen(*args, **kwargs):
        captured["env"] = kwargs["env"]
        return Proc()

    monkeypatch.setattr(executor.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(executor.os, "getpgid", lambda pid: pid)
    meta = executor.launch("awq", "org/model", script, "/tmp/out")
    snapshot = jobs / meta.job_id / "overlay"
    assert meta.overlay_path == str(snapshot)
    assert meta.overlay_sha256
    assert (snapshot / "overlay.patch").read_text() == (overlay / "overlay.patch").read_text()
    assert captured["env"]["QUANT_AGENT_OVERLAY_DIR"] == str(snapshot)
    assert captured["env"]["QUANT_AGENT_METHOD_REPO"] == str(jobs / meta.job_id / "method-repo")
    manifest = json.loads((jobs / meta.job_id / "reproducibility.json").read_text())
    assert manifest["overlay_sha256"] == meta.overlay_sha256


def test_prepare_overlay_worktree_applies_patch_without_changing_canonical_checkout(tmp_path, monkeypatch):
    from quant_agent import executor
    from quant_agent.port_overlay import PortOverlaySession, validate_overlay_bundle

    repo = tmp_path / ".venvs" / "awq" / "repo"
    repo.mkdir(parents=True)
    _git("init", cwd=repo)
    source = repo / "model.py"
    source.write_text("SUPPORTED = ['llama']\n")
    _git("add", "model.py", cwd=repo)
    _git(
        "-c", "user.name=Test", "-c", "user.email=test@example.com",
        "commit", "-m", "initial", cwd=repo,
    )
    commit = _git("rev-parse", "HEAD", cwd=repo)
    session = PortOverlaySession(
        root=tmp_path / "out" / "overlays" / "awq" / "org_model",
        method_id="awq", model_id="org/model", base_commit=commit,
    )
    result = session.write(
        patch=(
            "diff --git a/model.py b/model.py\n"
            "--- a/model.py\n+++ b/model.py\n"
            "@@ -1 +1,2 @@\n SUPPORTED = ['llama']\n+SUPPORTED.append('qwen2')\n"
        ),
        rationale="Add Qwen2 dispatch.",
    )
    overlay = Path(result["overlay_dir"])
    snapshot = tmp_path / "jobs" / "job" / "overlay"
    snapshot.parent.mkdir(parents=True)
    import shutil
    shutil.copytree(overlay, snapshot)
    monkeypatch.setattr(executor, "VENV_ROOT", tmp_path / ".venvs")
    worktree = executor._prepare_overlay_worktree(
        "awq", validate_overlay_bundle(snapshot), snapshot, snapshot.parent,
    )
    assert source.read_text() == "SUPPORTED = ['llama']\n"
    assert "qwen2" in (worktree / "model.py").read_text()
    executor._cleanup_overlay_worktree("awq", worktree)
    assert not worktree.exists()
