from __future__ import annotations

import json
from dataclasses import asdict

from quant_agent.executor import JobMeta


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
    assert out["fix_note"] == "pinned foo==1.2"
