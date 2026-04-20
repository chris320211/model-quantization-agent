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
