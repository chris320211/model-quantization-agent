"""measurement: subprocess wrapper + JSON parsing. Mocks the actual GPU run."""
from __future__ import annotations

import json
import re
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from quant_agent import measurement
from quant_agent.measurement import (
    MEASURE_SCRIPT,
    load_metrics,
    metrics_summary,
    parse_stdout_metrics,
    run_measurement,
    write_measure_script,
    require_measurement_adapter,
    UnsupportedMeasurementAdapter,
)
from quant_agent.pareto import Metrics


# Script template ------------------------------------------------------------


def test_measure_script_is_valid_python():
    import ast
    ast.parse(MEASURE_SCRIPT)


def test_measure_script_reads_required_env_vars():
    """The embedded script must consume MEASURE_MODEL_PATH and MEASURE_OUTPUT_JSON."""
    assert "MEASURE_MODEL_PATH" in MEASURE_SCRIPT
    assert "MEASURE_OUTPUT_JSON" in MEASURE_SCRIPT
    assert "MEASURE_REPEATS" in MEASURE_SCRIPT
    assert 'device_map="auto"' in MEASURE_SCRIPT
    assert "decode_samples_ms_per_token" in MEASURE_SCRIPT


# write_measure_script --------------------------------------------------------


def test_write_measure_script_creates_executable_file(tmp_path):
    p = write_measure_script(tmp_path)
    assert p.exists()
    assert p.read_text().startswith("#!/usr/bin/env python")
    # chmod 0o755 — owner-execute bit set
    mode = p.stat().st_mode & 0o777
    assert mode & 0o100


# parse_stdout_metrics --------------------------------------------------------


def test_parse_stdout_metrics_picks_sentinel_line():
    payload = {
        "prefill_ms": 12.3, "decode_ms": 45.6, "vram_gb": 7.8, "ppl": 9.0,
    }
    stdout = "noise\nnoise\nMEASUREMENT_RESULT=" + json.dumps(payload) + "\nmore noise\n"
    m = parse_stdout_metrics(stdout)
    assert m == Metrics(prefill_ms=12.3, decode_ms=45.6, vram_gb=7.8, ppl=9.0)


def test_parse_stdout_metrics_returns_none_when_missing():
    assert parse_stdout_metrics("no sentinel here") is None


def test_parse_stdout_metrics_handles_malformed_json():
    assert parse_stdout_metrics("MEASUREMENT_RESULT={not json}\n") is None


# load_metrics ----------------------------------------------------------------


def test_load_metrics_reads_metrics_json(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({
        "prefill_ms": 1, "decode_ms": 2, "vram_gb": 3, "ppl": 4,
    }))
    m = load_metrics(tmp_path)
    assert m == Metrics(prefill_ms=1, decode_ms=2, vram_gb=3, ppl=4)


def test_load_metrics_returns_none_when_missing(tmp_path):
    assert load_metrics(tmp_path) is None


def test_load_metrics_returns_none_on_corrupt_json(tmp_path):
    (tmp_path / "metrics.json").write_text("{not json")
    assert load_metrics(tmp_path) is None


# run_measurement -------------------------------------------------------------


def test_run_measurement_parses_metrics_json(tmp_path, monkeypatch):
    """Successful run: metrics.json is on disk, exit code 0 → return Metrics."""
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    expected_payload = {
        "prefill_ms": 11.0, "decode_ms": 22.0, "vram_gb": 4.0, "ppl": 8.5,
    }

    def fake_run(cmd, stdout, stderr, env, timeout, check):
        # Pretend the child wrote metrics.json before exiting.
        (job_dir / "metrics.json").write_text(json.dumps(expected_payload))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(measurement.subprocess, "run", fake_run)

    venv_py = tmp_path / "fake-venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("")

    result = run_measurement(
        job_dir=job_dir,
        model_path="meta-llama/Llama-2-7b-hf",
        venv_python=venv_py,
    )
    assert result == Metrics(**expected_payload)
    assert (job_dir / "measure.py").exists()


def test_run_measurement_raises_on_subprocess_failure(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    def fake_run(cmd, stdout, stderr, env, timeout, check):
        # Simulate a child that wrote some log lines but did NOT produce metrics.json.
        (job_dir / "measure.log").write_text("CUDA out of memory\n" * 5)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(measurement.subprocess, "run", fake_run)

    venv_py = tmp_path / "fake-venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("")

    with pytest.raises(RuntimeError) as exc_info:
        run_measurement(
            job_dir=job_dir,
            model_path="x",
            venv_python=venv_py,
        )
    assert "exit=1" in str(exc_info.value) or "Measurement failed" in str(exc_info.value)


def test_run_measurement_passes_env_vars(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    captured: dict = {}

    def fake_run(cmd, stdout, stderr, env, timeout, check):
        captured["env"] = env
        captured["cmd"] = cmd
        (job_dir / "metrics.json").write_text(json.dumps(
            {"prefill_ms": 1, "decode_ms": 2, "vram_gb": 3, "ppl": 4}
        ))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(measurement.subprocess, "run", fake_run)
    venv_py = tmp_path / "py"
    venv_py.write_text("")

    run_measurement(
        job_dir=job_dir,
        model_path="quantized/awq-foo",
        venv_python=venv_py,
    )
    assert captured["env"]["MEASURE_MODEL_PATH"] == "quantized/awq-foo"
    assert captured["env"]["MEASURE_OUTPUT_JSON"].endswith("metrics.json")
    assert captured["env"]["MEASURE_REPEATS"] == "5"
    assert captured["env"]["MEASURE_DTYPE"] == "float16"


# metrics_summary -------------------------------------------------------------


def test_metrics_summary_format():
    m = Metrics(prefill_ms=12.3, decode_ms=45.6, vram_gb=7.8, ppl=9.0)
    s = metrics_summary(m)
    assert "12.3" in s
    assert "45.6" in s
    assert "7.80" in s
    assert "9.000" in s
    assert "ms/token" in s


@pytest.mark.parametrize("payload", [
    {"prefill_ms": float("nan"), "decode_ms": 1, "vram_gb": 1, "ppl": 1},
    {"prefill_ms": -1, "decode_ms": 1, "vram_gb": 1, "ppl": 1},
    {"prefill_ms": 1, "decode_ms": 1, "vram_gb": 1, "ppl": 0},
])
def test_metrics_reject_invalid_values(payload):
    with pytest.raises(ValueError):
        Metrics(**payload)


def test_unknown_method_requires_explicit_measurement_adapter():
    require_measurement_adapter("awq")
    with pytest.raises(UnsupportedMeasurementAdapter, match="not configured"):
        require_measurement_adapter("quip")
