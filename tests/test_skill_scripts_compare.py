"""Offline tests for method × model × GPU WikiText-2 ranking."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agents" / "skills" / "_shared" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import compare as compare_mod  # noqa: E402
import jobs as jobs_mod  # noqa: E402
import paths as paths_mod  # noqa: E402


def test_faster_quality_ok_method_wins_over_broken_fast_method():
    rows = [
        {
            "method_name": "BrokenFast",
            "quality_ok": False,
            "tokens_per_s": 9000.0,
            "peak_vram_gb": 2.0,
            "ppl_ratio": 8.0,
            "packed": True,
        },
        {
            "method_name": "Solid",
            "quality_ok": True,
            "tokens_per_s": 1000.0,
            "peak_vram_gb": 4.0,
            "ppl_ratio": 1.2,
            "packed": False,
        },
    ]
    ranked = compare_mod.rank_methods(rows)
    assert ranked[0]["method_name"] == "Solid"
    assert ranked[0]["rank"] == 1
    best = compare_mod.pick_best(rows)
    assert best is not None
    assert best["method_name"] == "Solid"


def test_same_board_ranks_two_quality_ok_methods_by_tok_s():
    rows = [
        {
            "method_name": "AWQ",
            "quality_ok": True,
            "tokens_per_s": 4000.0,
            "peak_vram_gb": 5.0,
            "ppl_ratio": 1.10,
            "packed": True,
        },
        {
            "method_name": "FlatQuant",
            "quality_ok": True,
            "tokens_per_s": 5105.9,
            "peak_vram_gb": 3.35,
            "ppl_ratio": 1.17,
            "packed": True,
        },
    ]
    best = compare_mod.pick_best(rows)
    assert best is not None
    assert best["method_name"] == "FlatQuant"
    ranked = compare_mod.rank_methods(rows)
    assert [row["method_name"] for row in ranked] == ["FlatQuant", "AWQ"]


def test_group_filename_is_model_and_gpu_not_method():
    name = compare_mod.group_filename(
        "microsoft/Phi-3-mini-4k-instruct", "g5.2xlarge"
    )
    assert "Phi-3-mini-4k-instruct" in name
    assert "g5.2xlarge" in name
    assert "flatquant" not in name.lower()
    assert "awq" not in name.lower()
    assert compare_mod.group_key("org/model", "g5.xlarge") == "org/model|g5.xlarge"


def _write_job(tmp_path: Path, job_id: str, *, packed: bool, tps: float, ratio: float) -> Path:
    output = tmp_path / "quantized" / job_id
    output.mkdir(parents=True)
    if packed:
        (output / "packed_w4a4.pt").write_bytes(b"x")
        (output / "quantization_config.json").write_text(
            json.dumps({"format": "packed_int4"}) + "\n"
        )
    else:
        (output / "pytorch_model.bin").write_bytes(b"x")
        (output / "quantization_config.json").write_text(
            json.dumps({"format": "fakequant_fp16"}) + "\n"
        )
    meta = jobs_mod.JobMeta(
        job_id=job_id,
        method_id="m",
        model_id="microsoft/Phi-3-mini-4k-instruct",
        venv="venv",
        script_path="script.py",
        output_dir=str(output),
        pid=1,
        started_at="2026-01-01T00:00:00Z",
        status="completed",
        verification_status="passed",
        benchmark_status="passed",
    )
    jobs_mod.write_meta(meta)
    job_dir = jobs_mod.job_dir(job_id)
    ppl = 6.0 * ratio
    payload = {
        "quantized": {
            "perplexity": ppl,
            "tokens_per_s": tps,
            "peak_vram_gb": 3.35 if packed else 8.0,
        },
        "fp16_baseline": {
            "perplexity": 6.0,
            "tokens_per_s": 3300.0,
            "peak_vram_gb": 9.68,
        },
        "comparison": {
            "ppl_ratio": ratio,
            "quality_ok": ratio <= 1.5,
            "improved_throughput": tps > 3300.0,
            "improved_vram": True,
        },
    }
    (job_dir / "benchmark.json").write_text(json.dumps(payload) + "\n")
    return output


def test_record_job_replaces_same_method_and_ranks_two_methods(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(paths_mod, "COMPARE_ROOT", tmp_path / "compare")
    monkeypatch.setattr(paths_mod, "REPO_ROOT", tmp_path)
    job_a = "20260101T000000Z-aaaaaa"
    job_b = "20260101T000000Z-bbbbbb"
    job_c = "20260101T000000Z-cccccc"
    _write_job(tmp_path, job_a, packed=True, tps=3140.0, ratio=1.17)
    first = compare_mod.record_job(
        job_id=job_a,
        request={
            "method_name": "FlatQuant",
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "gpu_instance": "g5.2xlarge",
            "gpu_name": "NVIDIA A10G",
            "slug": "flatquant-phi3-mini-4k-g52xlarge",
        },
        hub_url="https://huggingface.co/example/flatquant-phi3",
    )
    assert first["best"]["method_name"] == "FlatQuant"
    assert first["methods"][0]["packed"] is True
    assert first["methods"][0]["hub_url"].endswith("flatquant-phi3")

    _write_job(tmp_path, job_b, packed=True, tps=5105.9, ratio=1.17)
    second = compare_mod.record_job(
        job_id=job_b,
        request={
            "method_name": "FlatQuant",
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "gpu_instance": "g5.2xlarge",
            "slug": "flatquant-phi3-mini-4k-g52xlarge",
        },
    )
    methods = {row["method_name"]: row for row in second["methods"]}
    assert len(methods) == 1
    assert methods["FlatQuant"]["job_id"] == job_b
    assert methods["FlatQuant"]["tokens_per_s"] == 5105.9
    assert methods["FlatQuant"]["hub_url"].endswith("flatquant-phi3")

    _write_job(tmp_path, job_c, packed=True, tps=4000.0, ratio=1.10)
    third = compare_mod.record_job(
        job_id=job_c,
        request={
            "method_name": "AWQ",
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "gpu_instance": "g5.2xlarge",
            "slug": "awq-phi3-mini-4k-g52xlarge",
        },
    )
    names = [row["method_name"] for row in third["methods"]]
    assert names[0] == "FlatQuant"
    assert "AWQ" in names
    assert third["best"]["method_name"] == "FlatQuant"
    index = json.loads((tmp_path / "compare" / "index.json").read_text())
    assert index["groups"][0]["n_methods"] == 2
    assert index["groups"][0]["best_method"] == "FlatQuant"
    readme = (tmp_path / "compare" / "README.md").read_text()
    assert "microsoft/Phi-3-mini-4k-instruct" in readme
    assert "g5.2xlarge" in readme
    assert "**Best:** FlatQuant" in readme
    catalog = json.loads((tmp_path / "compare" / "catalog.json").read_text())
    assert len(catalog["rows"]) == 2
    by_method = {row["method_name"]: row for row in catalog["rows"]}
    assert by_method["FlatQuant"]["is_best"] is True
    assert by_method["AWQ"]["is_best"] is False
    assert by_method["FlatQuant"]["hub_repo_id"] == "example/flatquant-phi3"
    matched = compare_mod.query(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        gpu_instance="g5.2xlarge",
        method_name="awq",
    )
    assert len(matched) == 1
    assert matched[0]["method_name"] == "AWQ"
    best = compare_mod.query(gpu_instance="g5.2xlarge", best_only=True)
    assert [row["method_name"] for row in best] == ["FlatQuant"]
    howto = compare_mod.fetch_howto(by_method["FlatQuant"])
    assert howto["status"] == "ok"
    assert "huggingface-cli download example/flatquant-phi3" in howto["command"]
    local = compare_mod.fetch_howto({"artifact_dir": "quantized/x"})
    assert local["status"] == "local_only"


def test_contribution_json_is_the_pr_unit(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(paths_mod, "COMPARE_ROOT", tmp_path / "compare")
    monkeypatch.setattr(paths_mod, "REPO_ROOT", tmp_path)
    job_id = "20260101T000000Z-dddddd"
    _write_job(tmp_path, job_id, packed=True, tps=5105.9, ratio=1.17)
    compare_mod.record_job(
        job_id=job_id,
        request={
            "method_name": "FlatQuant",
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "gpu_instance": "g5.2xlarge",
            "slug": "flatquant-phi3-mini-4k-g52xlarge",
        },
    )
    contrib = {
        "schema_version": 1,
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "method_name": "AWQ",
        "gpu_instance": "g5.2xlarge",
        "gpu_name": "NVIDIA A10G",
        "quality_ok": True,
        "ppl": 6.6,
        "ppl_ratio": 1.1,
        "tokens_per_s": 4000.0,
        "peak_vram_gb": 4.2,
        "fp16_ppl": 6.0,
        "fp16_tokens_per_s": 3300.0,
        "fp16_peak_vram_gb": 9.68,
        "packed": True,
        "hub_repo_id": "someone/awq-phi3",
        "dataset": "wikitext-2-raw-v1",
        "max_seq_len": 2048,
    }
    src = tmp_path / "incoming.json"
    src.write_text(json.dumps(contrib) + "\n")
    dest = compare_mod.accept_contribution(src)
    assert dest.name.endswith("AWQ.json")
    rows = {row["method_name"]: row for row in compare_mod.query()}
    assert set(rows) == {"FlatQuant", "AWQ"}
    assert rows["AWQ"]["hub_repo_id"] == "someone/awq-phi3"
    assert rows["FlatQuant"]["is_best"] is True
    howto = compare_mod.fetch_howto(rows["AWQ"])
    assert "huggingface-cli download someone/awq-phi3" in howto["command"]


def test_validate_contribution_requires_hub_and_quality():
    base = {
        "model_id": "org/model",
        "method_name": "AWQ",
        "gpu_instance": "g5.xlarge",
        "quality_ok": True,
        "ppl": 11.0,
        "ppl_ratio": 1.1,
        "tokens_per_s": 100.0,
        "peak_vram_gb": 4.0,
        "fp16_ppl": 10.0,
        "fp16_tokens_per_s": 80.0,
        "fp16_peak_vram_gb": 8.0,
        "hub_repo_id": "org/awq-model",
    }
    compare_mod.validate_contribution(base)
    bad_quality = dict(base, quality_ok=False)
    try:
        compare_mod.validate_contribution(bad_quality)
    except ValueError as exc:
        assert "quality_ok" in str(exc)
    else:
        raise AssertionError("expected quality_ok rejection")
    missing_hub = dict(base, hub_repo_id=None)
    try:
        compare_mod.validate_contribution(missing_hub)
    except ValueError as exc:
        assert "hub_repo_id" in str(exc)
    else:
        raise AssertionError("expected hub_repo_id rejection")


def test_pick_best_none_when_nothing_quality_ok():
    assert compare_mod.pick_best([{"quality_ok": False, "tokens_per_s": 9}]) is None


def test_quant_skills_point_at_compare_index():
    parent = (ROOT / "agents" / "skills" / "quant" / "SKILL.md").read_text()
    bench = (ROOT / "agents" / "skills" / "quant-benchmark" / "SKILL.md").read_text()
    publish = (ROOT / "agents" / "skills" / "quant-publish" / "SKILL.md").read_text()
    assert "compare/" in parent
    assert "catalog.json" in parent
    assert "compare/" in bench
    assert "contributions" in parent
    assert "contributions" in publish
