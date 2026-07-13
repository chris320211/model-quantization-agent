from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_skill_catalogs_match_packaged_catalogs():
    canonical = ROOT / "src" / "quant_agent" / "data"
    for mirror in (
        ROOT / ".agents" / "skills" / "quant" / "reference",
        ROOT / ".claude" / "skills" / "quant" / "reference",
    ):
        for name in (
            "methods.yaml", "method_capabilities.yaml", "model_aliases.yaml",
            "aws_instances.yaml", "gpu_specs.yaml",
        ):
            assert (mirror / name).read_bytes() == (canonical / name).read_bytes()
    assert len(yaml.safe_load((canonical / "methods.yaml").read_text())) == 35


def test_skill_docs_have_no_obsolete_codex_paths():
    for path in (ROOT / ".agents" / "skills").glob("*/SKILL.md"):
        assert ".Codex/" not in path.read_text()


def test_credential_loader_does_not_execute_values(tmp_path):
    credentials = tmp_path / "credentials"
    marker = tmp_path / "executed"
    credentials.write_text(f"HF_TOKEN=$(touch {marker})\n")
    credentials.chmod(0o600)
    loader = ROOT / ".agents" / "skills" / "_shared" / "load_env.sh"
    result = subprocess.run(
        ["bash", "-c", 'source "$1" "$2"; test -n "$HF_TOKEN"', "bash", str(loader), str(credentials)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not marker.exists()


def test_measurement_skill_is_thin_canonical_launcher():
    path = ROOT / ".agents" / "skills" / "quant-tune" / "reference" / "measure.py"
    text = path.read_text()
    assert "from quant_agent.measurement import MEASURE_SCRIPT" in text
    assert "AutoModelForCausalLM" not in text


def test_skill_mirrors_match_exactly():
    agents = ROOT / ".agents" / "skills"
    claude = ROOT / ".claude" / "skills"
    relative_paths = (
        "quant/SKILL.md",
        "quant/reference/pipeline_contract.md",
        "quant/scripts/evaluate_compatibility.py",
        "quant/scripts/method_env.py",
        "quant/scripts/write_overlay.py",
        "quant/scripts/validate_script.py",
        "quant-execute/SKILL.md",
        "quant-execute/scripts/launch_job.py",
        "quant-tune/SKILL.md",
        "quant-setup/SKILL.md",
    )
    for relative in relative_paths:
        assert (agents / relative).read_bytes() == (claude / relative).read_bytes()


def test_compatibility_skill_helper_uses_canonical_engine():
    import json

    helper = ROOT / ".agents" / "skills" / "quant" / "scripts" / "evaluate_compatibility.py"
    result = subprocess.run(
        [
            sys.executable, "-B", str(helper), "--params-b", "7", "--vram-gb", "24",
            "--compute-capability", "8.6", "--gpu-arch", "Ampere",
            "--architecture", "LlamaForCausalLM",
        ],
        capture_output=True, text=True, check=True,
    )
    decisions = json.loads(result.stdout)
    assert len(decisions) == 35
    assert {row["status"] for row in decisions} <= {
        "blocked", "eligible", "port_required", "unknown",
    }
    fp8 = next(row for row in decisions if row["method_id"] == "fp8")
    assert fp8["status"] == "blocked"


def test_skill_helpers_parse_and_validate_without_provider_credentials(tmp_path):
    import json
    import os

    env = dict(os.environ)
    env["QUANT_AGENT_WORKSPACE"] = str(tmp_path)
    env.pop("OPENAI_API_KEY", None)

    scripts = [
        ROOT / ".agents" / "skills" / "quant" / "scripts" / "method_env.py",
        ROOT / ".agents" / "skills" / "quant" / "scripts" / "write_overlay.py",
        ROOT / ".agents" / "skills" / "quant" / "scripts" / "validate_script.py",
        ROOT / ".agents" / "skills" / "quant-execute" / "scripts" / "launch_job.py",
    ]
    for script in scripts:
        result = subprocess.run(
            [sys.executable, "-B", str(script), "--help"], env=env,
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr

    candidate = tmp_path / "script.py"
    candidate.write_text("MODEL_ID = 'org/model'\nOUTPUT_DIR = './quantized/awq-org__model'\n")
    validate = ROOT / ".agents" / "skills" / "quant" / "scripts" / "validate_script.py"
    result = subprocess.run(
        [
            sys.executable, "-B", str(validate), str(candidate), "--method-id", "awq",
            "--model-id", "org/model", "--output-dir", "./quantized/awq-org__model",
        ],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True

    patch = tmp_path / "port.patch"
    patch.write_text(
        "diff --git a/modeling.py b/modeling.py\n"
        "--- a/modeling.py\n"
        "+++ b/modeling.py\n"
        "@@ -1 +1,2 @@\n"
        " x = 1\n"
        "+y = 2\n"
    )
    overlay = ROOT / ".agents" / "skills" / "quant" / "scripts" / "write_overlay.py"
    result = subprocess.run(
        [
            sys.executable, "-B", str(overlay),
            "--method-id", "awq", "--model-id", "org/model",
            "--base-commit", "a" * 40, "--patch-file", str(patch),
            "--rationale", "test architecture dispatch",
            "--evidence-file", "modeling.py", "--target-module", "model.layers.*",
        ],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    bundle = Path(payload["overlay_dir"])
    assert bundle.is_dir()
    assert (bundle / "overlay.patch").read_text() == patch.read_text()
    assert (bundle / "manifest.json").is_file()
