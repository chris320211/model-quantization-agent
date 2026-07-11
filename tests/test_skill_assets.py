from __future__ import annotations

from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_skill_catalogs_match_packaged_catalogs():
    canonical = ROOT / "src" / "quant_agent" / "data"
    for mirror in (
        ROOT / ".agents" / "skills" / "quant" / "reference",
        ROOT / ".claude" / "skills" / "quant" / "reference",
    ):
        for name in ("methods.yaml", "model_aliases.yaml", "aws_instances.yaml", "gpu_specs.yaml"):
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
