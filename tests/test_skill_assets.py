from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".agents" / "skills" / "_shared" / "scripts"
SKILL_NAMES = (
    "quant",
    "quant-setup",
    "quant-gather",
    "quant-port",
    "quant-run",
    "quant-verify",
    "quant-kernel",
)
REQUIRED_SCRIPTS = (
    "adapter.py",
    "clone.py",
    "env.py",
    "gpu.py",
    "install_venv.py",
    "jobs.py",
    "launch.py",
    "overlay.py",
    "paper.py",
    "paths.py",
    "refuse.py",
    "request.py",
    "snapshot.py",
    "validate_script.py",
    "verify.py",
)


def test_skill_docs_have_no_obsolete_codex_paths():
    for path in (ROOT / ".agents" / "skills").glob("*/SKILL.md"):
        text = path.read_text()
        assert ".Codex/" not in text
        assert "quant-agent ask" not in text
        assert "quant-agent setup" not in text
    subagents = (ROOT / ".agents" / "skills" / "_shared" / "subagents.md").read_text()
    assert "One stage per subagent" in subagents
    assert "quant-setup" in subagents


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


def test_skill_mirrors_match_exactly():
    agents = ROOT / ".agents" / "skills"
    for tree in (ROOT / ".claude" / "skills", ROOT / ".cursor" / "skills"):
        for name in SKILL_NAMES:
            relative = f"{name}/SKILL.md"
            assert (agents / relative).read_bytes() == (tree / relative).read_bytes()
        for relative in (
            "_shared/pipeline_contract.md",
            "_shared/subagents.md",
            "_shared/load_env.sh",
        ):
            assert (agents / relative).read_bytes() == (tree / relative).read_bytes()


def test_shared_scripts_exist_and_do_not_import_quant_agent():
    for name in REQUIRED_SCRIPTS:
        path = SCRIPTS / name
        assert path.is_file(), name
        text = path.read_text()
        assert "import quant_agent" not in text
        assert "from quant_agent" not in text


def test_skills_point_at_shared_scripts():
    for name in ("quant-gather", "quant-port", "quant-run", "quant-verify", "quant-kernel"):
        text = (ROOT / ".agents" / "skills" / name / "SKILL.md").read_text()
        assert ".agents/skills/_shared/scripts" in text
        assert "out/ports/" not in text
    port = (ROOT / ".agents" / "skills" / "quant-port" / "SKILL.md").read_text()
    kernel = (ROOT / ".agents" / "skills" / "quant-kernel" / "SKILL.md").read_text()
    assert "out/overlays/" in port
    assert "kernel_triton" in kernel


def test_obsolete_catalog_product_is_gone():
    for tree in (
        ROOT / ".agents" / "skills",
        ROOT / ".claude" / "skills",
        ROOT / ".cursor" / "skills",
    ):
        assert not (tree / "quant-execute").exists()
        assert not (tree / "quant-tune").exists()
        assert not (tree / "quant" / "reference" / "methods.yaml").exists()
    assert not (ROOT / "src" / "quant_agent").exists()
    assert not (ROOT / "docs").exists()
    assert not (ROOT / "reports").exists()
    assert not (ROOT / "agent_flowchart.html").exists()
    assert not (ROOT / "scripts").exists()
    assert (ROOT / ".agents" / "skills" / "_shared" / "reference" / "aws_instances.yaml").is_file()
    assert (ROOT / ".agents" / "skills" / "_shared" / "reference" / "gpu_specs.yaml").is_file()
    readme = (ROOT / "README.md").read_text()
    assert "quant-agent ask" not in readme
    assert "quant-agent setup" not in readme
