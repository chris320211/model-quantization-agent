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
    "quant-benchmark",
    "quant-diagnose",
    "quant-kernel",
    "quant-publish",
)
REQUIRED_SCRIPTS = (
    "adapter.py",
    "benchmark.py",
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
    "diagnose.py",
    "publish.py",
    "compare.py",
)


def test_skill_docs_have_no_obsolete_codex_paths():
    for path in (ROOT / ".agents" / "skills").glob("*/SKILL.md"):
        text = path.read_text()
        assert ".Codex/" not in text
        assert "quant-agent ask" not in text
        assert "quant-agent setup" not in text
    contract = (ROOT / ".agents" / "skills" / "_shared" / "pipeline_contract.md").read_text()
    assert "Retry loop (parent)" in contract
    assert "retry_gpu_jobs_max" in contract
    assert "issue_codes" in contract
    subagents = (ROOT / ".agents" / "skills" / "_shared" / "subagents.md").read_text()
    assert "One stage per subagent" in subagents
    assert "quant-setup" in subagents
    assert "recommended_action" in subagents
    assert "compare/" in subagents
    parent = (ROOT / ".agents" / "skills" / "quant" / "SKILL.md").read_text()
    assert "Retry loop (you)" in parent
    assert "quant-retry" in parent
    assert "every method × model × GPU" in parent
    assert "compare/" in parent
    assert "LOOP:" in parent
    diagnose = (ROOT / ".agents" / "skills" / "quant-diagnose" / "SKILL.md").read_text()
    assert "method-agnostic" in diagnose
    assert "1.5" in diagnose
    gather = (ROOT / ".agents" / "skills" / "quant-gather" / "SKILL.md").read_text()
    assert "retry_gpu_jobs_max=2" in gather
    kernel = (ROOT / ".agents" / "skills" / "quant-kernel" / "SKILL.md").read_text()
    assert "pack_i4" in kernel
    assert "Linear4bit" in kernel
    assert "cuda_kernel_dtype_mismatch" in kernel
    assert "scaled_dot_product_attention" in kernel or "kron_matmul" in kernel
    assert "0-d tensors" in kernel or "Python float" in kernel
    assert "eval_runtime_flags_missing" in diagnose
    assert "prefill_kernel_missing" in diagnose
    assert "restore_packed_eval_runtime" in kernel or "use_diag" in kernel
    assert "recommended_action" in parent
    assert "inspect overlay" in parent or "Never inspect overlay" in parent
    assert "cuda_kernel_dtype_mismatch" in diagnose
    port = (ROOT / ".agents" / "skills" / "quant-port" / "SKILL.md").read_text()
    assert "cast_linear4bit_kernel_dtypes" in port or "float16" in kernel
    assert "quantized_save" in port
    assert "pack_i4" in contract
    assert "cuda_kernel_dtype_mismatch" in contract
    assert "compare/" in contract
    assert "catalog.json" in contract
    assert "contributions" in contract
    assert "model_id" in contract and "gpu_instance" in contract
    publish = (ROOT / ".agents" / "skills" / "quant-publish" / "SKILL.md").read_text()
    assert "compare/" in publish
    assert "artifact store" in publish.lower() or "not the comparison" in publish.lower()



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


def test_skills_resolve_python_or_python3():
    marker = "command -v python || command -v python3"
    for name in SKILL_NAMES:
        text = (ROOT / ".agents" / "skills" / name / "SKILL.md").read_text()
        assert marker in text, name
    for relative in ("_shared/pipeline_contract.md", "_shared/subagents.md"):
        text = (ROOT / ".agents" / "skills" / relative).read_text()
        assert marker in text, relative
    readme = (ROOT / "README.md").read_text()
    assert marker in readme
    contract = (ROOT / ".agents" / "skills" / "_shared" / "pipeline_contract.md").read_text()
    assert "drops_runtime" in contract


def test_skills_point_at_shared_scripts():
    for name in ("quant-gather", "quant-port", "quant-run", "quant-verify", "quant-benchmark", "quant-diagnose", "quant-kernel", "quant-publish"):
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
