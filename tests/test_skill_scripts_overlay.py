from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agents" / "skills" / "_shared" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from adapter import (  # noqa: E402
    InvalidInferenceAdapter,
    adapter_loads_hub_id,
    validate_adapter_source,
)
import overlay as overlay_mod  # noqa: E402
from overlay import PortOverlaySession, validate_unified_patch  # noqa: E402
import verify as verify_mod  # noqa: E402


PATCH = """diff --git a/quant/modeling.py b/quant/modeling.py
--- a/quant/modeling.py
+++ b/quant/modeling.py
@@ -1 +1,2 @@
 SUPPORTED = [\"llama\"]
+SUPPORTED.append(\"qwen2\")
"""

COMMIT = "a" * 40

VALID_ADAPTER = '''QUANT_AGENT_ADAPTER_API = 1

def load_model_and_tokenizer(*, model_path, model_id, dtype, device, trust_remote_code):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, None
'''

HUB_ADAPTER = '''QUANT_AGENT_ADAPTER_API = 1

def load_model_and_tokenizer(*, model_path, model_id, dtype, device, trust_remote_code):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return model, None
'''

AUTOMODEL_HUB_ADAPTER = '''QUANT_AGENT_ADAPTER_API = 1

def load_model_and_tokenizer(*, model_path, model_id, dtype, device, trust_remote_code):
    return AutoModelForCausalLM(model_id), None
'''


def _git(*args, cwd=None):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_adapter_module_omits_capability_helpers():
    import adapter as adapter_mod

    assert not hasattr(adapter_mod, "has_generic_adapter")
    assert not hasattr(adapter_mod, "load_capabilities")
    assert not hasattr(adapter_mod, "requires_generated_adapter")


@pytest.mark.parametrize(
    "source, message",
    [
        ("def load_model_and_tokenizer():\n    pass\n", "ADAPTER_API"),
        (
            "QUANT_AGENT_ADAPTER_API = 1\n"
            "def load_model_and_tokenizer(model_path):\n    pass\n",
            "missing parameters",
        ),
        (
            "QUANT_AGENT_ADAPTER_API = 1\n"
            "MODEL = download_model()\n"
            "def load_model_and_tokenizer(*, model_path, model_id, dtype, device, trust_remote_code):\n"
            "    return MODEL, None\n",
            "top-level assignments",
        ),
    ],
)
def test_adapter_missing_api_is_rejected(source, message):
    with pytest.raises(InvalidInferenceAdapter, match=message):
        validate_adapter_source(source)


def test_adapter_loads_hub_id_true_for_from_pretrained_and_automodel():
    assert adapter_loads_hub_id(HUB_ADAPTER) is True
    assert adapter_loads_hub_id(AUTOMODEL_HUB_ADAPTER) is True
    assert adapter_loads_hub_id(
        "QUANT_AGENT_ADAPTER_API = 1\n"
        "def load_model_and_tokenizer(*, model_path, model_id, dtype, device, trust_remote_code):\n"
        "    return from_pretrained(model_id), None\n"
    ) is True


def test_adapter_loads_hub_id_false_when_loading_model_path():
    assert adapter_loads_hub_id(VALID_ADAPTER) is False
    validate_adapter_source(VALID_ADAPTER)
    assert adapter_loads_hub_id(
        "QUANT_AGENT_ADAPTER_API = 1\n"
        "def load_model_and_tokenizer(*, model_path, model_id, dtype, device, trust_remote_code):\n"
        "    note = model_id\n"
        "    return helper(model_path), note\n"
    ) is False


def test_adapter_loads_hub_id_allows_model_eval():
    source = (
        "QUANT_AGENT_ADAPTER_API = 1\n"
        "def load_model_and_tokenizer(*, model_path, model_id, dtype, device, trust_remote_code):\n"
        "    model = AutoModelForCausalLM.from_pretrained(model_path)\n"
        "    model.eval()\n"
        "    return model, None\n"
    )
    assert adapter_loads_hub_id(source) is False
    validate_adapter_source(source)


def test_adapter_loads_hub_id_catches_literals_aliases_and_kwargs():
    assert adapter_loads_hub_id(
        "QUANT_AGENT_ADAPTER_API = 1\n"
        "def load_model_and_tokenizer(*, model_path, model_id, dtype, device, trust_remote_code):\n"
        "    return AutoModelForCausalLM.from_pretrained('org/model'), None\n"
    ) is True
    assert adapter_loads_hub_id(
        "QUANT_AGENT_ADAPTER_API = 1\n"
        "def load_model_and_tokenizer(*, model_path, model_id, dtype, device, trust_remote_code):\n"
        "    name = model_id\n"
        "    return AutoModelForCausalLM.from_pretrained(name), None\n"
    ) is True
    assert adapter_loads_hub_id(
        "QUANT_AGENT_ADAPTER_API = 1\n"
        "def load_model_and_tokenizer(*, model_path, model_id, dtype, device, trust_remote_code):\n"
        "    return AutoModelForCausalLM.from_pretrained(**{'pretrained_model_name_or_path': model_id}), None\n"
    ) is True


def test_symlink_overlay_patch_is_rejected():
    patch = (
        "diff --git a/evil b/evil\n"
        "new file mode 120000\n"
        "--- /dev/null\n"
        "+++ b/evil\n"
        "@@ -0,0 +1 @@\n"
        "+/etc/passwd\n"
    )
    with pytest.raises(ValueError, match="symlink"):
        validate_unified_patch(patch)


def test_safe_path_component_rejects_dotdot():
    with pytest.raises(ValueError, match="unsafe slug"):
        overlay_mod.safe_path_component("..", "slug")
    with pytest.raises(ValueError, match="unsafe strategy"):
        overlay_mod.safe_path_component("foo/bar", "strategy")


def test_patch_validation_rejects_escape_binary_and_rename():
    with pytest.raises(ValueError, match="unsafe overlay path"):
        validate_unified_patch("diff --git a/../x b/../x\n--- a/../x\n+++ b/../x\n")
    with pytest.raises(ValueError, match="unsafe overlay path"):
        validate_unified_patch("diff --git a/.git/config b/.git/config\n--- a/.git/config\n+++ b/.git/config\n")
    with pytest.raises(ValueError, match="unsafe overlay path"):
        validate_unified_patch("diff --git a/.env b/.env\n--- a/.env\n+++ b/.env\n")
    with pytest.raises(ValueError, match="binary"):
        validate_unified_patch(PATCH + "GIT binary patch\n")
    with pytest.raises(ValueError, match="renames"):
        validate_unified_patch("diff --git a/a.py b/b.py\n--- a/a.py\n+++ b/b.py\n")


def test_overlay_session_requires_40_char_commit(tmp_path):
    with pytest.raises(ValueError, match="40-character"):
        PortOverlaySession(
            root=tmp_path, method_id="slug", model_id="org/model", base_commit=None,
        )
    with pytest.raises(ValueError, match="40-character"):
        PortOverlaySession(
            root=tmp_path, method_id="slug", model_id="org/model", base_commit="abc",
        )


def test_overlay_session_writes_slug_strategy_hash_bundle(tmp_path):
    root = tmp_path / "out" / "overlays" / "awq-qwen" / "dispatch"
    session = PortOverlaySession(
        root=root, method_id="awq-qwen", model_id="org/model", base_commit=COMMIT,
    )
    payload = session.write(patch=PATCH, rationale="Add Qwen2 dispatch.")
    assert payload["status"] == "ok"
    overlay_dir = Path(payload["overlay_dir"])
    assert overlay_dir.parent == root
    assert (overlay_dir / "overlay.patch").read_text() == PATCH
    manifest = json.loads((overlay_dir / "manifest.json").read_text())
    assert manifest["base_commit"] == COMMIT
    assert manifest["model_id"] == "org/model"


def test_overlay_rejects_adapter_path_not_present_in_patch(tmp_path):
    session = PortOverlaySession(
        root=tmp_path, method_id="slug", model_id="org/model", base_commit=COMMIT,
    )
    payload = session.write(
        patch=PATCH,
        rationale="Bad adapter declaration.",
        inference_adapter_path="missing_adapter.py",
    )
    assert payload["status"] == "error"
    assert "included in the unified diff" in payload["message"]


def test_overlay_write_rejects_adapter_missing_api(tmp_path):
    adapter_path = "quant_agent_inference_adapter.py"
    patch = f"""diff --git a/{adapter_path} b/{adapter_path}
new file mode 100644
--- /dev/null
+++ b/{adapter_path}
@@ -0,0 +1,2 @@
+def load_model_and_tokenizer(*, model_path, model_id, dtype, device, trust_remote_code):
+    return None, None
"""
    session = PortOverlaySession(
        root=tmp_path, method_id="slug", model_id="org/model", base_commit=COMMIT,
    )
    payload = session.write(
        patch=patch,
        rationale="Invalid adapter.",
        inference_adapter_path=adapter_path,
    )
    assert payload["status"] == "error"
    assert "ADAPTER_API" in payload["message"]


def test_overlay_refuses_to_write_into_venvs(tmp_path, monkeypatch):
    venvs = tmp_path / ".venvs"
    venvs.mkdir()
    monkeypatch.setattr(overlay_mod.paths, "VENV_ROOT", venvs)
    with pytest.raises(RuntimeError, match="\\.venvs"):
        overlay_mod._assert_not_venv_write(venvs / "slug" / "repo" / "model.py")


def test_overlay_cli_write_and_check(tmp_path):
    env = dict(os.environ)
    env["QUANT_AGENT_WORKSPACE"] = str(tmp_path)
    patch = tmp_path / "port.patch"
    patch.write_text(PATCH)
    result = subprocess.run(
        [
            sys.executable, "-B", str(SCRIPTS / "overlay.py"), "write",
            "--slug", "awq-qwen", "--strategy", "dispatch",
            "--model-id", "org/model", "--base-commit", COMMIT,
            "--patch-file", str(patch), "--rationale", "Add architecture dispatch.",
            "--evidence-file", "quant/modeling.py",
        ],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    overlay_dir = Path(payload["overlay_dir"])
    assert overlay_dir.parent == tmp_path / "out" / "overlays" / "awq-qwen" / "dispatch"
    assert (overlay_dir / "overlay.patch").is_file()
    assert (overlay_dir / "manifest.json").is_file()

    checked = subprocess.run(
        [
            sys.executable, "-B", str(SCRIPTS / "overlay.py"), "check",
            "--overlay-dir", str(overlay_dir),
        ],
        env=env, capture_output=True, text=True, check=False,
    )
    assert checked.returncode == 0, checked.stderr + checked.stdout
    check_payload = json.loads(checked.stdout)
    assert check_payload["status"] == "ok"
    assert "quant/modeling.py" in check_payload["patched_paths"]


def test_apply_check_without_git_validates_patch_text(tmp_path, monkeypatch):
    session = PortOverlaySession(
        root=tmp_path, method_id="slug", model_id="org/model", base_commit=COMMIT,
    )
    payload = session.write(patch=PATCH, rationale="Dispatch.")
    overlay_dir = Path(payload["overlay_dir"])
    monkeypatch.setattr(overlay_mod.shutil, "which", lambda _: None)
    result = overlay_mod.apply_check(overlay_dir)
    assert result["status"] == "ok"
    assert result["mode"] == "patch-text"
    with pytest.raises(RuntimeError, match="git is required"):
        overlay_mod.apply_check(overlay_dir, repo=tmp_path / "repo")


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
def test_apply_check_does_not_modify_canonical_clone(tmp_path):
    repo = tmp_path / "method-repo"
    repo.mkdir()
    _git("init", cwd=repo)
    source = repo / "quant" / "modeling.py"
    source.parent.mkdir()
    source.write_text('SUPPORTED = ["llama"]\n')
    _git("add", "quant/modeling.py", cwd=repo)
    _git(
        "-c", "user.name=Test", "-c", "user.email=test@example.com",
        "commit", "-m", "initial", cwd=repo,
    )
    commit = _git("rev-parse", "HEAD", cwd=repo)
    session = PortOverlaySession(
        root=tmp_path / "overlays", method_id="slug", model_id="org/model", base_commit=commit,
    )
    payload = session.write(patch=PATCH, rationale="Add Qwen2.")
    overlay_dir = Path(payload["overlay_dir"])
    before = source.read_text()
    result = overlay_mod.apply_check(overlay_dir, repo=repo, commit=commit)
    assert result["status"] == "ok"
    assert result["mode"] == "git-apply"
    assert source.read_text() == before
    assert "qwen2" not in source.read_text()


def test_job_id_regex_matches_executor_format():
    import ast
    import jobs as jobs_mod
    from launch import default_output_dir

    assert jobs_mod.valid_job_id("20260710T120000Z-abcdef")
    assert not jobs_mod.valid_job_id("../../etc")
    assert not jobs_mod.valid_job_id("foo/bar")
    assert default_output_dir("awq-qwen") == "./quantized/awq-qwen"
    ast.parse(verify_mod.SMOKE_SCRIPT)


def test_authenticity_rejects_snapshot_and_empty_output(tmp_path, monkeypatch):
    snapshot = tmp_path / "hf-snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    empty = tmp_path / "quantized-empty"
    empty.mkdir()
    real = tmp_path / "quantized"
    real.mkdir()
    (real / "weights.safetensors").write_text("x")
    monkeypatch.setattr(verify_mod.paths, "REPO_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="Hugging Face snapshot"):
        verify_mod.check_authenticity(output_dir=snapshot, hf_snapshot_path=snapshot)
    with pytest.raises(RuntimeError, match="missing or empty"):
        verify_mod.check_authenticity(output_dir=empty, hf_snapshot_path=snapshot)
    with pytest.raises(RuntimeError, match="Hub model_id"):
        verify_mod.check_authenticity(
            output_dir=real, hf_snapshot_path=snapshot, adapter_source=HUB_ADAPTER,
        )
    verify_mod.check_authenticity(
        output_dir=real, hf_snapshot_path=snapshot, adapter_source=VALID_ADAPTER,
    )
    clone = tmp_path / "quantized-clone"
    clone.mkdir()
    (clone / "weights.safetensors").write_text("x")
    twin = tmp_path / "hf-twin"
    twin.mkdir()
    (twin / "weights.safetensors").write_text("x")
    with pytest.raises(RuntimeError, match="match the Hugging Face snapshot"):
        verify_mod.check_authenticity(output_dir=clone, hf_snapshot_path=twin)


def test_child_env_rejects_unknown_extra_and_strips_home_without_hf():
    import env as env_mod

    with pytest.raises(ValueError, match="not allowed"):
        env_mod.child_env({"MEASURE_SMOKE_ONLY": "1", "OPENAI_API_KEY": "sk"}, include_hf=False)
    built = env_mod.child_env(include_hf=False)
    assert "HF_TOKEN" not in built
    assert "HUGGINGFACE_HUB_TOKEN" not in built
    assert built.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == "1"


def test_validate_script_requires_literals_and_overlay_header(tmp_path):
    env = dict(os.environ)
    env["QUANT_AGENT_WORKSPACE"] = str(tmp_path)
    patch = tmp_path / "port.patch"
    patch.write_text(PATCH)
    write = subprocess.run(
        [
            sys.executable, "-B", str(SCRIPTS / "overlay.py"), "write",
            "--slug", "awq-qwen", "--strategy", "dispatch",
            "--model-id", "org/model", "--base-commit", COMMIT,
            "--patch-file", str(patch), "--rationale", "Dispatch.",
            "--target-module", "quant.modeling",
        ],
        env=env, capture_output=True, text=True, check=False,
    )
    assert write.returncode == 0, write.stderr + write.stdout
    overlay_dir = Path(json.loads(write.stdout)["overlay_dir"])
    script = tmp_path / "quantize.py"
    script.write_text(
        f"# QUANT_AGENT_OVERLAY_DIR={overlay_dir.resolve()}\n"
        "MODEL_ID = 'org/model'\n"
        "OUTPUT_DIR = './quantized/awq-qwen'\n"
        "os.environ['QUANT_AGENT_OVERLAY_DIR']\n"
        "os.environ['QUANT_AGENT_METHOD_REPO']\n"
    )
    result = subprocess.run(
        [
            sys.executable, "-B", str(SCRIPTS / "validate_script.py"), str(script),
            "--model-id", "org/model", "--output-dir", "./quantized/awq-qwen",
            "--overlay-dir", str(overlay_dir),
        ],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)["ok"] is True

    bad = tmp_path / "bad.py"
    bad.write_text("MODEL_ID = 'other'\nOUTPUT_DIR = './quantized/awq-qwen'\n")
    failed = subprocess.run(
        [
            sys.executable, "-B", str(SCRIPTS / "validate_script.py"), str(bad),
            "--model-id", "org/model", "--output-dir", "./quantized/awq-qwen",
        ],
        env=env, capture_output=True, text=True, check=False,
    )
    assert failed.returncode != 0
    assert json.loads(failed.stdout)["ok"] is False

