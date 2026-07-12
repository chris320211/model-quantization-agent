import json

import pytest

from quant_agent.port_overlay import (
    PortOverlaySession,
    directory_sha256,
    make_write_port_overlay_tool,
    overlay_path_from_script,
    validate_overlay_script,
    validate_unified_patch,
)


PATCH = """diff --git a/quant/modeling.py b/quant/modeling.py
--- a/quant/modeling.py
+++ b/quant/modeling.py
@@ -1 +1,2 @@
 SUPPORTED = [\"llama\"]
+SUPPORTED.append(\"qwen2\")
"""


def test_patch_validation_rejects_escape_binary_and_rename():
    with pytest.raises(ValueError, match="unsafe overlay path"):
        validate_unified_patch("diff --git a/../x b/../x\n--- a/../x\n+++ b/../x\n")
    with pytest.raises(ValueError, match="binary"):
        validate_unified_patch(PATCH + "GIT binary patch\n")
    with pytest.raises(ValueError, match="renames"):
        validate_unified_patch("diff --git a/a.py b/b.py\n--- a/a.py\n+++ b/b.py\n")


def test_overlay_session_writes_content_addressed_bundle(tmp_path):
    session = PortOverlaySession(
        root=tmp_path, method_id="awq", model_id="org/model", base_commit="abc",
    )
    tool = make_write_port_overlay_tool(session)
    payload = json.loads(tool.invoke({
        "patch": PATCH,
        "rationale": "Add Qwen2 dispatch using inspected target modules.",
        "evidence_files": ["quant/modeling.py"],
        "target_modules": ["model.layers.*.self_attn.q_proj"],
    }))
    assert payload["status"] == "ok"
    assert session.overlay_dir is not None
    assert (session.overlay_dir / "overlay.patch").read_text() == PATCH
    assert directory_sha256(session.overlay_dir)
    assert json.loads((session.overlay_dir / "manifest.json").read_text())["model_id"] == "org/model"


def test_port_script_contract_requires_executor_managed_environment(tmp_path):
    session = PortOverlaySession(
        root=tmp_path / "overlays", method_id="awq", model_id="org/model", base_commit=None,
    )
    session.write(patch=PATCH, rationale="Test overlay contract.")
    overlay = session.overlay_dir
    assert overlay is not None
    code = f'''# QUANT_AGENT_OVERLAY_DIR={overlay}
import os
OVERLAY = os.environ["QUANT_AGENT_OVERLAY_DIR"]
REPO = os.environ["QUANT_AGENT_METHOD_REPO"]
'''
    validate_overlay_script(code, overlay)
    assert overlay_path_from_script(code) == overlay.resolve()
    with pytest.raises(ValueError, match="environment usage"):
        validate_overlay_script(
            f"# QUANT_AGENT_OVERLAY_DIR={overlay}\nOVERLAY = '{overlay}'\n", overlay
        )
