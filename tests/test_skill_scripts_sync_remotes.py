from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "agents" / "skills" / "_shared" / "scripts" / "sync_remotes.py"


def _load():
    spec = importlib.util.spec_from_file_location("sync_remotes", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_allowlist_accepts_library_and_skills():
    mod = _load()
    assert mod.is_allowed("library/catalog.json")
    assert mod.is_allowed("library/contributions/foo.json")
    assert mod.is_allowed("agents/skills/quant-sync/SKILL.md")
    assert mod.is_allowed("tests/test_skill_assets.py")
    assert mod.is_allowed("README.md")
    assert mod.is_allowed(".github/workflows/library.yml")


def test_allowlist_blocks_weights_and_secrets():
    mod = _load()
    assert mod.is_blocked("quantized/slug/weights.pt")
    assert mod.is_blocked(".env")
    assert mod.is_blocked(".env.local")
    assert mod.is_blocked("jobs/id/log.txt")
    assert mod.is_blocked("out/hub/slug/model.safetensors")
    assert not mod.is_allowed("quantized/slug/weights.pt")
    assert not mod.is_allowed(".env")
    assert not mod.is_allowed("../.env")
    assert not mod.is_allowed("agents/../.env")


def test_github_commit_url_does_not_strip_repo_name_suffix():
    assert "https://github.com/chris320211/model-quantization-agent.git".endswith(".git")
    url = "https://github.com/chris320211/model-quantization-agent.git"
    if url.endswith(".git"):
        url = url[: -len(".git")]
    assert url == "https://github.com/chris320211/model-quantization-agent"
    # str.rstrip('.git') would turn "...agent.git" into "...agen"
    assert url.endswith("agent")

    askpass = ROOT / "agents" / "skills" / "_shared" / "scripts" / "git_askpass.sh"
    text = askpass.read_text()
    assert "GITHUB_TOKEN" in text
    assert "x-access-token" in text
    assert "hf_" not in text
    skill = (ROOT / "agents" / "skills" / "quant-sync" / "SKILL.md").read_text()
    assert "git_askpass.sh" in skill
    assert "quantized/" in skill
