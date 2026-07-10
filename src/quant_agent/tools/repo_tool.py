"""Clone-and-adapt tools for research-repo quantization methods.

For methods that don't ship a pip package (SmoothQuant, FlatQuant, QuaRot, etc.),
the Adapt agent clones the repo, builds a method-specific venv, inspects the
cloned source locally, and writes a wrapper script that invokes the repo's
example entrypoint with the user's model filled in.

Layout per method:
    .venvs/<method_id>/
      bin/python           # the venv
      repo/                # git clone of the method's primary repo
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from ..config import REPO_ROOT, child_env
from .torch_spec import detect_torch_spec

log = logging.getLogger(__name__)

VENV_ROOT = REPO_ROOT / ".venvs"
_INSTALL_TIMEOUT = 900
_RUN_TIMEOUT_DEFAULT = 120
_OUTPUT_TAIL_LINES = 60
_READ_MAX_BYTES = 20_000

# A GitHub repo URL and nothing else — no shell metacharacters, no query string.
_GITHUB_URL_RE = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?$")


def _catalog_repo_urls(method_id: str) -> set[str] | None:
    """Return the set of catalog-declared repo URLs for a method id, or None if the
    id isn't in the catalog. Trailing slashes normalized off."""
    from .recommender import load_catalog  # local import avoids a tools import cycle

    for m in load_catalog():
        if m.get("id") == method_id:
            return {u.rstrip("/") for u in (m.get("repos") or [])}
    return None


def _safe_join(root: Path, rel: str) -> Path | None:
    """Join ``rel`` under ``root`` and confirm the result stays inside ``root``.

    Returns the resolved path, or None if ``rel`` escapes ``root`` (e.g. '../../.env',
    an absolute path, or a symlink target outside the tree).
    """
    base = root.resolve()
    try:
        candidate = (base / rel).resolve()
    except (OSError, RuntimeError):
        return None
    if candidate == base or base in candidate.parents:
        return candidate
    return None


def _baseline_packages() -> list[str]:
    """Baseline install steps, with a torch pin chosen for the local GPU."""
    spec = detect_torch_spec()
    return [
        "pip install --upgrade pip wheel",
        spec.pip_install(),
        "pip install transformers accelerate safetensors sentencepiece",
    ]


def _venv_dir(method_id: str) -> Path:
    return VENV_ROOT / method_id


def _repo_dir(method_id: str) -> Path:
    return _venv_dir(method_id) / "repo"


def _venv_python(method_id: str) -> Path:
    return _venv_dir(method_id) / "bin" / "python"


def _tail(text: str, n: int = _OUTPUT_TAIL_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return "...(%d earlier lines)\n%s" % (len(lines) - n, "\n".join(lines[-n:]))


def _run(cmd: str, cwd: Path | None, timeout: int, env_extra: dict | None = None) -> dict:
    # Allowlisted env only — a cloned repo's install steps / setup.py must never see
    # the parent's cloud secrets (see config.child_env). env_extra (e.g. HF token) merges last.
    env = child_env(env_extra)
    try:
        r = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": _tail(e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")),
            "stderr": f"timed out after {timeout}s",
        }
    return {
        "ok": r.returncode == 0,
        "exit_code": r.returncode,
        "stdout": _tail(r.stdout or ""),
        "stderr": _tail(r.stderr or ""),
    }


@tool
def clone_method_repo(method_id: str, repo_url: str) -> str:
    """Clone a quantization method's GitHub repo into .venvs/<method_id>/repo/.

    Idempotent: if the repo is already cloned, returns its path without re-cloning.
    Returns JSON: {status, path, already_present?, error?}.
    """
    # Validate the URL shape before it reaches any subprocess — an LLM (whose context
    # includes untrusted README/RAG text) supplies repo_url.
    if not _GITHUB_URL_RE.match(repo_url.strip()):
        return json.dumps(
            {"status": "error", "error": f"repo_url is not a plain GitHub repo URL: {repo_url!r}"}
        )
    # Pin to the catalog: the URL must be one this method actually declares in
    # seed/methods.yaml, so a hijacked candidate can't point us at an arbitrary repo.
    known = _catalog_repo_urls(method_id)
    if known is None:
        return json.dumps({"status": "error", "error": f"unknown method_id: {method_id!r}"})
    if repo_url.strip().rstrip("/") not in known:
        return json.dumps(
            {
                "status": "error",
                "error": f"repo_url {repo_url!r} is not a catalog repo for {method_id!r}. "
                f"Known: {sorted(known)}",
            }
        )

    venv_dir = _venv_dir(method_id)
    repo = _repo_dir(method_id)
    venv_dir.mkdir(parents=True, exist_ok=True)

    if repo.exists() and (repo / ".git").exists():
        return json.dumps({"status": "ok", "path": str(repo), "already_present": True})

    if repo.exists():
        shutil.rmtree(repo)

    # argv form (no shell) so the URL can never be interpreted as a command.
    try:
        r = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url.strip(), str(repo)],
            capture_output=True,
            text=True,
            timeout=300,
            env=child_env(include_hf=False),
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "error", "error": "git clone timed out after 300s"})
    if r.returncode != 0:
        return json.dumps(
            {"status": "error", "error": _tail(r.stderr or r.stdout or "clone failed")}, indent=2
        )
    return json.dumps({"status": "ok", "path": str(repo), "already_present": False}, indent=2)


@tool
def install_method_venv(method_id: str, install_steps: list[str]) -> str:
    """Create .venvs/<method_id>/ and run install_steps inside it.

    A baseline of torch + transformers is installed first, then the caller's steps
    run with CWD set to the cloned repo directory (so 'pip install -e .' works).
    Idempotent-ish: the venv is reused if it exists, but install_steps rerun every call.
    Use this after clone_method_repo. Returns JSON with per-step results.
    """
    venv_dir = _venv_dir(method_id)
    repo = _repo_dir(method_id)
    py = _venv_python(method_id)

    if not repo.exists():
        return json.dumps(
            {"status": "error", "error": f"repo not cloned: {repo}. Call clone_method_repo first."}
        )

    if not py.exists():
        create = _run(f"python3 -m venv {venv_dir}", cwd=None, timeout=120)
        if not create["ok"]:
            return json.dumps({"status": "error", "stage": "venv-create", **create}, indent=2)

    activate = f"source {venv_dir}/bin/activate"
    results: list[dict] = []
    for step in _baseline_packages() + list(install_steps):
        # Audit trail — install_steps are LLM-chosen and run in a shell by design.
        log.info("install_method_venv[%s] step: %s", method_id, step)
        cmd = f"{activate} && {step}"
        r = _run(cmd, cwd=repo, timeout=_INSTALL_TIMEOUT)
        results.append({"step": step, **r})
        if not r["ok"]:
            return json.dumps(
                {"status": "error", "stage": "install", "results": results}, indent=2
            )
    return json.dumps(
        {"status": "ok", "python": str(py), "repo": str(repo), "results": results}, indent=2
    )


@tool
def run_in_venv(method_id: str, command: str, cwd_in_repo: str | None = None, timeout: int = _RUN_TIMEOUT_DEFAULT) -> str:
    """Run a shell command inside the method's venv (venv activated, HF_TOKEN exported).

    Use for smoke-testing the entry point (e.g. 'python examples/quant_llama.py --help')
    or inspecting installed packages. CWD defaults to the cloned repo; pass cwd_in_repo
    to chdir into a subdirectory of the repo. Timeout defaults to 120s.
    """
    venv_dir = _venv_dir(method_id)
    repo = _repo_dir(method_id)
    if not _venv_python(method_id).exists():
        return json.dumps(
            {"status": "error", "error": f"venv missing: {venv_dir}. Call install_method_venv first."}
        )
    if cwd_in_repo:
        cwd = _safe_join(repo, cwd_in_repo)
        if cwd is None:
            return json.dumps({"status": "error", "error": f"cwd_in_repo escapes repo: {cwd_in_repo!r}"})
    else:
        cwd = repo
    if not cwd.exists():
        return json.dumps({"status": "error", "error": f"cwd not found: {cwd}"})

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or ""
    env_extra = {"HUGGINGFACE_HUB_TOKEN": hf_token, "HF_TOKEN": hf_token} if hf_token else None

    cmd = f"source {venv_dir}/bin/activate && {command}"
    r = _run(cmd, cwd=cwd, timeout=timeout, env_extra=env_extra)
    return json.dumps({"status": "ok" if r["ok"] else "error", **r}, indent=2)


@tool
def list_repo_dir(method_id: str, path: str = "") -> str:
    """List files/directories in the cloned repo (.venvs/<method_id>/repo/<path>).

    Local filesystem read — no GitHub API calls, no rate limits. Use after clone.
    """
    root = _repo_dir(method_id)
    if not root.exists():
        return json.dumps({"status": "error", "error": f"repo not cloned: {root}"})
    target = _safe_join(root, path)
    if target is None:
        return json.dumps({"status": "error", "error": f"path escapes repo: {path!r}"})
    if not target.exists():
        return json.dumps({"status": "error", "error": f"path not found: {target}"})
    if not target.is_dir():
        return json.dumps({"status": "error", "error": f"not a directory: {target}"})
    entries = []
    for p in sorted(target.iterdir()):
        entries.append({"name": p.name, "type": "dir" if p.is_dir() else "file"})
    return json.dumps({"status": "ok", "path": str(target.relative_to(root)) or ".", "entries": entries}, indent=2)


@tool
def read_repo_file(method_id: str, path: str, max_bytes: int = _READ_MAX_BYTES) -> str:
    """Read a file from the cloned repo (.venvs/<method_id>/repo/<path>).

    Truncates to max_bytes to keep tool results bounded. Use after clone for source
    inspection.
    """
    root = _repo_dir(method_id)
    if not root.exists():
        return json.dumps({"status": "error", "error": f"repo not cloned: {root}"})
    target = _safe_join(root, path)
    if target is None:
        return json.dumps({"status": "error", "error": f"path escapes repo: {path!r}"})
    if not target.exists() or not target.is_file():
        return json.dumps({"status": "error", "error": f"file not found: {target}"})
    data = target.read_bytes()
    truncated = len(data) > max_bytes
    text = data[:max_bytes].decode("utf-8", errors="replace")
    return json.dumps(
        {
            "status": "ok",
            "path": str(target.relative_to(root)),
            "size_bytes": len(data),
            "truncated": truncated,
            "content": text,
        },
        indent=2,
    )
