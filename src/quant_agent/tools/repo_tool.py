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
import os
import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from ..config import REPO_ROOT

VENV_ROOT = REPO_ROOT / ".venvs"
_INSTALL_TIMEOUT = 900
_RUN_TIMEOUT_DEFAULT = 120
_OUTPUT_TAIL_LINES = 60
_READ_MAX_BYTES = 20_000

_BASELINE_PACKAGES = [
    "pip install --upgrade pip wheel",
    "pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1",
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
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
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
    venv_dir = _venv_dir(method_id)
    repo = _repo_dir(method_id)
    venv_dir.mkdir(parents=True, exist_ok=True)

    if repo.exists() and (repo / ".git").exists():
        return json.dumps({"status": "ok", "path": str(repo), "already_present": True})

    if repo.exists():
        shutil.rmtree(repo)

    result = _run(f"git clone --depth 1 {repo_url} {repo}", cwd=None, timeout=300)
    if not result["ok"]:
        return json.dumps(
            {"status": "error", "error": result["stderr"] or result["stdout"]}, indent=2
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
    for step in _BASELINE_PACKAGES + list(install_steps):
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
    cwd = repo / cwd_in_repo if cwd_in_repo else repo
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
    target = root / path
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
    target = root / path
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
