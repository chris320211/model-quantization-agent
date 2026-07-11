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
import fcntl
import logging
import os
import re
import shlex
import shutil
import subprocess
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from langchain_core.tools import tool

from ..config import REPO_ROOT, child_env, require_host_execution
from .torch_spec import detect_torch_spec
from ..runtime_deps import RUNTIME_PACKAGES

log = logging.getLogger(__name__)

VENV_ROOT = REPO_ROOT / ".venvs"
_INSTALL_TIMEOUT = 900
_RUN_TIMEOUT_DEFAULT = 120
_OUTPUT_TAIL_LINES = 60
_READ_MAX_BYTES = 20_000
_READ_HARD_MAX_BYTES = 200_000
_LIST_MAX_ENTRIES = 500

# A GitHub repo URL and nothing else — no shell metacharacters, no query string.
_GITHUB_URL_RE = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?$")
_METHOD_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _catalog_repo_urls(method_id: str) -> set[str] | None:
    """Return the set of catalog-declared repo URLs for a method id, or None if the
    id isn't in the catalog. Trailing slashes normalized off."""
    from .recommender import load_catalog  # local import avoids a tools import cycle

    for m in load_catalog():
        if m.get("id") == method_id:
            return {u.rstrip("/") for u in (m.get("repos") or [])}
    return None


def _known_method(method_id: str) -> bool:
    return bool(_METHOD_ID_RE.fullmatch(method_id)) and _catalog_repo_urls(method_id) is not None


@contextmanager
def _method_lock(method_id: str):
    lock_dir = VENV_ROOT / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{method_id}.lock"
    with lock_path.open("a+") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _locked_method(fn):
    @wraps(fn)
    def wrapped(method_id: str, *args, **kwargs):
        if not _known_method(method_id):
            return json.dumps({"status": "error", "error": f"unknown or invalid method_id: {method_id!r}"})
        with _method_lock(method_id):
            return fn(method_id, *args, **kwargs)
    return wrapped


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


def _baseline_packages(python: Path) -> list[list[str]]:
    """Baseline install steps, with a torch pin chosen for the local GPU."""
    spec = detect_torch_spec()
    return [
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel"],
        spec.pip_install_argv(str(python)),
        [
            str(python), "-m", "pip", "install",
            *RUNTIME_PACKAGES,
        ],
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


def _run_argv(
    cmd: list[str],
    cwd: Path | None,
    timeout: int,
    env_extra: dict[str, str] | None = None,
    *,
    include_hf: bool = False,
) -> dict:
    require_host_execution("subprocess command")
    # Allowlisted env only — a cloned repo's install steps / setup.py must never see
    # the parent's cloud secrets (see config.child_env). env_extra (e.g. HF token) merges last.
    env = child_env(env_extra, include_hf=include_hf)
    try:
        r = subprocess.run(
            cmd,
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


_SAFE_ENV_ASSIGNMENTS = {"TORCH_CUDA_ARCH_LIST", "MAX_JOBS"}


def _parse_venv_command(command: str, python: Path) -> tuple[list[str], dict[str, str]]:
    """Turn a narrow command string into argv without invoking a shell.

    Adapt/Fix only need Python entry points and pip. Shell operators, substitutions,
    redirects, and arbitrary executables are deliberately unsupported.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    if any(token in command for token in ("\n", "\r", ";", "&&", "||", "|", ">", "<", "`", "$(")):
        raise ValueError("shell operators and substitutions are not allowed")
    try:
        parts = shlex.split(command, posix=True)
    except ValueError as e:
        raise ValueError(f"invalid command quoting: {e}") from e
    if not parts:
        raise ValueError("command must be non-empty")

    extra: dict[str, str] = {}
    while parts and "=" in parts[0] and not parts[0].startswith(("-", "./", "/")):
        key, value = parts.pop(0).split("=", 1)
        if key not in _SAFE_ENV_ASSIGNMENTS:
            raise ValueError(f"environment assignment {key!r} is not allowed")
        extra[key] = value
    if not parts:
        raise ValueError("command contains no executable")

    executable, *args = parts
    if executable in {"python", "python3", str(python)}:
        return [str(python), *args], extra
    if executable in {"pip", "pip3"}:
        return [str(python), "-m", "pip", *args], extra
    raise ValueError(f"executable {executable!r} is not allowed; use python or pip")


@tool
@_locked_method
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
    # the packaged catalog, so a hijacked candidate can't point at an arbitrary repo.
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
        origin = subprocess.run(
            ["git", "-C", str(repo), "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=15, env=child_env(),
        )
        expected = repo_url.strip().rstrip("/")
        actual = (origin.stdout or "").strip().rstrip("/")
        if origin.returncode != 0 or actual != expected:
            return json.dumps({
                "status": "error",
                "error": f"existing checkout origin mismatch: expected {expected!r}, got {actual!r}",
            })
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15, env=child_env(),
        )
        return json.dumps({
            "status": "ok", "path": str(repo), "already_present": True,
            "commit_sha": (head.stdout or "").strip() or None,
        })

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
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=15, env=child_env(),
    )
    return json.dumps({
        "status": "ok", "path": str(repo), "already_present": False,
        "commit_sha": (head.stdout or "").strip() or None,
    }, indent=2)


@tool
@_locked_method
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
        create = _run_argv(["python3", "-m", "venv", str(venv_dir)], cwd=None, timeout=120)
        if not create["ok"]:
            return json.dumps({"status": "error", "stage": "venv-create", **create}, indent=2)

    results: list[dict] = []
    steps: list[tuple[str, list[str], dict[str, str]]] = [
        (shlex.join(argv), argv, {}) for argv in _baseline_packages(py)
    ]
    for step in install_steps:
        try:
            argv, extra = _parse_venv_command(step, py)
        except ValueError as e:
            return json.dumps({"status": "error", "stage": "command-policy", "error": str(e)})
        steps.append((step, argv, extra))
    for step, argv, extra in steps:
        # Audit trail — install_steps are LLM-chosen and run in a shell by design.
        log.info("install_method_venv[%s] step: %s", method_id, step)
        r = _run_argv(argv, cwd=repo, timeout=_INSTALL_TIMEOUT, env_extra=extra)
        results.append({"step": step, **r})
        if not r["ok"]:
            return json.dumps(
                {"status": "error", "stage": "install", "results": results}, indent=2
            )
    return json.dumps(
        {"status": "ok", "python": str(py), "repo": str(repo), "results": results}, indent=2
    )


@tool
@_locked_method
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

    try:
        argv, env_extra = _parse_venv_command(command, _venv_python(method_id))
    except ValueError as e:
        return json.dumps({"status": "error", "stage": "command-policy", "error": str(e)})
    r = _run_argv(
        argv, cwd=cwd, timeout=timeout, env_extra=env_extra, include_hf=True,
    )
    return json.dumps({"status": "ok" if r["ok"] else "error", **r}, indent=2)


@tool
def list_repo_dir(method_id: str, path: str = "") -> str:
    """List files/directories in the cloned repo (.venvs/<method_id>/repo/<path>).

    Local filesystem read — no GitHub API calls, no rate limits. Use after clone.
    """
    if not _known_method(method_id):
        return json.dumps({"status": "error", "error": f"unknown or invalid method_id: {method_id!r}"})
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
    all_entries = sorted(target.iterdir())
    for p in all_entries[:_LIST_MAX_ENTRIES]:
        entries.append({"name": p.name, "type": "dir" if p.is_dir() else "file"})
    return json.dumps({
        "status": "ok", "path": str(target.relative_to(root)) or ".",
        "entries": entries, "truncated": len(all_entries) > _LIST_MAX_ENTRIES,
    }, indent=2)


@tool
def read_repo_file(method_id: str, path: str, max_bytes: int = _READ_MAX_BYTES) -> str:
    """Read a file from the cloned repo (.venvs/<method_id>/repo/<path>).

    Truncates to max_bytes to keep tool results bounded. Use after clone for source
    inspection.
    """
    if not _known_method(method_id):
        return json.dumps({"status": "error", "error": f"unknown or invalid method_id: {method_id!r}"})
    root = _repo_dir(method_id)
    if not root.exists():
        return json.dumps({"status": "error", "error": f"repo not cloned: {root}"})
    target = _safe_join(root, path)
    if target is None:
        return json.dumps({"status": "error", "error": f"path escapes repo: {path!r}"})
    if not target.exists() or not target.is_file():
        return json.dumps({"status": "error", "error": f"file not found: {target}"})
    max_bytes = max(1, min(int(max_bytes), _READ_HARD_MAX_BYTES))
    with target.open("rb") as f:
        data = f.read(max_bytes + 1)
    truncated = len(data) > max_bytes
    text = data[:max_bytes].decode("utf-8", errors="replace")
    return json.dumps(
        {
            "status": "ok",
            "path": str(target.relative_to(root)),
            "size_bytes": target.stat().st_size,
            "truncated": truncated,
            "content": text,
        },
        indent=2,
    )
