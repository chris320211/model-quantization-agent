"""Script write + validate tool for the Adapt agent.

Validation is two-stage:
  1. ``ast.parse`` to catch syntax errors.
  2. Top-level ``import`` dry-run against the method's venv so typos like
     ``import autowaq`` surface before launch. We only import top-level
     modules — never execute module bodies with side effects.

A ``ValidationSession`` holds the per-run retry budget so the Adapt agent
can see ``attempts_left`` in the tool result and decide whether to iterate
or accept a validated-failure write.
"""
from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from ..config import REPO_ROOT, child_env, require_host_execution
from ..executor import venv_python

_MAX_ATTEMPTS = 3
_DRY_IMPORT_TIMEOUT = 30
_DEFAULT_OUTPUT_ROOT = REPO_ROOT / "out"


def _contained(root: Path, path: str) -> Path | None:
    """Resolve ``path`` and return it only if it stays under ``root``; else None.

    Guards write_script against an LLM-supplied absolute path or '../' escape that
    would land a generated script outside the output directory.
    """
    base = root.resolve()
    try:
        candidate = Path(path).resolve()
    except (OSError, RuntimeError):
        return None
    if candidate == base or base in candidate.parents:
        return candidate
    return None


def _collect_top_level_modules(tree: ast.Module) -> list[str]:
    mods: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                mods.append(node.module.split(".")[0])
    seen: set[str] = set()
    ordered: list[str] = []
    for m in mods:
        if m and m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def validate(code: str, method_id: str) -> tuple[bool, str, str]:
    """Return (ok, stage, message). stage in {'parse', 'dry-import', 'ok'}."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, "parse", f"{e.msg} at line {e.lineno}"

    py = venv_python(method_id)
    if not py.exists():
        # Venv not built yet — trust ast.parse. The Adapt agent is expected to
        # build the venv via install_method_venv before the script runs; if it
        # didn't, launch() will surface the error.
        return True, "ok", "dry-import skipped (venv not yet built for this method)"

    modules = _collect_top_level_modules(tree)
    if not modules:
        return True, "ok", "no top-level imports"

    probe = ";".join(f"import {m}" for m in modules)
    require_host_execution("dry-import validation")
    try:
        r = subprocess.run(
            [str(py), "-c", probe],
            capture_output=True,
            text=True,
            timeout=_DRY_IMPORT_TIMEOUT,
            # Importing a module runs its top-level code; give the probe a minimal
            # env so a package can't exfiltrate cloud secrets on import.
            env=child_env(include_hf=False),
        )
    except subprocess.TimeoutExpired:
        return False, "dry-import", f"timed out after {_DRY_IMPORT_TIMEOUT}s"
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-10:]
        return False, "dry-import", "\n".join(tail)
    return True, "ok", f"imported: {', '.join(modules)}"


class ValidationSession:
    """Per-run retry counter for the Adapt agent's write_script tool."""

    def __init__(
        self,
        method_id: str,
        max_attempts: int = _MAX_ATTEMPTS,
        allowed_root: Path | None = None,
    ) -> None:
        self.method_id = method_id
        self.attempts_left = max_attempts
        # Generated scripts may only be written under this directory.
        self.allowed_root = (allowed_root or _DEFAULT_OUTPUT_ROOT)
        self.validated_path: Path | None = None

    def write(self, path: str, code: str) -> dict:
        out = _contained(self.allowed_root, path)
        if out is None:
            return {
                "status": "error",
                "stage": "path",
                "message": f"refusing to write outside {self.allowed_root}: {path!r}",
                "attempts_left": self.attempts_left,
            }

        ok, stage, msg = validate(code, self.method_id)
        if ok:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(code)
            self.validated_path = out
            return {
                "status": "ok",
                "stage": stage,
                "message": msg,
                "path": str(out),
                "attempts_left": self.attempts_left,
            }

        self.attempts_left -= 1
        if self.attempts_left > 0:
            return {
                "status": "error",
                "stage": stage,
                "message": msg,
                "attempts_left": self.attempts_left,
            }

        # Exhausted: fail closed. A known-invalid artifact must never become an
        # executable handoff merely because the retry budget was consumed.
        return {
            "status": "error-exhausted",
            "stage": stage,
            "message": msg,
            "attempts_left": 0,
        }


def make_write_script_tool(session: ValidationSession):
    """Build a @tool bound to this session so retry state is preserved across calls."""

    @tool
    def write_script(path: str, code: str) -> str:
        """Validate Python code (ast.parse + top-level dry-import in the method venv)
        and, on success, write it to `path`. On validation failure, returns an error
        payload with `attempts_left` so you can revise and call again.

        After attempts are exhausted no file is written; the Adapt invocation fails.
        """
        return json.dumps(session.write(path, code), indent=2)

    return write_script
