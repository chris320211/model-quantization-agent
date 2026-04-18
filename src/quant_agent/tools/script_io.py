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

from ..executor import METHOD_TO_VENV, venv_python

_MAX_ATTEMPTS = 3
_DRY_IMPORT_TIMEOUT = 30


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

    venv = METHOD_TO_VENV.get(method_id)
    if venv is None:
        return False, "dry-import", f"No venv mapping for method '{method_id}'"
    py = venv_python(venv)
    if not py.exists():
        return False, "dry-import", (
            f"Venv python not found at {py}. Run scripts/bootstrap_ec2.sh {venv} first."
        )

    modules = _collect_top_level_modules(tree)
    if not modules:
        return True, "ok", "no top-level imports"

    probe = ";".join(f"import {m}" for m in modules)
    try:
        r = subprocess.run(
            [str(py), "-c", probe],
            capture_output=True,
            text=True,
            timeout=_DRY_IMPORT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, "dry-import", f"timed out after {_DRY_IMPORT_TIMEOUT}s"
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-10:]
        return False, "dry-import", "\n".join(tail)
    return True, "ok", f"imported: {', '.join(modules)}"


class ValidationSession:
    """Per-run retry counter for the Adapt agent's write_script tool."""

    def __init__(self, method_id: str, max_attempts: int = _MAX_ATTEMPTS) -> None:
        self.method_id = method_id
        self.attempts_left = max_attempts

    def write(self, path: str, code: str) -> dict:
        ok, stage, msg = validate(code, self.method_id)
        if ok:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(code)
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

        # Exhausted: still write so the user can inspect, with a warning header.
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        header = f"# WARNING: failed validation at stage={stage}: {msg}\n"
        out.write_text(header + code)
        return {
            "status": "error-exhausted",
            "stage": stage,
            "message": msg,
            "path": str(out),
            "attempts_left": 0,
        }


def make_write_script_tool(session: ValidationSession):
    """Build a @tool bound to this session so retry state is preserved across calls."""

    @tool
    def write_script(path: str, code: str) -> str:
        """Validate Python code (ast.parse + top-level dry-import in the method venv)
        and, on success, write it to `path`. On validation failure, returns an error
        payload with `attempts_left` so you can revise and call again.

        After attempts are exhausted the script is written anyway with a warning
        header so the user can inspect the last attempt.
        """
        return json.dumps(session.write(path, code), indent=2)

    return write_script
