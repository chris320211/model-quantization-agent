"""Script write + validate tool for the Adapt agent.

Validation is staged and fail-closed:
  1. ``ast.parse`` catches syntax errors.
  2. A top-level ``import`` probe against the method's venv catches typos like
     ``import autowaq`` before launch.
  3. Optional static semantic checks prove that run-specific model, output, and
     tune-locked values occur in the generated program.
  4. An optional, explicitly configured command can smoke-test a temporary copy
     of the script with a bounded timeout.

A ``ValidationSession`` holds the per-run retry budget so the Adapt agent
can see ``attempts_left`` in the tool result and decide whether to iterate
or accept a validated-failure write.
"""
from __future__ import annotations

import ast
import json
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from langchain_core.tools import tool

from ..config import REPO_ROOT, child_env, require_host_execution
from ..executor import venv_python

_MAX_ATTEMPTS = 3
_DRY_IMPORT_TIMEOUT = 30
_DEFAULT_SMOKE_TIMEOUT = 120
_MAX_SMOKE_TIMEOUT = 300
_DEFAULT_OUTPUT_ROOT = REPO_ROOT / "out"


class ValidationStage(str, Enum):
    """Stable stage identifiers used in tool payloads and typed reports."""

    PARSE = "parse"
    DRY_IMPORT = "dry-import"
    STATIC_SEMANTICS = "static-semantics"
    SMOKE = "smoke"
    OK = "ok"


class CheckStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class StageResult:
    """Result for one validation stage."""

    stage: ValidationStage
    status: CheckStatus
    message: str

    @property
    def ok(self) -> bool:
        return self.status is not CheckStatus.FAILED


@dataclass(frozen=True)
class ValidationResult:
    """Typed result containing all stages reached before success or failure."""

    checks: tuple[StageResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def stage(self) -> ValidationStage:
        for check in self.checks:
            if check.status is CheckStatus.FAILED:
                return check.stage
        return ValidationStage.OK

    @property
    def message(self) -> str:
        if not self.checks:
            return "no validation stages ran"
        for check in self.checks:
            if check.status is CheckStatus.FAILED:
                return check.message
        return "; ".join(check.message for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stage": self.stage.value,
            "message": self.message,
            "checks": [
                {
                    "stage": check.stage.value,
                    "status": check.status.value,
                    "message": check.message,
                }
                for check in self.checks
            ],
        }


@dataclass(frozen=True)
class SmokeCommand:
    """Explicit smoke-test command; ``{script}`` is replaced by a temp path."""

    argv: tuple[str, ...]
    timeout_seconds: int = _DEFAULT_SMOKE_TIMEOUT

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("smoke command cannot be empty")
        if any(not isinstance(arg, str) for arg in self.argv):
            raise TypeError("smoke command arguments must be strings")
        if not any("{script}" in arg for arg in self.argv):
            raise ValueError("smoke command must contain a {script} placeholder")
        if not 1 <= self.timeout_seconds <= _MAX_SMOKE_TIMEOUT:
            raise ValueError(
                f"smoke timeout must be between 1 and {_MAX_SMOKE_TIMEOUT} seconds"
            )


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


def _literal_value(node: ast.AST) -> tuple[bool, Any]:
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False, None


def _same_literal(actual: Any, expected: Any) -> bool:
    """Compare literals without treating ``True`` as the integer ``1``."""
    return type(actual) is type(expected) and actual == expected


def _contains_literal(tree: ast.Module, expected: Any) -> bool:
    for node in ast.walk(tree):
        ok, actual = _literal_value(node)
        if ok and _same_literal(actual, expected):
            return True
    return False


def _target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        ok, value = _literal_value(node.slice)
        if ok and isinstance(value, str):
            return value
    return None


def _has_named_literal(tree: ast.Module, name: str, expected: Any) -> bool:
    """Find a locked ``name=value`` in kwargs, assignments, or dict entries."""
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == name:
            ok, actual = _literal_value(node.value)
            if ok and _same_literal(actual, expected):
                return True
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value_node = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_target_name(target) == name for target in targets):
                ok, actual = _literal_value(value_node)
                if ok and _same_literal(actual, expected):
                    return True
        elif isinstance(node, ast.Dict):
            for key_node, value_node in zip(node.keys, node.values):
                if key_node is None:
                    continue
                key_ok, key = _literal_value(key_node)
                if key_ok and key == name:
                    value_ok, actual = _literal_value(value_node)
                    if value_ok and _same_literal(actual, expected):
                        return True
    return False


def _static_semantic_check(
    tree: ast.Module,
    *,
    expected_model_id: str | None,
    expected_output_dir: str | Path | None,
    locked_hyperparameters: Mapping[str, Any] | None,
) -> StageResult:
    missing: list[str] = []
    if expected_model_id is not None and not _contains_literal(tree, expected_model_id):
        missing.append(f"model_id={expected_model_id!r}")

    if expected_output_dir is not None:
        exact_output = str(expected_output_dir)
        if not _contains_literal(tree, exact_output):
            missing.append(f"output_dir={exact_output!r}")

    for name, expected in (locked_hyperparameters or {}).items():
        if not _has_named_literal(tree, name, expected):
            missing.append(f"locked {name}={expected!r}")

    configured = (
        expected_model_id is not None
        or expected_output_dir is not None
        or bool(locked_hyperparameters)
    )
    if not configured:
        return StageResult(
            ValidationStage.STATIC_SEMANTICS,
            CheckStatus.SKIPPED,
            "static semantic checks not configured",
        )
    if missing:
        return StageResult(
            ValidationStage.STATIC_SEMANTICS,
            CheckStatus.FAILED,
            "generated code does not reference required values: " + ", ".join(missing),
        )
    return StageResult(
        ValidationStage.STATIC_SEMANTICS,
        CheckStatus.PASSED,
        "required model, output, and locked hyperparameter values are present",
    )


def _run_smoke(code: str, smoke: SmokeCommand | None, work_dir: Path | None) -> StageResult:
    if smoke is None:
        return StageResult(
            ValidationStage.SMOKE,
            CheckStatus.SKIPPED,
            "smoke command not configured",
        )

    directory = (work_dir or _DEFAULT_OUTPUT_ROOT).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix=".validation-",
            dir=directory,
            delete=False,
        ) as candidate:
            candidate.write(code)
            temp_path = Path(candidate.name)

        argv = [arg.replace("{script}", str(temp_path)) for arg in smoke.argv]
        require_host_execution("generated-script smoke validation")
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=smoke.timeout_seconds,
                env=child_env(include_hf=False),
                cwd=str(directory),
            )
        except subprocess.TimeoutExpired:
            return StageResult(
                ValidationStage.SMOKE,
                CheckStatus.FAILED,
                f"timed out after {smoke.timeout_seconds}s",
            )
        except OSError as exc:
            return StageResult(
                ValidationStage.SMOKE,
                CheckStatus.FAILED,
                f"could not start smoke command: {exc}",
            )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-10:]
            return StageResult(
                ValidationStage.SMOKE,
                CheckStatus.FAILED,
                "\n".join(tail) or f"smoke command exited {result.returncode}",
            )
        return StageResult(
            ValidationStage.SMOKE,
            CheckStatus.PASSED,
            "smoke command completed successfully",
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def validate_staged(
    code: str,
    method_id: str,
    *,
    expected_model_id: str | None = None,
    expected_output_dir: str | Path | None = None,
    locked_hyperparameters: Mapping[str, Any] | None = None,
    smoke_command: SmokeCommand | None = None,
    smoke_work_dir: Path | None = None,
) -> ValidationResult:
    """Run staged validation and stop immediately at the first failed stage."""
    checks: list[StageResult] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        checks.append(
            StageResult(
                ValidationStage.PARSE,
                CheckStatus.FAILED,
                f"{e.msg} at line {e.lineno}",
            )
        )
        return ValidationResult(tuple(checks))
    checks.append(StageResult(ValidationStage.PARSE, CheckStatus.PASSED, "valid Python AST"))

    py = venv_python(method_id)
    if not py.exists():
        # Venv not built yet — trust ast.parse. The Adapt agent is expected to
        # build the venv via install_method_venv before the script runs; if it
        # didn't, launch() will surface the error.
        checks.append(
            StageResult(
                ValidationStage.DRY_IMPORT,
                CheckStatus.SKIPPED,
                "dry-import skipped (venv not yet built for this method)",
            )
        )
    else:
        modules = _collect_top_level_modules(tree)
        if not modules:
            checks.append(
                StageResult(
                    ValidationStage.DRY_IMPORT,
                    CheckStatus.PASSED,
                    "no top-level imports",
                )
            )
        else:
            probe = ";".join(f"import {m}" for m in modules)
            require_host_execution("dry-import validation")
            try:
                r = subprocess.run(
                    [str(py), "-c", probe],
                    capture_output=True,
                    text=True,
                    timeout=_DRY_IMPORT_TIMEOUT,
                    # Imports run module top-level code; do not expose cloud secrets.
                    env=child_env(include_hf=False),
                )
            except subprocess.TimeoutExpired:
                checks.append(
                    StageResult(
                        ValidationStage.DRY_IMPORT,
                        CheckStatus.FAILED,
                        f"timed out after {_DRY_IMPORT_TIMEOUT}s",
                    )
                )
                return ValidationResult(tuple(checks))
            except OSError as exc:
                checks.append(
                    StageResult(
                        ValidationStage.DRY_IMPORT,
                        CheckStatus.FAILED,
                        f"could not start import probe: {exc}",
                    )
                )
                return ValidationResult(tuple(checks))
            if r.returncode != 0:
                tail = (r.stderr or r.stdout or "").strip().splitlines()[-10:]
                checks.append(
                    StageResult(
                        ValidationStage.DRY_IMPORT,
                        CheckStatus.FAILED,
                        "\n".join(tail) or f"import probe exited {r.returncode}",
                    )
                )
                return ValidationResult(tuple(checks))
            checks.append(
                StageResult(
                    ValidationStage.DRY_IMPORT,
                    CheckStatus.PASSED,
                    f"imported: {', '.join(modules)}",
                )
            )

    semantic = _static_semantic_check(
        tree,
        expected_model_id=expected_model_id,
        expected_output_dir=expected_output_dir,
        locked_hyperparameters=locked_hyperparameters,
    )
    checks.append(semantic)
    if not semantic.ok:
        return ValidationResult(tuple(checks))

    smoke = _run_smoke(code, smoke_command, smoke_work_dir)
    checks.append(smoke)
    return ValidationResult(tuple(checks))


def validate(code: str, method_id: str) -> tuple[bool, str, str]:
    """Backward-compatible tuple API; use ``validate_staged`` for full detail."""
    result = validate_staged(code, method_id)
    return result.ok, result.stage.value, result.message


class ValidationSession:
    """Per-run retry counter for the Adapt agent's write_script tool."""

    def __init__(
        self,
        method_id: str,
        max_attempts: int = _MAX_ATTEMPTS,
        allowed_root: Path | None = None,
        *,
        expected_model_id: str | None = None,
        expected_output_dir: str | Path | None = None,
        locked_hyperparameters: Mapping[str, Any] | None = None,
        smoke_command: SmokeCommand | Sequence[str] | None = None,
        smoke_timeout_seconds: int = _DEFAULT_SMOKE_TIMEOUT,
    ) -> None:
        self.method_id = method_id
        self.attempts_left = max_attempts
        # Generated scripts may only be written under this directory.
        self.allowed_root = (allowed_root or _DEFAULT_OUTPUT_ROOT)
        self.expected_model_id = expected_model_id
        self.expected_output_dir = expected_output_dir
        self.locked_hyperparameters = dict(locked_hyperparameters or {})
        if smoke_command is None or isinstance(smoke_command, SmokeCommand):
            self.smoke_command = smoke_command
        else:
            if isinstance(smoke_command, str):
                raise TypeError("smoke_command must be a sequence of arguments, not a string")
            self.smoke_command = SmokeCommand(
                tuple(smoke_command), timeout_seconds=smoke_timeout_seconds
            )
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

        validation = validate_staged(
            code,
            self.method_id,
            expected_model_id=self.expected_model_id,
            expected_output_dir=self.expected_output_dir,
            locked_hyperparameters=self.locked_hyperparameters,
            smoke_command=self.smoke_command,
            smoke_work_dir=out.parent,
        )
        if validation.ok:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(code)
            self.validated_path = out
            return {
                "status": "ok",
                "stage": validation.stage.value,
                "message": validation.message,
                "validation": validation.as_dict(),
                "path": str(out),
                "attempts_left": self.attempts_left,
            }

        self.attempts_left -= 1
        if self.attempts_left > 0:
            return {
                "status": "error",
                "stage": validation.stage.value,
                "message": validation.message,
                "validation": validation.as_dict(),
                "attempts_left": self.attempts_left,
            }

        # Exhausted: fail closed. A known-invalid artifact must never become an
        # executable handoff merely because the retry budget was consumed.
        return {
            "status": "error-exhausted",
            "stage": validation.stage.value,
            "message": validation.message,
            "validation": validation.as_dict(),
            "attempts_left": 0,
        }


def make_write_script_tool(
    session: ValidationSession,
    *,
    expected_model_id: str | None = None,
    expected_output_dir: str | Path | None = None,
    locked_hyperparameters: Mapping[str, Any] | None = None,
):
    """Build a @tool bound to this session so retry state is preserved across calls."""

    # Optional overrides let callers add per-tool expectations while preserving
    # the historical ``make_write_script_tool(session)`` API.
    if expected_model_id is not None:
        session.expected_model_id = expected_model_id
    if expected_output_dir is not None:
        session.expected_output_dir = expected_output_dir
    if locked_hyperparameters is not None:
        session.locked_hyperparameters = dict(locked_hyperparameters)

    @tool
    def write_script(path: str, code: str) -> str:
        """Run syntax, import, configured semantic, and optional smoke validation,
        then write Python code to `path`. On failure, returns an error payload with
        `attempts_left` and per-stage results so you can revise and call again.

        After attempts are exhausted no file is written; the Adapt invocation fails.
        """
        return json.dumps(session.write(path, code), indent=2)

    return write_script
