#!/usr/bin/env python3
"""Validated contract for method-specific quantized-model inference adapters.

Research repositories persist many mutually incompatible artifact formats. A port
overlay may carry a small adapter that reopens the saved quantized artifact.
This module validates that adapter statically and never executes it.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ADAPTER_API_VERSION = 1
ADAPTER_FUNCTION = "load_model_and_tokenizer"
MAX_ADAPTER_BYTES = 100_000


class InvalidInferenceAdapter(ValueError):
    """Raised when generated adapter source does not satisfy the static contract."""


def _literal_assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                return None
    return None


def validate_adapter_source(source: str) -> ast.Module:
    """Parse and validate the stable adapter API without executing generated code."""
    if not isinstance(source, str) or not source.strip():
        raise InvalidInferenceAdapter("inference adapter must be non-empty Python source")
    if len(source.encode("utf-8")) > MAX_ADAPTER_BYTES:
        raise InvalidInferenceAdapter(
            f"inference adapter exceeds {MAX_ADAPTER_BYTES} bytes"
        )
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise InvalidInferenceAdapter(
            f"inference adapter has invalid Python syntax: {exc.msg}"
        ) from exc

    api_version = _literal_assignment(tree, "QUANT_AGENT_ADAPTER_API")
    if type(api_version) is not int or api_version != ADAPTER_API_VERSION:
        raise InvalidInferenceAdapter(
            f"inference adapter must declare QUANT_AGENT_ADAPTER_API = {ADAPTER_API_VERSION}"
        )

    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == ADAPTER_FUNCTION
    ]
    if len(functions) != 1 or isinstance(functions[0], ast.AsyncFunctionDef):
        raise InvalidInferenceAdapter(
            f"inference adapter must define one synchronous {ADAPTER_FUNCTION} function"
        )
    function = functions[0]
    parameter_names = {
        arg.arg
        for arg in (
            list(function.args.posonlyargs)
            + list(function.args.args)
            + list(function.args.kwonlyargs)
        )
    }
    required = {"model_path", "model_id", "dtype", "device", "trust_remote_code"}
    missing = sorted(required - parameter_names)
    if missing:
        raise InvalidInferenceAdapter(
            f"{ADAPTER_FUNCTION} is missing parameters: {', '.join(missing)}"
        )

    # Generated adapters should be declarative at import time. Heavy model loading,
    # downloads, subprocesses, and CUDA work belong inside the loader function.
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        if isinstance(node, ast.Expr) and index == 0 and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError) as exc:
                raise InvalidInferenceAdapter(
                    "inference adapter top-level assignments must be literals"
                ) from exc
            continue
        raise InvalidInferenceAdapter(
            "inference adapter may contain only imports, literal constants, and functions "
            "at module scope"
        )
    return tree


def validate_adapter_file(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise InvalidInferenceAdapter(f"inference adapter is missing: {resolved}")
    validate_adapter_source(resolved.read_text())
    return resolved


def _is_name(node: ast.AST, ident: str) -> bool:
    return isinstance(node, ast.Name) and node.id == ident


def _call_uses_name(call: ast.Call, ident: str) -> bool:
    for arg in call.args:
        if _is_name(arg, ident):
            return True
    for kw in call.keywords:
        if _is_name(kw.value, ident):
            return True
    return False


def _func_is_from_pretrained(func: ast.AST) -> bool:
    if isinstance(func, ast.Name):
        return func.id == "from_pretrained"
    return isinstance(func, ast.Attribute) and func.attr == "from_pretrained"


def _name_or_attr_startswith(node: ast.AST, prefix: str) -> bool:
    if isinstance(node, ast.Name):
        return node.id.startswith(prefix)
    if isinstance(node, ast.Attribute):
        if node.attr.startswith(prefix):
            return True
        return _name_or_attr_startswith(node.value, prefix)
    return False


def _func_is_automodel(func: ast.AST) -> bool:
    return _name_or_attr_startswith(func, "AutoModel")


_LOAD_FUNC_NAMES = frozenset({"from_pretrained", "snapshot_download", "hf_hub_download"})


def _func_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _load_target(call: ast.Call) -> ast.AST | None:
    if any(kw.arg is None for kw in call.keywords):
        return None
    for kw in call.keywords:
        if kw.arg in {"pretrained_model_name_or_path", "repo_id"}:
            return kw.value
    if call.args:
        return call.args[0]
    return None


def adapter_loads_hub_id(source: str) -> bool:
    """True when a load target is anything other than the Name ``model_path``.

    Catches ``from_pretrained(model_id)``, string Hub ids, aliases, ``**kwargs``,
    and ``eval``/``exec``. Adapters must reopen the saved artifact.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _func_name(node.func)
            if name in {"eval", "exec"}:
                return True
            is_load = name in _LOAD_FUNC_NAMES or (
                name is not None and name.startswith("AutoModel")
            )
            if not is_load:
                continue
            target = _load_target(node)
            if target is None or not _is_name(target, "model_path"):
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an inference adapter without executing it")
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--check-hub-id",
        action="store_true",
        help="Fail if the adapter loads from_pretrained(model_id) / AutoModel*(model_id)",
    )
    args = parser.parse_args()
    source = args.path.read_text()
    validate_adapter_source(source)
    loads_hub = adapter_loads_hub_id(source)
    payload = {"ok": True, "loads_hub_id": loads_hub, "path": str(args.path.resolve())}
    if args.check_hub_id and loads_hub:
        payload["ok"] = False
        payload["message"] = "inference adapter loads Hub model_id instead of the saved artifact"
        print(json.dumps(payload, indent=2))
        return 1
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
