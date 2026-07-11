"""Adapt-only entry point: skip Research/selection and drive Adapt directly.

Used for isolated testing of the Adapt agent against a known (method, model)
pair — e.g. `flatquant` + `meta-llama/Llama-2-7b-hf`. Loads the method's
repo_url + bit-width from the packaged catalog, constructs a MethodCandidate
with placeholder scoring (Adapt doesn't read these fields), and invokes
adapt_agent.run.
"""
from __future__ import annotations

import typer
import yaml

from . import adapt_agent
from .config import load_settings
from .config import host_execution_policy
from .schemas import MethodCandidate


def _load_method_entry(method_id: str) -> dict:
    s = load_settings()
    with s.seed_path.open() as f:
        raw = yaml.safe_load(f) or []
    for entry in raw:
        if entry.get("id") == method_id:
            return entry
    raise ValueError(
        f"Method id '{method_id}' not in {s.seed_path}. "
        f"Available: {sorted(e.get('id', '?') for e in raw)}"
    )


def _build_candidate(entry: dict, bits: int | None) -> MethodCandidate:
    repos = entry.get("repos") or []
    if not repos:
        raise ValueError(f"Method '{entry.get('id')}' has no repos configured.")
    available_bits = entry.get("bits") or []
    chosen_bits = bits if bits is not None else (available_bits[0] if available_bits else 4)
    if available_bits and chosen_bits not in available_bits:
        raise ValueError(
            f"Method '{entry.get('id')}' does not support {chosen_bits}-bit. "
            f"Supported: {available_bits}"
        )
    return MethodCandidate(
        id=entry["id"],
        name=entry["name"],
        repo_url=repos[0],
        bits=chosen_bits,
        # Adapt doesn't read these; supply plausible defaults for schema validity.
        est_vram_gb=0.0,
        quality_score=entry.get("quality", 3),
        speed_score=entry.get("speedup", 3),
        needs_calibration=bool(entry.get("needs_calibration", False)),
        summary=entry.get("notes", ""),
    )


def run(
    method_id: str,
    model_id: str,
    bits: int | None = None,
    trust_remote_code: bool = False,
    allow_unsafe_host_execution: bool = False,
) -> tuple[str, str]:
    """Invoke the Adapt agent directly. Returns (script_path, script_code)."""
    entry = _load_method_entry(method_id)
    candidate = _build_candidate(entry, bits)

    typer.echo(
        f"Adapt-only: method={candidate.id} ({candidate.bits}-bit), "
        f"repo={candidate.repo_url}, model={model_id}",
        err=True,
    )
    with host_execution_policy(allow_unsafe_host_execution):
        return adapt_agent.run(
            model_id=model_id, method=candidate, trust_remote_code=trust_remote_code
        )
