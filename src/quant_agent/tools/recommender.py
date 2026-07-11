from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

import yaml
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..config import DATA_ROOT


# VRAM overhead assumed for activations, KV cache, CUDA workspace, etc.
# Tunable, but a conservative 1.4x on weight bytes approximates real-world headroom
# for short-context inference; long context needs more and KIVI helps.
_VRAM_OVERHEAD_FACTOR = 1.4


class Constraints(BaseModel):
    params_b: float = Field(..., description="Model parameter count in billions.")
    vram_gb: float = Field(..., description="Available GPU VRAM in gigabytes.")
    target_bits: int | None = Field(
        None, description="Preferred weight bit-width (2/3/4/8). Omit to let the recommender pick."
    )
    backend: Literal["vllm", "transformers", "tgi", "llama_cpp", "tensorrt_llm", "any"] = "any"
    priority: Literal["quality", "speed", "balanced"] = "balanced"
    have_calibration_data: bool = True
    allow_qat: bool = False
    need_activation_quant: bool = False
    need_kv_cache_quant: bool = False


@lru_cache(maxsize=1)
def load_catalog() -> list[dict]:
    with (DATA_ROOT / "methods.yaml").open() as f:
        return yaml.safe_load(f)


def _weight_bytes_gb(params_b: float, bits: int) -> float:
    return params_b * 1e9 * bits / 8 / 1e9


def _fits(params_b: float, vram_gb: float, bits: int) -> bool:
    return _weight_bytes_gb(params_b, bits) * _VRAM_OVERHEAD_FACTOR <= vram_gb


def _score(method: dict, c: Constraints, bits: int) -> float:
    priority_weights = {
        "quality": (0.6, 0.15, 0.25),
        "speed": (0.2, 0.55, 0.25),
        "balanced": (0.4, 0.35, 0.25),
    }
    wq, ws, wm = priority_weights[c.priority]
    score = wq * method["quality"] + ws * method["speedup"] + wm * method["maturity"]

    # Small bonus for matching the requested bit width exactly (vs just being compatible).
    if c.target_bits and bits == c.target_bits:
        score += 0.3

    # Prefer calibration-free methods when the user has no calibration data.
    if not c.have_calibration_data and not method["needs_calibration"]:
        score += 0.5

    return round(score, 3)


def _filter(methods: list[dict], c: Constraints) -> list[tuple[dict, int]]:
    """Return (method, chosen_bits) pairs that satisfy hard constraints."""
    out = []
    for m in methods:
        if c.backend != "any" and c.backend not in (m.get("inference_backends") or []):
            continue
        if m["qat"] and not c.allow_qat:
            continue
        if not c.have_calibration_data and m["needs_calibration"]:
            continue
        if c.need_activation_quant and "activations" not in (m.get("quantizes") or []):
            continue
        if c.need_kv_cache_quant and "kv_cache" not in (m.get("quantizes") or []):
            continue

        supported_bits = m.get("bits") or []
        if c.target_bits is not None:
            if c.target_bits not in supported_bits:
                continue
            bits_candidates = [c.target_bits]
        else:
            # When unspecified, prefer 4-bit (the common LLM sweet spot), then
            # fall back to other widths from most to least conservative.
            preference = [4, 8, 5, 6, 3, 2, 1]
            bits_candidates = [b for b in preference if b in supported_bits]

        for bits in bits_candidates:
            if _fits(c.params_b, c.vram_gb, bits):
                out.append((m, bits))
                break
    return out


def rank(constraints: Constraints, top_k: int = 5) -> list[dict]:
    catalog = load_catalog()
    candidates = _filter(catalog, constraints)
    scored = [
        {
            "id": m["id"],
            "name": m["name"],
            "bits": bits,
            "score": _score(m, constraints, bits),
            "weight_gb": round(_weight_bytes_gb(constraints.params_b, bits), 2),
            "backends": m.get("inference_backends"),
            "needs_calibration": m["needs_calibration"],
            "template": m.get("template"),
            "notes": m.get("notes"),
        }
        for (m, bits) in candidates
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


@tool
def recommend_quantization(
    params_b: float,
    vram_gb: float,
    target_bits: int | None = None,
    backend: str = "any",
    priority: str = "balanced",
    have_calibration_data: bool = True,
    allow_qat: bool = False,
    need_activation_quant: bool = False,
    need_kv_cache_quant: bool = False,
) -> str:
    """Deterministically rank quantization methods against user constraints.

    Call this AFTER hf_model_info has given you params_b. The result is authoritative
    for method selection — the LLM should explain and cite, not override this ranking
    unless the user asks for a specific method.

    Args:
        params_b: Model parameter count in billions (from hf_model_info).
        vram_gb: User's available GPU VRAM in GB.
        target_bits: Optional preferred bit width (2/3/4/8).
        backend: One of 'vllm', 'transformers', 'tgi', 'llama_cpp', 'tensorrt_llm', 'any'.
        priority: 'quality' | 'speed' | 'balanced'.
        have_calibration_data: Set False if user has no calibration set (excludes GPTQ/AWQ/etc.).
        allow_qat: Allow training-based methods like LLM-QAT.
        need_activation_quant: Require methods that quantize activations.
        need_kv_cache_quant: Require methods that quantize the KV cache.
    """
    c = Constraints(
        params_b=params_b,
        vram_gb=vram_gb,
        target_bits=target_bits,
        backend=backend,  # type: ignore[arg-type]
        priority=priority,  # type: ignore[arg-type]
        have_calibration_data=have_calibration_data,
        allow_qat=allow_qat,
        need_activation_quant=need_activation_quant,
        need_kv_cache_quant=need_kv_cache_quant,
    )
    ranked = rank(c)
    if not ranked:
        return json.dumps(
            {
                "error": "No methods satisfy the constraints. Try loosening backend, "
                "target_bits, or have_calibration_data."
            }
        )
    return json.dumps({"ranking": ranked}, indent=2)
