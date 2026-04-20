from __future__ import annotations

import json

from huggingface_hub import HfApi, hf_hub_download
from langchain_core.tools import tool

from ..config import load_settings


_DTYPE_BYTES: dict[str, int] = {
    "float32": 4, "fp32": 4, "torch.float32": 4,
    "float16": 2, "fp16": 2, "torch.float16": 2,
    "bfloat16": 2, "bf16": 2, "torch.bfloat16": 2,
    "int8": 1, "uint8": 1,
    "float8": 1, "fp8": 1,
}


def _bytes_per_param(torch_dtype: str | None) -> int:
    if not torch_dtype:
        return 2  # most modern LLM checkpoints ship fp16/bf16
    return _DTYPE_BYTES.get(torch_dtype.lower(), 2)


def _from_safetensors_total(info) -> float | None:
    try:
        safetensors = getattr(info, "safetensors", None)
        if safetensors and getattr(safetensors, "total", None):
            return round(safetensors.total / 1e9, 2)
    except Exception:
        return None
    return None


def _from_siblings(info, torch_dtype: str | None) -> float | None:
    """Sum *.safetensors sibling sizes and divide by bytes-per-param."""
    try:
        siblings = getattr(info, "siblings", None) or []
        total_bytes = 0
        for s in siblings:
            name = getattr(s, "rfilename", "") or ""
            size = getattr(s, "size", None)
            if name.endswith(".safetensors") and isinstance(size, int):
                total_bytes += size
        if total_bytes <= 0:
            return None
        return round(total_bytes / _bytes_per_param(torch_dtype) / 1e9, 2)
    except Exception:
        return None


def _from_config(cfg: dict) -> float | None:
    """Transformer param-count approximation: ~12 * L * H^2 (ignores embedding + ff ratio).

    Rough but useful when no weight files are accessible (HF Hub returns None for
    siblings on some repos). Accuracy typically within 10-20% for standard decoders.
    """
    try:
        layers = cfg.get("num_hidden_layers")
        hidden = cfg.get("hidden_size")
        if not layers or not hidden:
            return None
        approx = 12 * int(layers) * int(hidden) ** 2
        return round(approx / 1e9, 2)
    except Exception:
        return None


def _resolve_params_b(info, cfg: dict) -> tuple[float | None, str | None]:
    """Return (params_b, source) where source is one of 'safetensors_total',
    'siblings_sum', 'config_approx', or None."""
    exact = _from_safetensors_total(info)
    if exact is not None:
        return exact, "safetensors_total"

    torch_dtype = cfg.get("torch_dtype") if cfg else None
    approx = _from_siblings(info, torch_dtype)
    if approx is not None:
        return approx, "siblings_sum"

    cfg_approx = _from_config(cfg or {})
    if cfg_approx is not None:
        return cfg_approx, "config_approx"

    return None, None


@tool
def hf_model_info(model_id: str) -> str:
    """Look up a HuggingFace model's architecture, parameter count, and native dtype.

    Use this FIRST whenever the user names a model — the recommender needs the parameter
    count in billions and the architecture family to filter methods.
    Returns a JSON string with keys: model_id, architectures, params_b, params_b_source,
    torch_dtype, hidden_size, num_hidden_layers, vocab_size, tags.
    """
    s = load_settings()
    api = HfApi(token=s.hf_token)
    try:
        info = api.model_info(model_id)
    except Exception as e:
        return json.dumps({"error": f"HF lookup failed: {e}", "model_id": model_id})

    cfg: dict = {}
    try:
        path = hf_hub_download(
            repo_id=model_id, filename="config.json", token=s.hf_token
        )
        with open(path) as f:
            cfg = json.load(f)
    except Exception:
        pass

    params_b, source = _resolve_params_b(info, cfg)

    out = {
        "model_id": model_id,
        "architectures": cfg.get("architectures"),
        "params_b": params_b,
        "params_b_source": source,
        "torch_dtype": cfg.get("torch_dtype"),
        "hidden_size": cfg.get("hidden_size"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "vocab_size": cfg.get("vocab_size"),
        "tags": list(getattr(info, "tags", []) or [])[:15],
    }
    return json.dumps(out, indent=2)
