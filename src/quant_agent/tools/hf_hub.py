from __future__ import annotations

import json

from huggingface_hub import HfApi, hf_hub_download
from langchain_core.tools import tool

from ..config import load_settings


def _safe_params_b(info) -> float | None:
    # siblings[i].size is sometimes populated, safetensors index includes total_size
    try:
        safetensors = getattr(info, "safetensors", None)
        if safetensors and getattr(safetensors, "total", None):
            return round(safetensors.total / 1e9, 2)
    except Exception:
        pass
    return None


@tool
def hf_model_info(model_id: str) -> str:
    """Look up a HuggingFace model's architecture, parameter count, and native dtype.

    Use this FIRST whenever the user names a model — the recommender needs the parameter
    count in billions and the architecture family to filter methods.
    Returns a JSON string with keys: model_id, architectures, params_b, torch_dtype,
    hidden_size, num_hidden_layers, vocab_size, tags.
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

    out = {
        "model_id": model_id,
        "architectures": cfg.get("architectures"),
        "params_b": _safe_params_b(info),
        "torch_dtype": cfg.get("torch_dtype"),
        "hidden_size": cfg.get("hidden_size"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "vocab_size": cfg.get("vocab_size"),
        "tags": list(getattr(info, "tags", []) or [])[:15],
    }
    return json.dumps(out, indent=2)
