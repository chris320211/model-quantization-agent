"""Target-model architecture tools for the Adapt agent.

``hf_model_info`` is a lossy summary tuned for method *filtering* (family + size).
For *porting* a quantizer the agent needs two more things:

  - ``fetch_model_config``: the FULL config.json (num_key_value_heads for GQA,
    intermediate_size, tie_word_embeddings, MoE/rope fields, quantization_config)
    plus a ``trust_remote_code_required`` flag derived from ``auto_map``.
  - ``inspect_architecture_core``: the EXACT runtime module tree. It instantiates
    the model on the ``meta`` device (no weight download) inside the method's venv
    and dumps ``named_modules()`` — the ground truth for which Linear layers the
    quantizer will target. For ``trust_remote_code`` models this executes the
    repo's custom code, so it is gated behind an explicit trust ack.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess

from huggingface_hub import hf_hub_download
from langchain_core.tools import tool

from ..config import child_env, load_settings
from ..executor import venv_python

log = logging.getLogger(__name__)

_INTROSPECT_TIMEOUT = 300

# Runs inside the method venv. Meta-device instantiation: builds the module tree
# from config alone, no weights downloaded. Prints one sentinel line the parent greps.
_INTROSPECT_SCRIPT = r"""
import json, os, sys
model_id = os.environ["INTROSPECT_MODEL_ID"]
trust = os.environ.get("INTROSPECT_TRUST", "0") == "1"
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust)

rows = []
for name, mod in model.named_modules():
    cls = mod.__class__.__name__
    if isinstance(mod, nn.Linear):
        rows.append({"name": name, "cls": cls, "in": mod.in_features, "out": mod.out_features})
    elif isinstance(mod, nn.Embedding):
        rows.append({"name": name, "cls": cls, "num": mod.num_embeddings, "dim": mod.embedding_dim})
print("INTROSPECT_RESULT=" + json.dumps(rows))
"""

_RESULT_RE = re.compile(r"INTROSPECT_RESULT=(\[.*\])")


def _load_config(model_id: str) -> dict:
    s = load_settings()
    path = hf_hub_download(repo_id=model_id, filename="config.json", token=s.hf_token)
    with open(path) as f:
        return json.load(f)


def fetch_model_config_dict(model_id: str) -> dict:
    """Full config.json + trust flag. Returns {'error': ...} on failure (never raises)."""
    try:
        cfg = _load_config(model_id)
    except Exception as e:  # noqa: BLE001
        return {"error": f"config.json fetch failed: {e}", "model_id": model_id}
    auto_map = cfg.get("auto_map")
    return {
        "model_id": model_id,
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "trust_remote_code_required": bool(auto_map),
        "auto_map": auto_map,
        "config": cfg,
    }


@tool
def fetch_model_config(model_id: str) -> str:
    """Fetch the target model's FULL config.json (not the hf_model_info summary).

    Exposes the fields a quantizer needs and hf_model_info drops: num_key_value_heads
    (GQA), intermediate_size, tie_word_embeddings, head_dim, MoE/rope fields, and any
    existing quantization_config. `trust_remote_code_required` is true when the model
    ships custom modeling code (config has `auto_map`).
    """
    return json.dumps(fetch_model_config_dict(model_id), indent=2)


def _collapse(rows: list[dict]) -> list[dict]:
    """Collapse repeated indexed blocks (model.layers.0.., .1..) into one {i} pattern."""
    seen: dict[str, dict] = {}
    order: list[str] = []
    for r in rows:
        key = re.sub(r"\.\d+\.", ".{i}.", r["name"])
        key = re.sub(r"\.\d+$", ".{i}", key)
        if key not in seen:
            seen[key] = {k: r.get(k) for k in ("cls", "in", "out", "num", "dim") if r.get(k) is not None}
            seen[key]["pattern"] = key
            seen[key]["count"] = 0
            order.append(key)
        seen[key]["count"] += 1
    return [seen[k] for k in order]


def _config_only(reason: str, cfg_info: dict, **extra) -> str:
    return json.dumps(
        {
            "status": "config_only",
            "reason": reason,
            "model_id": cfg_info.get("model_id"),
            "architectures": cfg_info.get("architectures"),
            "config": cfg_info.get("config"),
            **extra,
        },
        indent=2,
    )


def inspect_architecture_core(
    model_id: str, method_id: str, *, trust_remote_code: bool = False
) -> str:
    """Meta-device module-tree introspection inside the method venv. JSON string.

    Falls back to a config-only summary when: the model needs trust_remote_code and
    it wasn't granted, the venv isn't built yet, or the meta-device load fails.
    """
    cfg_info = fetch_model_config_dict(model_id)
    if cfg_info.get("error"):
        return json.dumps({"status": "error", **cfg_info}, indent=2)

    if cfg_info.get("trust_remote_code_required") and not trust_remote_code:
        return _config_only(
            "model requires trust_remote_code (auto_map present) and trust was not "
            "granted; skipping meta-device instantiation of untrusted code.",
            cfg_info,
        )

    py = venv_python(method_id)
    if not py.exists():
        return _config_only(
            f"method venv not built yet ({py}); run install_method_venv first, "
            "then call inspect_model_architecture again for the full module tree.",
            cfg_info,
        )

    env = child_env(
        {
            "INTROSPECT_MODEL_ID": model_id,
            "INTROSPECT_TRUST": "1" if trust_remote_code else "0",
        }
    )
    try:
        r = subprocess.run(
            [str(py), "-c", _INTROSPECT_SCRIPT],
            capture_output=True,
            text=True,
            timeout=_INTROSPECT_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return _config_only(f"introspection timed out after {_INTROSPECT_TIMEOUT}s", cfg_info)

    m = _RESULT_RE.search(r.stdout or "")
    if not m:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-15:]
        return _config_only("meta-device load failed; see stderr_tail", cfg_info, stderr_tail=tail)

    rows = json.loads(m.group(1))
    return json.dumps(
        {
            "status": "ok",
            "model_id": model_id,
            "architectures": cfg_info.get("architectures"),
            "num_leaf_linear_embedding": len(rows),
            "modules": _collapse(rows),
        },
        indent=2,
    )
