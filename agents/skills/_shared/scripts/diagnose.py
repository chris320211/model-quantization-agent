#!/usr/bin/env python3
"""Diagnose a completed quant-run from logs, script, checkpoint, and benchmark.

Classifies known failure modes and recommends a bounded fix. Does not launch GPU jobs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jobs as jobs_mod
import paths
from request import RETRY_GPU_JOBS_MAX, load_request, write_request
from verify import resolve_workspace_path

PPL_EXPLODE_RATIO = 3.0
QUALITY_OK_RATIO = 1.5
VRAM_EPS_GB = 0.05
_OVERLAY_HEADER_RE = re.compile(r"^# QUANT_AGENT_OVERLAY_DIR=(.+)$", re.MULTILINE)
TPS_EPS_RATIO = 0.05
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt"}

ISSUE_UNWRAP_NO_LN_FUSE = "unwrap_without_ln_fuse"
ISSUE_DENSE_FP16 = "dense_fp16_checkpoint"
ISSUE_PPL_EXPLODED = "ppl_ratio_exploded"
ISSUE_VRAM_UNCHANGED = "vram_unchanged"
ISSUE_THROUGHPUT_UNCHANGED = "throughput_unchanged"
ISSUE_FAKEQUANT_NO_PACK = "fakequant_not_packed"
ISSUE_KERNEL_DTYPE = "packed_kernel_scale_dtype"
ISSUE_PACKED_LOADER = "packed_loader_failed"
ISSUE_EVAL_FLAGS = "eval_runtime_flags_missing"
ISSUE_PACKED_QUALITY_GAP = "packed_quality_gap"
ISSUE_PREFILL_KERNEL = "prefill_kernel_missing"

CODE_TRANSFORM_DROPPED = "transform_or_runtime_dropped_on_save"
CODE_FAKEQUANT_DENSE = "fakequant_saved_as_dense"
CODE_PPL_EXPLODED = "ppl_exploded"
CODE_KERNEL_DTYPE = "cuda_kernel_dtype_mismatch"
CODE_PACKED_LOADER = "packed_loader_failed"
CODE_EVAL_FLAGS = "eval_runtime_flags_missing"
CODE_PACKED_QUALITY_GAP = "packed_quality_gap"

GENERIC_CODES = {
    ISSUE_UNWRAP_NO_LN_FUSE: CODE_TRANSFORM_DROPPED,
    ISSUE_DENSE_FP16: CODE_FAKEQUANT_DENSE,
    ISSUE_FAKEQUANT_NO_PACK: CODE_FAKEQUANT_DENSE,
    ISSUE_PPL_EXPLODED: CODE_PPL_EXPLODED,
    ISSUE_VRAM_UNCHANGED: ISSUE_VRAM_UNCHANGED,
    ISSUE_THROUGHPUT_UNCHANGED: ISSUE_THROUGHPUT_UNCHANGED,
    ISSUE_KERNEL_DTYPE: CODE_KERNEL_DTYPE,
    ISSUE_PACKED_LOADER: CODE_PACKED_LOADER,
    ISSUE_EVAL_FLAGS: CODE_EVAL_FLAGS,
    ISSUE_PACKED_QUALITY_GAP: CODE_PACKED_QUALITY_GAP,
    ISSUE_PREFILL_KERNEL: ISSUE_PREFILL_KERNEL,
}

ACTION_RETRY_RANKED = "retry_ranked_overlay"
ACTION_AUTHOR_FIX = "author_fix"
ACTION_KERNEL = "kernel"
ACTION_NONE = "none"

# Packed-path GPU jobs (dtype/eval/loader fixes, prefill follow-up) may run
# past retry_gpu_jobs_max so a method that needs pack-then-prefill does not
# depend on a human raising the budget. Same caps for every method × model × GPU.
PACKED_PATH_OVERAGE = 2
_SECRET_LINE_RE = re.compile(
    r"token|secret|password|api[_-]?key|hf_token|authorization",
    re.I,
)
_EXCERPT_MAX_CHARS = 4000
KERNEL_ATTEMPT_CAP = 2
_ARCH_TOKENS = (
    "qkv_proj",
    "Phi3ForCausalLM",
    "gate_up_proj",
    "Qwen2Moe",
    "Mixtral",
)


def error_excerpt(
    *,
    stderr: str = "",
    stdout: str = "",
    verification_error: str | None = None,
    max_chars: int = _EXCERPT_MAX_CHARS,
) -> str:
    """Last traceback / verify error for diagnose_fix. No secret-like lines."""
    chunks: list[str] = []
    if verification_error and str(verification_error).strip():
        chunks.append(str(verification_error).strip())
    blob = "\n".join(part for part in (stderr, stdout) if part)
    if blob:
        marker = "Traceback (most recent call last):"
        idx = blob.rfind(marker)
        if idx >= 0:
            chunks.append(blob[idx:].strip())
        else:
            chunks.append("\n".join(blob.splitlines()[-40:]).strip())
    text = "\n---\n".join(part for part in chunks if part)
    cleaned = [
        line for line in text.splitlines() if not _SECRET_LINE_RE.search(line)
    ]
    text = "\n".join(cleaned).strip()
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def error_fix_notes(excerpt: str) -> list[str]:
    """Concrete fix hints inferred from the last process/verify traceback."""
    text = (excerpt or "").lower()
    if not text:
        return []
    notes: list[str] = []
    if (
        "flash_attn" in text
        or "flashattention2" in text
        or "flash attention 2" in text
    ):
        notes.append(
            "Last process/verify error is FlashAttention2 / missing flash_attn. "
            "Do not keep attn_implementation='flash_attention_2'. Use SDPA or eager."
        )
    if "expected all tensors to be on the same device" in text or (
        "cpu and cuda" in text and "device" in text
    ):
        notes.append(
            "Last process error is a CPU vs CUDA device mismatch. Move scales, "
            "RoPE/inv_freq, and pack inputs onto the weight device before "
            "matmul/pack."
        )
    if "sequence length is longer than the specified maximum" in text:
        notes.append(
            "Calibration concatenated too many tokens. Slice the corpus into "
            "model max_position_embeddings (or 2048) windows; do not tokenize "
            "the whole set as one sequence."
        )
    if "no module named" in text and "_cuda" in text:
        notes.append(
            "Last process error is a missing CUDA extension import. Import pack/"
            "quant helpers without pulling the FA2-only modeling path, or skip "
            "that import at pack time."
        )
    return notes


def _prior_issue_codes(request: dict) -> list[str]:
    raw = request.get("last_diagnose_path")
    if not raw:
        return []
    path = Path(str(raw))
    if not path.is_absolute():
        path = paths.REPO_ROOT / path
    prev = _read_json(path)
    if not prev:
        return []
    return [str(item) for item in (prev.get("issue_codes") or [])]


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else None


def analyze_script(code: str) -> dict:
    """Static signals from the launched quantize script."""
    tree_ok = True
    try:
        ast.parse(code)
    except SyntaxError:
        tree_ok = False
    unwrap = (
        "_unwrap_linears" in code
        or "child.linear" in code
        or "setattr(module, name, child.linear)" in code
        or (".linear.weight" in code and "copy_quantized" in code)
        or "_unwrap_linear" in code
        or "_copy_quantized_linears" in code
        or "export_llama_named" in code
    )
    copies_inner_linear = "linear.weight" in code and (
        "save_pretrained" in code or "_save_hf" in code or "vanilla" in code
    )
    official_reparam = "reparameterize_model" in code
    ln_fuse = "reparameterize_ln" in code or (
        "input_layernorm" in code and "diag_scale" in code
    )
    packed_int4 = (
        "pack_i4" in code
        or "packed_int4" in code
        or "Linear4bit" in code
        or "quantized_save" in code
    )
    drops_runtime = bool(
        unwrap
        or copies_inner_linear
        or "vanilla.save_pretrained" in code
        or "export_llama_named" in code
        or "_copy_quantized_linears" in code
    )
    keeps_runtime = (not drops_runtime) and (
        "flat_parameters.pth" in code
        or "load_flat_parameters" in code
        or "save_flat_parameters" in code
        or "flatquant_runtime" in code
        or "runtime.pt" in code
        or "inference_adapter" in code
        or "QUANT_AGENT_ADAPTER_API" in code
    )
    return {
        "parses": tree_ok,
        "unwraps_to_nn_linear": unwrap or copies_inner_linear,
        "drops_runtime": drops_runtime,
        "keeps_runtime": keeps_runtime,
        "official_reparameterize_model": official_reparam,
        "fuses_transform_into_layernorm": official_reparam or ln_fuse,
        "saves_packed_int4": packed_int4,
    }


def _tensor_meta(output_dir: Path) -> dict:
    """Describe saved weights without loading the model onto GPU."""
    files = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES
    ]
    dtypes: set[str] = set()
    n_tensors = 0
    nbytes = 0
    packed_cfg = False
    cfg = output_dir / "quantization_config.json"
    if cfg.is_file():
        try:
            payload = json.loads(cfg.read_text())
            fmt = str(payload.get("format") or "")
            packed_cfg = "int4" in fmt.lower() or "packed" in fmt.lower()
        except (OSError, json.JSONDecodeError, TypeError):
            packed_cfg = False
    try:
        from safetensors import safe_open
    except ImportError:
        safe_open = None
    for path in files:
        nbytes += path.stat().st_size
        if path.suffix == ".safetensors" and safe_open is not None:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    n_tensors += 1
                    tensor = handle.get_tensor(key)
                    dtypes.add(str(tensor.dtype).replace("torch.", ""))
    packed_file = any(
        "packed" in path.name.lower() or "int4" in path.name.lower() for path in files
    )
    packed_int4 = packed_cfg or packed_file
    return {
        "weight_files": len(files),
        "n_tensors": n_tensors,
        "dtypes": sorted(dtypes),
        "bytes": nbytes,
        "packed_int4": packed_int4,
        "dense_float_only": (not packed_int4)
        and bool(dtypes)
        and dtypes.issubset({"float16", "bfloat16", "float32"}),
    }


def _resolve_overlay_dir(raw: str) -> Path:
    overlay_path = Path(str(raw)).expanduser()
    if not overlay_path.is_absolute():
        overlay_path = (paths.REPO_ROOT / overlay_path).resolve()
    else:
        overlay_path = overlay_path.resolve()
    return overlay_path


def _overlay_keys(path: Path) -> set[str]:
    resolved = path.resolve() if path.exists() else path
    keys = {str(resolved).rstrip("/")}
    name = resolved.name
    parent = resolved.parent.name
    if name and name != "overlay":
        keys.add(name)
        if parent:
            keys.add(f"{parent}/{name}")
    return keys


def _patch_sha256(overlay_path: Path) -> str | None:
    patch = overlay_path / "overlay.patch"
    if not patch.is_file():
        return None
    return hashlib.sha256(patch.read_bytes()).hexdigest()


def _header_overlay(code: str) -> str | None:
    match = _OVERLAY_HEADER_RE.search(code)
    if not match:
        return None
    raw = match.group(1).strip()
    return raw or None


def _authored_strategy_overlays(slug: str, strategy: str) -> list[str]:
    root = paths.OVERLAYS_ROOT / slug / strategy
    if not root.is_dir():
        return []
    found: list[str] = []
    for child in sorted(root.iterdir()):
        if child.name.startswith(".") or not child.is_dir():
            continue
        if (child / "overlay.patch").is_file():
            found.append(str(child.resolve()))
    return found


def _authored_fix_overlays(slug: str) -> list[str]:
    return _authored_strategy_overlays(slug, "diagnose_fix")


def _kernel_dtype_mismatch(verification_error: str | None) -> bool:
    if not verification_error:
        return False
    text = verification_error.lower()
    if "float16" not in text and "fp16" not in text:
        return False
    return any(
        token in text
        for token in (
            "sym_dequant",
            "sym_quant",
            "scale_row",
            "scale_col",
            "weight_scales",
        )
    )


def _casts_kernel_fp16(text: str) -> bool:
    return (
        "cast_linear4bit_kernel_dtypes" in text
        or "KERNEL_SCALE_DTYPE" in text
        or ("weight_scales" in text and "torch.float16" in text)
    )


def _restores_eval_runtime(text: str) -> bool:
    return "restore_packed_eval_runtime" in text or (
        "_eval_mode" in text and "use_diag" in text
    )


def _activation_quant_in_overlay(text: str) -> bool:
    """True when the overlay still quantizes activations, not weight-only."""
    blob = text or ""
    lowered = blob.lower()
    if "quantizer = torch.nn.identity" in lowered or "quantizer = nn.identity" in lowered:
        return False
    if "disable_quant" in lowered and "true" in lowered:
        return False
    return (
        "nn.Quantizer()" in blob
        or "Quantizer()" in blob
        or "act_quant" in lowered
        or "a_bits" in lowered
    )


def _maybe_record_best_quality(
    request: dict,
    *,
    job_id: str,
    overlay_path: str | None,
    ppl_ratio,
) -> dict:
    """Keep the lowest WikiText-2 ppl_ratio overlay as the next-fix base."""
    if not isinstance(ppl_ratio, (int, float)) or not math.isfinite(float(ppl_ratio)):
        return request
    ratio = float(ppl_ratio)
    prev = request.get("best_ppl_ratio")
    prev_f = None
    try:
        if prev is not None and prev != "":
            prev_f = float(prev)
    except (TypeError, ValueError):
        prev_f = None
    if prev_f is not None and ratio >= prev_f:
        return request
    request["best_ppl_ratio"] = ratio
    request["best_job_id"] = str(job_id)
    if overlay_path:
        request["best_overlay_dir"] = str(overlay_path).rstrip("/")
    return request


def _uses_prefill_kernels(text: str) -> bool:
    """Packed GEMM is not enough: WikiText-2 tok/s is 2048-token prefill."""
    attn = (
        "scaled_dot_product_attention" in text
        or "flash_attention" in text
        or "_flash_attention_forward" in text
    )
    fused = (
        "kron_matmul" in text
        or "OnlineTrans" in text
        or "fused_trans" in text
    )
    return attn or fused


def _kernel_overlay_count(tried_overlays: list[str] | None) -> int:
    return sum(
        1
        for raw in tried_overlays or []
        if "kernel_triton" in str(raw).replace("\\", "/")
    )


def _allow_gpu(
    budget: dict,
    *,
    packed_path: bool = False,
) -> bool:
    """Quality retries respect remaining. Packed-path fixes may use overage."""
    if budget["remaining"] > 0:
        return True
    if packed_path and max(0, budget["used"] - budget["max"]) < PACKED_PATH_OVERAGE:
        return True
    return False


def _overlay_blob(overlay_path: Path, script_text: str) -> str:
    parts = [script_text]
    patch = overlay_path / "overlay.patch"
    if patch.is_file():
        parts.append(patch.read_text(errors="replace"))
    adapter = overlay_path / "quant_agent_inference_adapter.py"
    if adapter.is_file():
        parts.append(adapter.read_text(errors="replace"))
    return "\n".join(parts)


def _tried_keys(tried_overlays: list[str], current_overlay: str | None) -> set[str]:
    keys: set[str] = set()
    for raw in list(tried_overlays) + ([current_overlay] if current_overlay else []):
        if not raw:
            continue
        keys.update(_overlay_keys(_resolve_overlay_dir(str(raw))))
        keys.add(str(raw).rstrip("/"))
    return keys


def _script_for_overlay(overlay_path: Path) -> Path:
    parent_script = overlay_path.parent / "quantize.py"
    nested = overlay_path / "quantize.py"
    if parent_script.is_file():
        return parent_script
    return nested


def _score_candidate(
    sig: dict, script_text: str, overlay_text: str | None = None
) -> tuple[int, int, int, int, int, int, int] | None:
    if sig.get("drops_runtime") or (
        sig.get("unwraps_to_nn_linear") and not sig.get("keeps_runtime")
    ):
        return None
    if not (
        sig.get("keeps_runtime")
        or sig.get("fuses_transform_into_layernorm")
        or not sig.get("unwraps_to_nn_linear")
    ):
        return None
    blob = overlay_text if overlay_text is not None else script_text
    fused_arch = any(token in script_text for token in _ARCH_TOKENS)
    aliased = "export_llama_named" in script_text
    return (
        int(bool(sig.get("saves_packed_int4"))),
        int(_casts_kernel_fp16(blob)),
        int(_restores_eval_runtime(blob)),
        int(bool(sig.get("keeps_runtime"))),
        int(bool(sig.get("fuses_transform_into_layernorm"))),
        int(fused_arch),
        int(not aliased),
    )


def retry_budget(request: dict | None) -> dict:
    payload = request or {}
    used = int(payload.get("retry_gpu_jobs_used") or 0)
    raw_max = payload.get("retry_gpu_jobs_max", RETRY_GPU_JOBS_MAX)
    if raw_max is None or raw_max == "":
        maximum = RETRY_GPU_JOBS_MAX
    else:
        maximum = int(raw_max)
    remaining = max(0, maximum - used)
    return {"used": used, "max": maximum, "remaining": remaining}


def generic_issue_codes(issues: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for issue in issues:
        code = GENERIC_CODES.get(issue, issue)
        if code not in seen:
            seen.add(code)
            ordered.append(code)
    return ordered


def classify(
    *,
    script_signals: dict,
    comparison: dict | None,
    checkpoint: dict | None,
    ranked: list[str],
    current_overlay: str | None,
    tried_overlays: list[str] | None = None,
    retry_gpu_jobs_used: int = 0,
    retry_gpu_jobs_max: int = RETRY_GPU_JOBS_MAX,
    extra_overlays: list[str] | None = None,
    current_patch_sha: str | None = None,
    verification_status: str | None = None,
    verification_error: str | None = None,
    best_overlay_dir: str | None = None,
    best_ppl_ratio: float | None = None,
) -> dict:
    issues: list[str] = []
    notes: list[str] = []

    if script_signals.get("drops_runtime") or (
        script_signals.get("unwraps_to_nn_linear")
        and not script_signals.get("keeps_runtime")
    ):
        issues.append(ISSUE_UNWRAP_NO_LN_FUSE)
        notes.append(
            "Save path drops method runtime (vanilla HF Linear, unwrap, or copy of "
            "inner .linear.weight). reparameterize_model only fuses diag(T) into "
            "RMSNorm; inference still needs T(x) inside the method wrappers."
        )

    if checkpoint and checkpoint.get("dense_float_only"):
        issues.append(ISSUE_DENSE_FP16)
        issues.append(ISSUE_FAKEQUANT_NO_PACK)
        notes.append(
            "Saved checkpoint is dense float16/bfloat16 tensors. Low-bit values "
            "stored as fp16 cannot beat packed int4 VRAM; use the method's packed "
            "export or a kernel overlay."
        )

    current_blob = ""
    if current_overlay:
        current_blob = _overlay_blob(_resolve_overlay_dir(current_overlay), "")
    packed_from_overlay = bool(current_blob) and bool(
        analyze_script(current_blob).get("saves_packed_int4")
    )
    packed_artifact = bool(checkpoint and checkpoint.get("packed_int4")) or bool(
        script_signals.get("saves_packed_int4")
    ) or packed_from_overlay
    verify_failed = str(verification_status or "").lower() == "failed"
    if verify_failed and packed_artifact and _kernel_dtype_mismatch(verification_error):
        issues.append(ISSUE_KERNEL_DTYPE)
        notes.append(
            "Packed int4 generate hit a CUDA kernel dtype assert (scale_row/"
            "scale_col must be float16). Empty Linear4bit buffers default to "
            "float32, so load_state_dict keeps float32. Cast kernel-facing "
            "scales after pack and after load; do not fall back to dense fakequant."
        )
    elif verify_failed and packed_artifact:
        issues.append(ISSUE_PACKED_LOADER)
        notes.append(
            "Packed artifact failed verify. Stay on the packed/realquant path "
            "and author a loader fix; do not retry a dense ranked overlay."
        )

    ppl_ratio = None
    quality_ok = None
    if comparison:
        ppl_ratio = comparison.get("ppl_ratio")
        quality_ok = comparison.get("quality_ok")
        if isinstance(ppl_ratio, (int, float)) and math.isfinite(ppl_ratio):
            if ppl_ratio >= PPL_EXPLODE_RATIO:
                issues.append(ISSUE_PPL_EXPLODED)
            if ppl_ratio > QUALITY_OK_RATIO:
                quality_ok = False
        if comparison.get("improved_vram") is False:
            vram_delta = comparison.get("peak_vram_delta_gb")
            if vram_delta is None or abs(float(vram_delta)) <= VRAM_EPS_GB:
                issues.append(ISSUE_VRAM_UNCHANGED)
        if comparison.get("improved_throughput") is False:
            issues.append(ISSUE_THROUGHPUT_UNCHANGED)

    if (
        packed_artifact
        and ISSUE_THROUGHPUT_UNCHANGED in issues
        and not _uses_prefill_kernels(current_blob)
    ):
        issues.append(ISSUE_PREFILL_KERNEL)
        notes.append(
            "Packed GEMM is in place but the overlay still uses naive attention "
            "or unfused Python transform+quant. WikiText-2 tok/s is 2048-token "
            "prefill; kernel follow-up should use SDPA/flash and, if this repo "
            "ships a fused T+quant kernel, call that module directly with "
            "Python-scalar Triton args (0-d tensors compile as pointers)."
        )

    if packed_artifact and (
        ISSUE_PPL_EXPLODED in issues or quality_ok is False
    ):
        if _restores_eval_runtime(current_blob):
            issues.append(ISSUE_PACKED_QUALITY_GAP)
            notes.append(
                "Packed weights already restore eval-only flags (_eval_mode / "
                "use_diag=False). Do not author another eval-flag restore. "
                "Quality is still above the 1.5x WikiText-2 gate: patch a new "
                "official-repo path, not the last hypothesis."
            )
            if best_overlay_dir:
                notes.append(
                    "Start the next overlay from best_overlay_dir="
                    f"{best_overlay_dir} (lowest ppl_ratio on this slug). "
                    "Do not start from a regressed last overlay."
                )
            if (
                best_ppl_ratio is not None
                and ppl_ratio is not None
                and isinstance(ppl_ratio, (int, float))
                and math.isfinite(float(ppl_ratio))
                and float(ppl_ratio) > float(best_ppl_ratio)
            ):
                notes.append(
                    "Last GPU job regressed versus best_ppl_ratio. Base the "
                    "patch on best_overlay_dir."
                )
            if _activation_quant_in_overlay(current_blob):
                notes.append(
                    "This overlay still quantizes activations. If the method "
                    "repo has a weight-only or fp16-activation class "
                    "(Identity quantizer, disable_quant, a_bits=16) versus an "
                    "activation-quant class, try that official path on the "
                    "packed weights."
                )
            notes.append(
                "If this model family has tensors the repo's default Llama "
                "path skips (bias, tied embeddings, GQA), apply the same "
                "rotate/fuse the repo uses for those. Stay packed; do not "
                "reparameterize packed Linear4bit; do not fall back to dense "
                "fakequant. Do not retune from raw PPL."
            )
        else:
            issues.append(ISSUE_EVAL_FLAGS)
            notes.append(
                "Packed/fused weights plus exploded WikiText-2 usually means "
                "eval-only flags were not restored (_eval_mode, use_diag=False). "
                "Those flags are not in state_dict. Do not reparameterize packed "
                "Linear4bit again; do not fall back to dense fakequant."
            )

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for issue in issues:
        if issue not in seen:
            seen.add(issue)
            ordered.append(issue)

    next_overlay = None
    next_script = None
    action = ACTION_NONE
    budget = retry_budget(
        {
            "retry_gpu_jobs_used": retry_gpu_jobs_used,
            "retry_gpu_jobs_max": retry_gpu_jobs_max,
        }
    )
    quality_broken = (
        ISSUE_UNWRAP_NO_LN_FUSE in ordered
        or ISSUE_PPL_EXPLODED in ordered
        or ISSUE_KERNEL_DTYPE in ordered
        or ISSUE_PACKED_LOADER in ordered
        or ISSUE_EVAL_FLAGS in ordered
        or ISSUE_PACKED_QUALITY_GAP in ordered
        or quality_ok is False
    )
    require_packed = (
        ISSUE_KERNEL_DTYPE in ordered
        or ISSUE_PACKED_LOADER in ordered
        or ISSUE_EVAL_FLAGS in ordered
        or ISSUE_PACKED_QUALITY_GAP in ordered
    )
    if quality_ok is False and ISSUE_PPL_EXPLODED not in ordered:
        notes.append(
            "quality_ok is false (WikiText-2 PPL above the 1.5x fp16 gate); "
            "retry or author_fix instead of treating this as efficiency-only."
        )
    if quality_broken:
        skipped = _tried_keys(list(tried_overlays or []), current_overlay)
        best: tuple[tuple[int, int, int, int, int, int, int], str, str] | None = None
        seen_paths: set[str] = set()
        for raw in list(ranked) + list(extra_overlays or []):
            overlay_path = _resolve_overlay_dir(str(raw))
            marker = str(overlay_path).rstrip("/")
            if marker in seen_paths:
                continue
            seen_paths.add(marker)
            if _overlay_keys(overlay_path) & skipped:
                continue
            patch_sha = _patch_sha256(overlay_path)
            if current_patch_sha and patch_sha and patch_sha == current_patch_sha:
                continue
            script = _script_for_overlay(overlay_path)
            if not script.is_file():
                continue
            script_text = script.read_text(errors="replace")
            blob = _overlay_blob(overlay_path, script_text)
            sig = analyze_script(script_text)
            blob_sig = analyze_script(blob)
            if blob_sig.get("saves_packed_int4"):
                sig["saves_packed_int4"] = True
            if blob_sig.get("keeps_runtime"):
                sig["keeps_runtime"] = True
            if require_packed and not sig.get("saves_packed_int4"):
                continue
            if ISSUE_KERNEL_DTYPE in ordered and not _casts_kernel_fp16(blob):
                continue
            if ISSUE_EVAL_FLAGS in ordered and not _restores_eval_runtime(blob):
                continue
            if ISSUE_PACKED_QUALITY_GAP in ordered and not _restores_eval_runtime(blob):
                continue
            score = _score_candidate(sig, script_text, overlay_text=blob)
            if score is None:
                continue
            candidate = (score, str(overlay_path), str(script))
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is not None:
            next_overlay, next_script = best[1], best[2]
        allow = _allow_gpu(budget, packed_path=require_packed)
        if best is not None and allow:
            action = ACTION_RETRY_RANKED
            if budget["remaining"] == 0:
                notes.append(
                    "Quality GPU budget is 0; packed-path verify/eval-flag "
                    "fixes still retry an untried packed overlay (overage)."
                )
        elif allow:
            action = ACTION_AUTHOR_FIX
            notes.append(
            "No remaining ranked overlay avoids the save/runtime/kernel-dtype "
            "bug; author a diagnose_fix overlay from issue codes plus "
            "error_excerpt (the last traceback), not a generic packed overlay."
            )
        else:
            action = ACTION_NONE
            notes.append("Retry GPU budget exhausted; stop.")
            if best is not None:
                notes.append(
                    "Untried keep-runtime overlay is in next_overlay_dir; "
                    "run it only if the user extends the GPU budget."
                )
    elif not quality_broken:
        dense = ISSUE_DENSE_FP16 in ordered or ISSUE_FAKEQUANT_NO_PACK in ordered
        vram_stuck = ISSUE_VRAM_UNCHANGED in ordered
        slow = ISSUE_THROUGHPUT_UNCHANGED in ordered
        kernel_attempts = _kernel_overlay_count(tried_overlays)
        has_prefill = packed_artifact and _uses_prefill_kernels(current_blob)
        if packed_artifact and has_prefill and not dense:
            action = ACTION_NONE
            notes.append(
                "Packed artifact already uses SDPA/flash or this repo's fused "
                "T+quant kernel. Prefill kernel is complete; do not loop. A "
                "remaining tok/s gap is a method/GPU limit."
            )
        elif dense or vram_stuck or slow:
            if kernel_attempts >= KERNEL_ATTEMPT_CAP:
                action = ACTION_NONE
                notes.append(
                    "Kernel attempt cap reached (packed/realquant overlay plus "
                    "at most one prefill follow-up). Same cap for every method."
                )
            else:
                action = ACTION_KERNEL
                notes.append(
                    "Quality is inside the 1.5x WikiText-2 gate. Remaining gap "
                    "is efficiency: dense fakequant, unchanged VRAM, slower "
                    "tokens/s, or packed GEMM without a prefill kernel. Parent "
                    "launches quant-kernel (complete overlay: pack if the repo "
                    "has it, dtype cast, eval-flag restore, SDPA/flash). Not "
                    "another quality overlay. Packed-path kernel may run even "
                    "when retry.remaining is 0."
                )

    root = ordered[0] if ordered else "none"
    if ISSUE_UNWRAP_NO_LN_FUSE in ordered:
        root = ISSUE_UNWRAP_NO_LN_FUSE
    elif ISSUE_EVAL_FLAGS in ordered:
        root = ISSUE_EVAL_FLAGS
    elif ISSUE_PACKED_QUALITY_GAP in ordered:
        root = ISSUE_PACKED_QUALITY_GAP
    elif ISSUE_KERNEL_DTYPE in ordered:
        root = ISSUE_KERNEL_DTYPE
    elif ISSUE_PACKED_LOADER in ordered:
        root = ISSUE_PACKED_LOADER
    issue_codes = generic_issue_codes(ordered)
    return {
        "issues": ordered,
        "issue_codes": issue_codes,
        "root_cause": root,
        "root_cause_generic": GENERIC_CODES.get(root, root),
        "recommended_action": action,
        "next_overlay_dir": next_overlay,
        "next_script": next_script,
        "notes": notes,
        "ppl_ratio": ppl_ratio,
        "best_overlay_dir": best_overlay_dir,
        "best_ppl_ratio": best_ppl_ratio,
        "script_signals": script_signals,
        "checkpoint": checkpoint,
        "retry": budget,
    }


def diagnose_job(job_id: str, request: dict) -> dict:
    meta = jobs_mod.refresh_status(job_id)
    if meta.status not in {"completed", "failed"}:
        raise RuntimeError(
            f"diagnose requires a completed or failed job, got {meta.status}"
        )
    job_directory = jobs_mod.job_dir(job_id)
    script_path = Path(meta.script_path)
    if not script_path.is_file():
        script_path = job_directory / "script.py"
    code = script_path.read_text(errors="replace") if script_path.is_file() else ""
    script_signals = analyze_script(code)

    benchmark = _read_json(job_directory / "benchmark.json")
    comparison = None
    if isinstance(benchmark, dict):
        comparison = benchmark.get("comparison")
        if not isinstance(comparison, dict):
            comparison = None
    elif isinstance(meta.metrics, dict):
        comparison = meta.metrics.get("comparison") if isinstance(meta.metrics.get("comparison"), dict) else None

    output_dir = resolve_workspace_path(meta.output_dir)
    checkpoint = _tensor_meta(output_dir) if output_dir.is_dir() else None

    ranked = request.get("ranked") if isinstance(request.get("ranked"), list) else []
    ranked = [str(item) for item in ranked]
    tried = request.get("tried_overlays") if isinstance(request.get("tried_overlays"), list) else []
    tried = [str(item) for item in tried]
    header_overlay = _header_overlay(code)
    for extra in (
        request.get("winner_overlay_dir"),
        meta.overlay_path,
        header_overlay,
    ):
        if extra:
            tried.append(str(extra))
    slug = str(request.get("slug") or "")
    extra_overlays = []
    if slug:
        extra_overlays.extend(_authored_fix_overlays(slug))
        extra_overlays.extend(_authored_strategy_overlays(slug, "kernel_triton"))
    current_patch_sha = _patch_sha256(Path(meta.overlay_path)) if meta.overlay_path else None
    budget = retry_budget(request)
    best_overlay = request.get("best_overlay_dir")
    best_overlay_s = str(best_overlay).rstrip("/") if best_overlay else None
    best_ratio = None
    raw_best = request.get("best_ppl_ratio")
    try:
        if raw_best is not None and raw_best != "":
            best_ratio = float(raw_best)
    except (TypeError, ValueError):
        best_ratio = None
    report = classify(
        script_signals=script_signals,
        comparison=comparison,
        checkpoint=checkpoint,
        ranked=ranked,
        current_overlay=header_overlay or meta.overlay_path,
        tried_overlays=tried,
        retry_gpu_jobs_used=budget["used"],
        retry_gpu_jobs_max=budget["max"],
        extra_overlays=extra_overlays,
        current_patch_sha=current_patch_sha,
        verification_status=meta.verification_status,
        verification_error=meta.verification_error,
        best_overlay_dir=best_overlay_s,
        best_ppl_ratio=best_ratio,
    )
    try:
        logs = jobs_mod.tail(job_id, n_lines=200)
    except FileNotFoundError:
        logs = {"stderr.log": "", "stdout.log": ""}
    excerpt = error_excerpt(
        stderr=logs.get("stderr.log") or "",
        stdout=logs.get("stdout.log") or "",
        verification_error=meta.verification_error,
    )
    report.update(
        {
            "status": "diagnosed",
            "job_id": job_id,
            "overlay_path": meta.overlay_path,
            "output_dir": str(output_dir),
            "benchmark_status": meta.benchmark_status,
            "verification_status": meta.verification_status,
            "verification_error": meta.verification_error,
            "error_excerpt": excerpt,
            "prior_issue_codes": _prior_issue_codes(request),
        }
    )
    extra_notes = error_fix_notes(excerpt)
    if extra_notes:
        report["notes"] = list(report.get("notes") or []) + extra_notes
    overlay_for_best = header_overlay or meta.overlay_path
    if comparison and comparison.get("ppl_ratio") is not None:
        request = _maybe_record_best_quality(
            request,
            job_id=job_id,
            overlay_path=overlay_for_best,
            ppl_ratio=comparison.get("ppl_ratio"),
        )
        try:
            write_request(request)
        except (ValueError, OSError):
            pass
        report["best_overlay_dir"] = request.get("best_overlay_dir")
        report["best_job_id"] = request.get("best_job_id")
        report["best_ppl_ratio"] = request.get("best_ppl_ratio")
    dest = job_directory / "diagnose.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose why a quantized run missed LLM-metric or VRAM improvement"
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    payload = load_request(args.request)
    try:
        report = diagnose_job(args.job_id, payload)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "failed", "message": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
