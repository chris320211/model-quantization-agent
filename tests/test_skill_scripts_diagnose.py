"""Offline tests for quant-diagnose classification."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".agents" / "skills" / "_shared" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import diagnose as diagnose_mod  # noqa: E402


ADAPTER_UNWRAP = '''
def _unwrap_linears(model, flat_cls):
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, flat_cls):
                setattr(module, name, child.linear)

def _save_hf(model, tokenizer, output_dir, snapshot):
    model.save_pretrained(output_dir, safe_serialization=True)
'''

DISPATCH_FUSE = '''
import flatquant.flat_utils as flat_utils
flat_utils.reparameterize_model(model)
gptq_utils.rtn_fwrd(model, utils.DEV, args)
vanilla.save_pretrained(str(out))
qkv_proj
Phi3ForCausalLM
'''

KEEP_RUNTIME = '''
import flatquant.flat_utils as flat_utils
flat_utils.reparameterize_model(model)
flat_utils.load_flat_parameters(args, model, path=out)
torch.save({"state": "flatquant_runtime"}, os.path.join(out, "flat_parameters.pth"))
qkv_proj
Phi3ForCausalLM
gate_up_proj
'''


def test_analyze_script_detects_unwrap_without_ln_fuse():
    sig = diagnose_mod.analyze_script(ADAPTER_UNWRAP)
    assert sig["unwraps_to_nn_linear"] is True
    assert sig["fuses_transform_into_layernorm"] is False


def test_analyze_script_detects_official_reparam():
    sig = diagnose_mod.analyze_script(DISPATCH_FUSE)
    assert sig["official_reparameterize_model"] is True
    assert sig["fuses_transform_into_layernorm"] is True


def test_analyze_script_detects_vanilla_export_as_runtime_drop():
    sig = diagnose_mod.analyze_script(DISPATCH_FUSE)
    assert sig["drops_runtime"] is True
    assert sig["keeps_runtime"] is False
    assert sig["official_reparameterize_model"] is True


def test_analyze_script_detects_keep_runtime():
    sig = diagnose_mod.analyze_script(KEEP_RUNTIME)
    assert sig["drops_runtime"] is False
    assert sig["keeps_runtime"] is True


def test_classify_unwrap_recommends_ranked_overlay(tmp_path):
    ranked_dir = tmp_path / "dispatch" / "abc"
    ranked_dir.mkdir(parents=True)
    script = tmp_path / "dispatch" / "quantize.py"
    script.write_text(KEEP_RUNTIME)
    report = diagnose_mod.classify(
        script_signals=diagnose_mod.analyze_script(ADAPTER_UNWRAP),
        comparison={
            "ppl_ratio": 1000.0,
            "improved_vram": False,
            "improved_throughput": False,
            "peak_vram_delta_gb": 0.0,
        },
        checkpoint={"dense_float_only": True},
        ranked=[str(ranked_dir) + "/"],
        current_overlay=None,
    )
    assert diagnose_mod.ISSUE_UNWRAP_NO_LN_FUSE in report["issues"]
    assert diagnose_mod.ISSUE_PPL_EXPLODED in report["issues"]
    assert report["recommended_action"] == diagnose_mod.ACTION_RETRY_RANKED
    assert report["next_script"] == str(script)


def test_classify_prefers_fused_native_arch_over_alias(tmp_path):
    alias_dir = tmp_path / "llama_alias" / "aaa"
    alias_dir.mkdir(parents=True)
    (tmp_path / "llama_alias" / "quantize.py").write_text(
        DISPATCH_FUSE.replace("qkv_proj", "export_llama_named_hf_checkpoint")
        .replace("Phi3ForCausalLM", "LlamaForCausalLM")
        .replace("gate_up_proj", "mlp.down_proj")
    )
    native_dir = tmp_path / "dispatch" / "bbb"
    native_dir.mkdir(parents=True)
    (tmp_path / "dispatch" / "quantize.py").write_text(KEEP_RUNTIME)
    report = diagnose_mod.classify(
        script_signals=diagnose_mod.analyze_script(ADAPTER_UNWRAP),
        comparison={
            "ppl_ratio": 1000.0,
            "improved_vram": False,
            "improved_throughput": False,
            "peak_vram_delta_gb": 0.0,
        },
        checkpoint={"dense_float_only": True},
        ranked=[str(alias_dir) + "/", str(native_dir) + "/"],
        current_overlay=None,
    )
    assert report["recommended_action"] == diagnose_mod.ACTION_RETRY_RANKED
    assert report["next_script"].endswith("dispatch/quantize.py")
    assert diagnose_mod.CODE_TRANSFORM_DROPPED in report["issue_codes"]
    assert diagnose_mod.CODE_PPL_EXPLODED in report["issue_codes"]


def test_classify_skips_tried_overlay_and_respects_budget(tmp_path):
    ranked_dir = tmp_path / "dispatch" / "abc"
    ranked_dir.mkdir(parents=True)
    script = tmp_path / "dispatch" / "quantize.py"
    script.write_text(KEEP_RUNTIME)
    skipped = diagnose_mod.classify(
        script_signals=diagnose_mod.analyze_script(ADAPTER_UNWRAP),
        comparison={"ppl_ratio": 1000.0, "improved_vram": False, "improved_throughput": False},
        checkpoint={"dense_float_only": True},
        ranked=[str(ranked_dir) + "/"],
        current_overlay=None,
        tried_overlays=[str(ranked_dir)],
        retry_gpu_jobs_used=0,
        retry_gpu_jobs_max=2,
    )
    assert skipped["recommended_action"] == diagnose_mod.ACTION_AUTHOR_FIX
    exhausted = diagnose_mod.classify(
        script_signals=diagnose_mod.analyze_script(ADAPTER_UNWRAP),
        comparison={"ppl_ratio": 1000.0, "improved_vram": False, "improved_throughput": False},
        checkpoint={"dense_float_only": True},
        ranked=[str(ranked_dir) + "/"],
        current_overlay=None,
        retry_gpu_jobs_used=2,
        retry_gpu_jobs_max=2,
    )
    assert exhausted["recommended_action"] == diagnose_mod.ACTION_NONE
    assert exhausted["retry"]["remaining"] == 0
    assert exhausted["next_overlay_dir"] == str(ranked_dir.resolve())


def test_classify_vanilla_after_reparam_authors_fix(tmp_path):
    ranked_dir = tmp_path / "dispatch" / "abc"
    ranked_dir.mkdir(parents=True)
    (tmp_path / "dispatch" / "quantize.py").write_text(DISPATCH_FUSE)
    report = diagnose_mod.classify(
        script_signals=diagnose_mod.analyze_script(DISPATCH_FUSE),
        comparison={"ppl_ratio": 1e8, "improved_vram": False, "improved_throughput": False},
        checkpoint={"dense_float_only": True},
        ranked=[str(ranked_dir) + "/"],
        current_overlay=None,
        retry_gpu_jobs_used=0,
        retry_gpu_jobs_max=2,
    )
    assert diagnose_mod.ISSUE_UNWRAP_NO_LN_FUSE in report["issues"]
    assert report["recommended_action"] == diagnose_mod.ACTION_AUTHOR_FIX


def test_classify_efficiency_only_recommends_kernel():
    report = diagnose_mod.classify(
        script_signals={
            "unwraps_to_nn_linear": False,
            "fuses_transform_into_layernorm": True,
        },
        comparison={
            "ppl_ratio": 1.05,
            "improved_vram": False,
            "improved_throughput": False,
            "peak_vram_delta_gb": 0.0,
        },
        checkpoint={"dense_float_only": True},
        ranked=[],
        current_overlay=None,
    )
    assert diagnose_mod.ISSUE_PPL_EXPLODED not in report["issues"]
    assert report["recommended_action"] == diagnose_mod.ACTION_KERNEL


def test_classify_quality_ok_false_below_explode_still_retries(tmp_path):
    ranked_dir = tmp_path / "dispatch" / "abc"
    ranked_dir.mkdir(parents=True)
    script = tmp_path / "dispatch" / "quantize.py"
    script.write_text(KEEP_RUNTIME)
    report = diagnose_mod.classify(
        script_signals={
            "unwraps_to_nn_linear": False,
            "drops_runtime": False,
            "keeps_runtime": True,
            "fuses_transform_into_layernorm": True,
        },
        comparison={
            "ppl_ratio": 1.8,
            "quality_ok": False,
            "improved_vram": False,
            "improved_throughput": False,
            "peak_vram_delta_gb": 0.0,
        },
        checkpoint={"dense_float_only": True},
        ranked=[str(ranked_dir) + "/"],
        current_overlay=None,
        retry_gpu_jobs_used=0,
        retry_gpu_jobs_max=2,
    )
    assert diagnose_mod.ISSUE_PPL_EXPLODED not in report["issues"]
    assert report["recommended_action"] == diagnose_mod.ACTION_RETRY_RANKED
    assert report["next_script"] == str(script)


def test_classify_ppl_ratio_above_gate_without_quality_ok_flag(tmp_path):
    ranked_dir = tmp_path / "dispatch" / "abc"
    ranked_dir.mkdir(parents=True)
    script = tmp_path / "dispatch" / "quantize.py"
    script.write_text(KEEP_RUNTIME)
    report = diagnose_mod.classify(
        script_signals={
            "unwraps_to_nn_linear": False,
            "drops_runtime": False,
            "keeps_runtime": True,
            "fuses_transform_into_layernorm": True,
        },
        comparison={
            "ppl_ratio": 1.8,
            "improved_vram": False,
            "improved_throughput": False,
            "peak_vram_delta_gb": 0.0,
        },
        checkpoint={"dense_float_only": True},
        ranked=[str(ranked_dir) + "/"],
        current_overlay=None,
        retry_gpu_jobs_used=0,
        retry_gpu_jobs_max=2,
    )
    assert report["recommended_action"] == diagnose_mod.ACTION_RETRY_RANKED
    assert report["next_script"] == str(script)


def test_classify_searches_extra_diagnose_fix_overlays(tmp_path):
    ranked_dir = tmp_path / "dispatch" / "abc"
    ranked_dir.mkdir(parents=True)
    (tmp_path / "dispatch" / "quantize.py").write_text(DISPATCH_FUSE)
    fix_dir = tmp_path / "diagnose_fix" / "newhash"
    fix_dir.mkdir(parents=True)
    (tmp_path / "diagnose_fix" / "quantize.py").write_text(KEEP_RUNTIME)
    (fix_dir / "overlay.patch").write_text("diff --git a/x b/x\n")
    report = diagnose_mod.classify(
        script_signals=diagnose_mod.analyze_script(DISPATCH_FUSE),
        comparison={"ppl_ratio": 1e6, "quality_ok": False, "improved_vram": True},
        checkpoint={"dense_float_only": True},
        ranked=[str(ranked_dir) + "/"],
        current_overlay=None,
        extra_overlays=[str(fix_dir)],
        retry_gpu_jobs_used=0,
        retry_gpu_jobs_max=2,
    )
    assert report["recommended_action"] == diagnose_mod.ACTION_RETRY_RANKED
    assert report["next_overlay_dir"] == str(fix_dir.resolve())


def test_retry_budget_honors_explicit_zero_max():
    budget = diagnose_mod.retry_budget({"retry_gpu_jobs_used": 0, "retry_gpu_jobs_max": 0})
    assert budget["max"] == 0
    assert budget["remaining"] == 0


def test_classify_quality_ok_dense_or_slow_recommends_kernel():
    report = diagnose_mod.classify(
        script_signals={
            "unwraps_to_nn_linear": False,
            "drops_runtime": False,
            "keeps_runtime": True,
            "fuses_transform_into_layernorm": True,
        },
        comparison={
            "ppl_ratio": 1.17,
            "quality_ok": True,
            "improved_vram": True,
            "improved_throughput": False,
            "peak_vram_delta_gb": -0.7,
        },
        checkpoint={"dense_float_only": True},
        ranked=[],
        current_overlay=None,
    )
    assert diagnose_mod.ISSUE_PPL_EXPLODED not in report["issues"]
    assert report["recommended_action"] == diagnose_mod.ACTION_KERNEL


PACKED_SCRIPT = KEEP_RUNTIME + """
pack_i4
Linear4bit
packed_int4
cast_linear4bit_kernel_dtypes
torch.float16
weight_scales
"""


def test_kernel_dtype_mismatch_detects_sym_dequant_assert():
    err = (
        "deploy.sym_dequant\n"
        "assert scale_row.dtype == scale_col.dtype == torch.float16\n"
        "AssertionError"
    )
    assert diagnose_mod._kernel_dtype_mismatch(err) is True
    assert diagnose_mod._kernel_dtype_mismatch("OOM") is False


def test_classify_packed_verify_dtype_authors_fix_not_dense_ranked(tmp_path):
    ranked_dir = tmp_path / "dispatch" / "abc"
    ranked_dir.mkdir(parents=True)
    (tmp_path / "dispatch" / "quantize.py").write_text(KEEP_RUNTIME)
    report = diagnose_mod.classify(
        script_signals={
            "unwraps_to_nn_linear": False,
            "drops_runtime": False,
            "keeps_runtime": True,
            "fuses_transform_into_layernorm": True,
            "saves_packed_int4": True,
        },
        comparison=None,
        checkpoint={"dense_float_only": False, "packed_int4": True},
        ranked=[str(ranked_dir) + "/"],
        current_overlay=None,
        retry_gpu_jobs_used=0,
        retry_gpu_jobs_max=2,
        verification_status="failed",
        verification_error=(
            "File deploy/nn/linear.py\n"
            "return deploy.sym_dequant(x, scales_x, self.weight_scales)\n"
            "assert scale_row.dtype == scale_col.dtype == torch.float16\n"
            "AssertionError"
        ),
    )
    assert diagnose_mod.ISSUE_KERNEL_DTYPE in report["issues"]
    assert diagnose_mod.CODE_KERNEL_DTYPE in report["issue_codes"]
    assert report["recommended_action"] == diagnose_mod.ACTION_AUTHOR_FIX
    assert report["next_overlay_dir"] is None


def test_classify_packed_verify_dtype_retries_fp16_cast_overlay(tmp_path):
    dense_dir = tmp_path / "dispatch" / "abc"
    dense_dir.mkdir(parents=True)
    (tmp_path / "dispatch" / "quantize.py").write_text(KEEP_RUNTIME)
    packed_dir = tmp_path / "kernel_triton" / "fp16cast"
    packed_dir.mkdir(parents=True)
    (tmp_path / "kernel_triton" / "quantize.py").write_text(PACKED_SCRIPT)
    (packed_dir / "overlay.patch").write_text(
        "diff --git a/x b/x\n+cast_linear4bit_kernel_dtypes\n+weight_scales\n+torch.float16\n"
    )
    report = diagnose_mod.classify(
        script_signals={
            "unwraps_to_nn_linear": False,
            "drops_runtime": False,
            "keeps_runtime": True,
            "fuses_transform_into_layernorm": True,
            "saves_packed_int4": True,
        },
        comparison=None,
        checkpoint={"dense_float_only": False, "packed_int4": True},
        ranked=[str(dense_dir) + "/"],
        current_overlay=None,
        extra_overlays=[str(packed_dir)],
        retry_gpu_jobs_used=0,
        retry_gpu_jobs_max=2,
        verification_status="failed",
        verification_error=(
            "deploy.sym_dequant\nassert scale_row.dtype == scale_col.dtype == torch.float16"
        ),
    )
    assert report["recommended_action"] == diagnose_mod.ACTION_RETRY_RANKED
    assert report["next_overlay_dir"] == str(packed_dir.resolve())
    assert report["next_script"].endswith("kernel_triton/quantize.py")


def test_classify_packed_ppl_explode_requires_eval_flag_overlay(tmp_path):
    dense_dir = tmp_path / "dispatch" / "abc"
    dense_dir.mkdir(parents=True)
    (tmp_path / "dispatch" / "quantize.py").write_text(KEEP_RUNTIME)
    packed_dir = tmp_path / "kernel_triton" / "evalflags"
    packed_dir.mkdir(parents=True)
    (tmp_path / "kernel_triton" / "quantize.py").write_text(
        PACKED_SCRIPT + "\nrestore_packed_eval_runtime\nuse_diag\n_eval_mode\n"
    )
    (packed_dir / "overlay.patch").write_text(
        "diff --git a/x b/x\n+restore_packed_eval_runtime\n+use_diag\n+_eval_mode\n"
        "+cast_linear4bit_kernel_dtypes\n+weight_scales\n+torch.float16\n"
    )
    report = diagnose_mod.classify(
        script_signals={
            "unwraps_to_nn_linear": False,
            "drops_runtime": False,
            "keeps_runtime": True,
            "fuses_transform_into_layernorm": True,
            "saves_packed_int4": True,
        },
        comparison={
            "ppl_ratio": 1e7,
            "quality_ok": False,
            "improved_vram": True,
            "improved_throughput": False,
        },
        checkpoint={"dense_float_only": False, "packed_int4": True},
        ranked=[str(dense_dir) + "/"],
        current_overlay=None,
        extra_overlays=[str(packed_dir)],
        retry_gpu_jobs_used=0,
        retry_gpu_jobs_max=2,
        verification_status="passed",
    )
    assert diagnose_mod.ISSUE_EVAL_FLAGS in report["issues"]
    assert diagnose_mod.CODE_EVAL_FLAGS in report["issue_codes"]
    assert report["recommended_action"] == diagnose_mod.ACTION_RETRY_RANKED
    assert report["next_overlay_dir"] == str(packed_dir.resolve())


def test_classify_packed_slow_without_sdpa_is_prefill_kernel_missing(tmp_path):
    overlay = tmp_path / "kernel_triton" / "packed"
    overlay.mkdir(parents=True)
    (overlay / "overlay.patch").write_text(
        "diff --git a/x b/x\n+Linear4bit\n+pack_i4\n+torch.matmul(query, key)\n"
    )
    report = diagnose_mod.classify(
        script_signals={
            "unwraps_to_nn_linear": False,
            "drops_runtime": False,
            "keeps_runtime": True,
            "fuses_transform_into_layernorm": True,
            "saves_packed_int4": True,
        },
        comparison={
            "ppl_ratio": 1.17,
            "quality_ok": True,
            "improved_vram": True,
            "improved_throughput": False,
            "peak_vram_delta_gb": -5.0,
        },
        checkpoint={"dense_float_only": False, "packed_int4": True},
        ranked=[],
        current_overlay=str(overlay),
        retry_gpu_jobs_used=6,
        retry_gpu_jobs_max=6,
    )
    assert diagnose_mod.ISSUE_PREFILL_KERNEL in report["issues"]
    assert report["recommended_action"] == diagnose_mod.ACTION_KERNEL
    assert report["retry"]["remaining"] == 0


def test_classify_packed_slow_with_sdpa_is_not_prefill_missing(tmp_path):
    overlay = tmp_path / "kernel_triton" / "sdpa"
    overlay.mkdir(parents=True)
    (overlay / "overlay.patch").write_text(
        "diff --git a/x b/x\n+scaled_dot_product_attention\n+kron_matmul\n"
    )
    report = diagnose_mod.classify(
        script_signals={
            "unwraps_to_nn_linear": False,
            "drops_runtime": False,
            "keeps_runtime": True,
            "fuses_transform_into_layernorm": True,
            "saves_packed_int4": True,
        },
        comparison={
            "ppl_ratio": 1.17,
            "quality_ok": True,
            "improved_vram": True,
            "improved_throughput": False,
            "peak_vram_delta_gb": -5.0,
        },
        checkpoint={"dense_float_only": False, "packed_int4": True},
        ranked=[],
        current_overlay=str(overlay),
    )
    assert diagnose_mod.ISSUE_PREFILL_KERNEL not in report["issues"]
    assert report["recommended_action"] == diagnose_mod.ACTION_NONE


def test_classify_kernel_attempt_cap_stops(tmp_path):
    overlay = tmp_path / "kernel_triton" / "third"
    overlay.mkdir(parents=True)
    (overlay / "overlay.patch").write_text(
        "diff --git a/x b/x\n+Linear4bit\n+pack_i4\n+torch.matmul(query, key)\n"
    )
    first = tmp_path / "kernel_triton" / "first"
    second = tmp_path / "kernel_triton" / "second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    report = diagnose_mod.classify(
        script_signals={
            "unwraps_to_nn_linear": False,
            "drops_runtime": False,
            "keeps_runtime": True,
            "fuses_transform_into_layernorm": True,
            "saves_packed_int4": True,
        },
        comparison={
            "ppl_ratio": 1.17,
            "quality_ok": True,
            "improved_vram": True,
            "improved_throughput": False,
            "peak_vram_delta_gb": -5.0,
        },
        checkpoint={"dense_float_only": False, "packed_int4": True},
        ranked=[],
        current_overlay=str(overlay),
        tried_overlays=[str(first), str(second)],
        retry_gpu_jobs_used=2,
        retry_gpu_jobs_max=2,
    )
    assert report["recommended_action"] == diagnose_mod.ACTION_NONE


def test_classify_packed_dtype_still_fixes_when_quality_budget_exhausted(tmp_path):
    report = diagnose_mod.classify(
        script_signals={
            "unwraps_to_nn_linear": False,
            "drops_runtime": False,
            "keeps_runtime": True,
            "fuses_transform_into_layernorm": True,
            "saves_packed_int4": True,
        },
        comparison=None,
        checkpoint={"dense_float_only": False, "packed_int4": True},
        ranked=[],
        current_overlay=None,
        retry_gpu_jobs_used=2,
        retry_gpu_jobs_max=2,
        verification_status="failed",
        verification_error=(
            "deploy.sym_dequant\nassert scale_row.dtype == scale_col.dtype == torch.float16"
        ),
    )
    assert diagnose_mod.ISSUE_KERNEL_DTYPE in report["issues"]
    assert report["recommended_action"] == diagnose_mod.ACTION_AUTHOR_FIX
    assert report["retry"]["remaining"] == 0

