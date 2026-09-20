---
name: quant-diagnose
description: >-
  After a completed run whose benchmark did not beat the fp16 baseline,
  classify why (script, checkpoint, WikiText-2 metrics) and recommend a
  bounded fix: retry the next ranked overlay, author a targeted overlay
  patch, or hand off to quant-kernel. Use as a quant-diagnose subagent.
---

# Quant Diagnose

You are a **subagent**. Do only diagnose (and a **validate-only** overlay patch
if the helper says `author_fix`). Need `job_id` and the request JSON.
Do not launch a GPU job. Do not invent a retry loop; the parent executes
`recommended_action` with a GPU budget. The helper is **method-agnostic**:
same issue codes for every method × model × GPU. Do not ask the parent
mid-stage. If `HF_TOKEN` is unset and `.env` exists,
`source agents/skills/_shared/load_env.sh .env` (never print values).

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

## Do

```bash
$S/diagnose.py --job-id <job_id> --request out/requests/<slug>.json
```

The helper inspects the launched script, saved tensors,
`jobs/<job_id>/benchmark.json`, **stderr/stdout traceback**, port `ranked`,
authored `diagnose_fix` overlays, wrapper `# QUANT_AGENT_OVERLAY_DIR=`,
`overlay.patch` sha256, `tried_overlays`, and `retry_gpu_jobs_*`. It writes
   `error_excerpt` (last traceback / verify error), `notes` inferred from that
   excerpt, `prior_issue_codes` from `last_diagnose_path`, and `best_overlay_dir`
   / `best_ppl_ratio` (lowest WikiText-2 ratio on this slug). Return its JSON.

Quality gate: WikiText-2 `ppl_ratio > 1.5` or `quality_ok: false` is a
**quality** failure (retry / author_fix), not kernel. PPL ≥ 3× is
`ppl_exploded`. Skip the overlay that just ran even when the job copied
it to `jobs/<id>/overlay`.

Known **specific** issues and **typed** `issue_codes`:

| issue | issue_code | meaning |
| --- | --- | --- |
| `unwrap_without_ln_fuse` | `transform_or_runtime_dropped_on_save` | Wrappers stripped, or inner `.linear.weight` copied into vanilla HF, so inference drops method runtime. Official reparam / LN fuse is not enough if save is vanilla Linear. |
| `dense_fp16_checkpoint` / `fakequant_not_packed` | `fakequant_saved_as_dense` | Low-bit values stored as fp16. Packed export was not written. |
| `packed_kernel_scale_dtype` | `cuda_kernel_dtype_mismatch` | Packed generate hit a CUDA scale dtype assert (`float16` `scale_row`/`scale_col`). Cast kernel-facing scales after pack and load. |
| `packed_loader_failed` | `packed_loader_failed` | Packed artifact failed verify for another loader reason. Stay on packed path. |
| `eval_runtime_flags_missing` | `eval_runtime_flags_missing` | Packed/fused weights loaded without restoring `_eval_mode` / `use_diag=False` (not in state_dict), so transforms apply twice and PPL explodes. Only when the current overlay does **not** already restore those flags. |
| `packed_quality_gap` | `packed_quality_gap` | Packed artifact, eval flags already restored, WikiText-2 still above 1.5×. Do not restore flags again. Start from `best_overlay_dir`. If the repo has a weight-only / fp16-activation class, try that official path. |
| `prefill_kernel_missing` | `prefill_kernel_missing` | Packed GEMM is saved but attention is still naive matmul (or T+quant is still unfused Python). WikiText-2 tok/s is 2048 prefill. |
| `ppl_ratio_exploded` | `ppl_exploded` | WikiText-2 PPL ≥ 3× fp16. |
| `vram_unchanged` / `throughput_unchanged` | same | No efficiency win. |

## Fix policy (bounded)

1. **`retry_ranked_overlay`** — helper picks an untried overlay (port `ranked`
   **plus** `out/overlays/<slug>/diagnose_fix/*`) that **keeps method runtime**
   (no vanilla unwrap / inner-weight export). Prefer native arch over an alias
   export. Return `next_overlay_dir` and `next_script`. The **parent** runs one
   `quant-run`. Budget exhausted still fills `next_overlay_dir` if a candidate
   exists.
2. **`author_fix`** — no remaining overlay keeps runtime, and retry budget
   remains. Author a new overlay with strategy `diagnose_fix` using
   `overlay.py write` + `apply-check` + `validate_script.py` +
   `adapter.py --check-hub-id`. Do not compile CUDA. Keep-runtime pattern
   (any method): save the method's transform/clip/packed payload plus an
   inference adapter that reloads **that artifact**, reapplies the method
   wrappers, and evals **with wrappers still on**. Do not copy inner
   `.linear.weight` into vanilla HF. Read `issue_codes`, `error_excerpt`,
   `notes`, and `prior_issue_codes` from diagnose JSON; also tail the job
   stderr. Patch the **last exception**, not a generic packed overlay. Do
   not retune from raw PPL. If `packed_quality_gap`, start from
   `best_overlay_dir` (not a regressed last overlay). Do not restore eval
   flags again. If the cloned repo has a weight-only or fp16-activation class
   versus an activation-quant class, try that official path on packed weights.
3. **`kernel`** — quality is inside the 1.5× gate but the run is still dense
   fakequant, VRAM-flat, slower tokens/s, or packed GEMM without a prefill
   kernel (`prefill_kernel_missing`). Parent launches `quant-kernel`. The
   **first** kernel overlay must be complete (pack/realquant if the repo has
   it, dtype cast, eval-flag restore, SDPA/flash). A **second** kernel is
   only if that overlay packed GEMM but still used naive attention. Helper
   returns `none` once SDPA/flash or the repo fused T+quant is present, or
   after two `kernel_triton` overlays. If a packed overlay fails verify with
   `cuda_kernel_dtype_mismatch`, that is `author_fix` / retry a packed overlay
   that casts scales — not “kernel already tried, stop” and not the next dense
   ranked overlay. Packed-path fixes may be recommended even when
   `retry.remaining` is 0 (overage cap in the helper).
4. **`none`** — budget exhausted, or nothing left to try. Report and stop.

Do not retry OOM-at-same-config, gated auth, or disk full. Never edit `.venvs/<slug>/repo`.

## Return

```text
root_cause: <specific code>
issue_codes: <comma-separated typed codes>
error_excerpt: <last traceback or ->
action: retry_ranked_overlay | author_fix | kernel | none
next_overlay: <path or ->
next_script: <path or ->
retry: used/max remaining
diagnose_json: jobs/<job_id>/diagnose.json
```
