---
name: quant-port
description: >-
  Adapt a gathered quantization method to the chosen model. Coordinates up to
  three named port-strategy subagents, validates their overlays without a GPU
  job, and returns a ranked winner. Use after quant-gather.
---

# Quant Port

If the prompt has **no** `strategy:`, you are the **coordinator**.
If it has `strategy: dispatch|llama_alias|adapter_only|diagnose_fix`, you are a **worker**.
Request JSON path is required. Never edit the cloned repo.

`PY=$(command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

## Coordinator

Do not write the overlay yourself. Launch **up to three** worker subagents
**in parallel**, each with a different strategy:

| strategy | When | What it tries |
| --- | --- | --- |
| `dispatch` | Repo already switches on architecture names | Add this family to that switch |
| `llama_alias` | Target looks Llama/Qwen-like | Map q/k/v/o/mlp names onto the repo’s Llama path |
| `adapter_only` | Save/load is the hard part | Leave the quantizer; add a small inference adapter |

Skip a strategy that cannot apply (no dispatch table, not Llama-like). Never invent
a fourth random style. Workers must **not** compile CUDA; that happened once in gather.

When they return, rank:

1. Validation passed
2. Smaller overlay
3. Fewer files touched

Write `winner_overlay_dir`, `winner_script`, and `ranked` back into the request
JSON via `request.py --write-json` (keep all required keys).

Return the ranked list. The parent runs **only the winner** on GPU. If verify
later fails with a loader/arch bug, the parent runs the **next** overlay — it
does not spawn three new authors.

## Worker

If it has `strategy: diagnose_fix`, you are a **bounded fix worker** (not a
fourth random port style). Read `jobs/<job_id>/diagnose.json`. Author one overlay
that addresses those **typed** `issue:` codes (for example keep-runtime save
for `transform_or_runtime_dropped_on_save`). Do not retune from raw PPL. Validate
only; do not launch a GPU job. Same worker for every method × model × GPU.

Do only your `strategy`. Read the request JSON, paper, and `.venvs/<slug>/repo`
(read-only). Author a unified diff and a wrapper script.

Write the overlay:

```bash
$S/overlay.py write --slug <slug> --strategy <strategy> --model-id <id> \
  --base-commit <40-char-sha> --patch-file /tmp/port.patch \
  --rationale "..." --evidence-file <path> --target-module <mod> \
  [--inference-adapter-path inference_adapter.py]
```

Validate without a GPU job:

```bash
$S/overlay.py apply-check --overlay-dir <bundle> --repo .venvs/<slug>/repo \
  --commit <sha>
$S/validate_script.py <script.py> --model-id <id> --output-dir ./quantized/<slug> \
  --overlay-dir <bundle>
$S/adapter.py <adapter.py> --check-hub-id   # adapter_only only
```

The wrapper must contain, in the first 20 lines:

```python
# QUANT_AGENT_OVERLAY_DIR=<absolute overlay bundle path>
```

and string literals `"QUANT_AGENT_OVERLAY_DIR"` and `"QUANT_AGENT_METHOD_REPO"`
(the launcher injects those env vars). `MODEL_ID` and `OUTPUT_DIR` must be the
exact request values. Do not `from_pretrained(model_id)` for the quantized load
path — load `OUTPUT_DIR` / `model_path`.

If the cloned repo has packed/realquant (`pack_i4`, `Linear4bit`,
`--quantized_save`, `deploy/` int4 matmul), the wrapper should **save that
packed artifact** after calibration, not dense fp16 RTN. Fakequant is the
calibrator; packed weights are what later stages reload. If the repo's CUDA
bindings assert `float16` scales (`sym_quant` / `sym_dequant`), the overlay
must cast kernel-facing `weight_scales` / activation scales to that dtype
on pack **and** on adapter load (`cast_linear4bit_kernel_dtypes` or
equivalent). Empty packed modules default to float32 buffers. The adapter
must also restore eval-only flags that `state_dict` does not store
(`restore_packed_eval_runtime`: `_eval_mode`, `use_diag=False`) and must
not call `reparameterize_model` again on already packed weights.

Place the wrapper next to the overlay as `out/overlays/<slug>/<strategy>/quantize.py`
or inside the returned bundle notes. Do not launch a job. Do not build custom kernels.

## Return (coordinator)

```text
winner: out/overlays/<slug>/dispatch/<hash>/
script: out/overlays/<slug>/dispatch/quantize.py
ranked: dispatch, llama_alias
failed: adapter_only (reason)
```
