---
name: quant-port
description: >-
  Adapt a gathered quantization method to the chosen model. Coordinates up to
  three named port-strategy subagents, validates their overlays without a GPU
  job, and returns a ranked winner. Use after quant-gather.
---

# Quant Port

If the prompt has **no** `strategy:`, you are the **coordinator**.
If it has `strategy: dispatch|llama_alias|adapter_only`, you are a **worker**.
Request JSON path is required. Never edit the cloned repo.

`S=python .agents/skills/_shared/scripts`

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

Place the wrapper next to the overlay as `out/overlays/<slug>/<strategy>/quantize.py`
or inside the returned bundle notes. Do not launch a job. Do not build custom kernels.

## Return (coordinator)

```text
winner: out/overlays/<slug>/dispatch/<hash>/
script: out/overlays/<slug>/dispatch/quantize.py
ranked: dispatch, llama_alias
failed: adapter_only (reason)
```
