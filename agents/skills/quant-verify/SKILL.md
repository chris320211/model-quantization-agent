---
name: quant-verify
description: >-
  Reload a completed quantization job’s saved artifact and require prompt
  generation. Use only as a quant-verify subagent launched by the parent quant
  skill after quant-run. Loader bugs go back to the next ranked overlay; do
  not treat process exit 0 as success. LLM metric comparison against fp16 is
  quant-benchmark, not this skill.
disable-model-invocation: true
---

# Quant Verify

You are a **subagent**. Do only verify. Need `job_id` and the request JSON.
Do not ask the parent mid-stage. If `HF_TOKEN` is unset and `.env` exists,
`source agents/skills/_shared/load_env.sh .env` (never print values).

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

## Do

```bash
$S/verify.py --job-id <job_id> --request out/requests/<slug>.json \
  --allow-unsafe-host-execution
```

The helper:

1. Refuses Hub-fp16 impersonation (`output_dir` must not be the HF snapshot;
   weight files must not hash-match the snapshot; adapters must load
   `model_path`, not `model_id` or a Hub string).
2. Reloads the **saved** weights (generic HF or overlay adapter) with Hub
   downloads disabled.
3. Requires at least one alphanumeric generated token.

`--baseline` is a VRAM-only generate smoke of the snapshot. It is **not** the
LLM metrics benchmark. After verify passes, the parent always launches
`quant-benchmark`.

Process exit 0 from the quantize job is **not** success. Only this script is.

If verification fails with an arch/loader bug on an **unpacked** artifact,
return `failed` and `loader_arch`; the parent diagnoses and may try the next
ranked overlay (counts against the retry GPU budget). If the saved format is
already packed/realquant (`packed_int4` / `Linear4bit`) and generate hits a
CUDA `float16` scale assert, return `failed` and `cuda_kernel_dtype_mismatch`;
the parent stays on the packed path (`author_fix`), and does **not** fall
back to dense fakequant. Do not author a new overlay here.

## Return

`passed` or `failed`, plus a one-line reason.
