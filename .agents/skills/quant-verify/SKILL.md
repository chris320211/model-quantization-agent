---
name: quant-verify
description: >-
  Reload a completed quantization job’s saved artifact and require prompt
  generation. Use as a quant-verify subagent after quant-run. Loader bugs go
  back to the next ranked overlay; do not treat process exit 0 as success.
---

# Quant Verify

You are a **subagent**. Do only verify. Need `job_id` and the request JSON.

`S=python .agents/skills/_shared/scripts`

## Do

```bash
$S/verify.py --job-id <job_id> --request out/requests/<slug>.json \
  --allow-unsafe-host-execution [--baseline]
```

The helper:

1. Refuses Hub-fp16 impersonation (`output_dir` must not be the HF snapshot;
   weight files must not hash-match the snapshot; adapters must load
   `model_path`, not `model_id` or a Hub string).
2. Reloads the **saved** weights (generic HF or overlay adapter) with Hub
   downloads disabled.
3. Requires at least one alphanumeric generated token.
4. With `--baseline`, smokes the **original snapshot** as fp16 and records
   `fp16_baseline.peak_vram_gb` plus `improved_vs_fp16`. Do not treat the
   quantized smoke as the baseline.

Process exit 0 from the quantize job is **not** success. Only this script is.

If verification fails with an arch/loader bug, return `failed` and say so; the
parent tries the next ranked overlay. Do not author a new overlay here.

## Return

`passed` or `failed`, plus a one-line reason. Include metrics if `--baseline`.
