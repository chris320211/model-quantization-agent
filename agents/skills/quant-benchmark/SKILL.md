---
name: quant-benchmark
description: >-
  After a passed quant-verify, always compare LLM metrics for the saved
  quantized artifact against the original fp16 Hugging Face snapshot on this
  GPU. Use only as a quant-benchmark subagent launched by the parent quant
  skill. WikiText-2 perplexity is required; VRAM-only smoke is not this stage.
  Does not write library/catalog.json.
disable-model-invocation: true
---

# Quant Benchmark

You are a **subagent**. Do only benchmark. Need `job_id` and the request JSON.
Do not ask the parent mid-stage. If `HF_TOKEN` is unset and `.env` exists,
`source agents/skills/_shared/load_env.sh .env` (never print values).
Requires **quant-verify passed**. Do not treat generate-smoke or
`verify.py --baseline` as this report.

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

## Do

```bash
$S/benchmark.py --job-id <job_id> --request out/requests/<slug>.json \
  --allow-unsafe-host-execution
```

The helper **always** evaluates both sides on the same WikiText-2 test corpus:

1. Reloads the **saved** quantized artifact (authenticity checks from verify).
2. Reloads the **fp16 snapshot** as the baseline (not Hub download, not the
   quantized smoke).
3. Records perplexity, loss, tokens evaluated, tokens/s, and peak VRAM.
   Tokens/s is **steady-state 2048-token prefill** (the helper warms up one
   window first). Packed CUDA/Triton kernels compile on the first forward;
   do not treat that compile as the throughput result.
4. Writes `jobs/<job_id>/benchmark.json` and `metrics` on the job.

Do **not** record onto `library/catalog.json`. Failed and non-beneficial
WikiText-2 jobs stay off the public board. `quant-catalog` writes the row
after a beneficial publish.

Do not skip the fp16 side. Do not invent a different dataset unless the user
named one; the default is WikiText-2 test, 2048-token windows, 65536 tokens.

## Return

```text
status: passed
quantized_ppl: <float>
fp16_ppl: <float>
ppl_ratio: <quant / fp16>
quality_ok: <ratio <= 1.5>
peak_vram_gb: <quant> vs <fp16>
tokens_per_s: <quant> vs <fp16>
efficiency_improved: <vram or throughput better>
compare_best_method: <method_name or none from existing library query>
```

If the helper fails, return `failed` and the structured skill-step report.
Do not author overlays. Kernel is the parent's decision after this report.
