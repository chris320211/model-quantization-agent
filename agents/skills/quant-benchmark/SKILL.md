---
name: quant-benchmark
description: >-
  After a passed quant-verify, always compare LLM metrics for the saved
  quantized artifact against the original fp16 Hugging Face snapshot on this
  GPU. Use as a quant-benchmark subagent. WikiText-2 perplexity is required;
  VRAM-only smoke is not this stage.
---

# Quant Benchmark

You are a **subagent**. Do only benchmark. Need `job_id` and the request JSON.
Requires **quant-verify passed**. Do not treat generate-smoke or
`verify.py --baseline` as this report.

`PY=$(command -v python || command -v python3)`
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
5. Records the method onto `compare/catalog.json` and
   `compare/groups/<model>__<gpu>.json` so library users can filter
   model / method / instance. Hugging Face Hub is not this index.

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
compare_best_method: <method_name or none>
compare_path: compare/groups/<model>__<gpu>.json
```

If the helper fails, return `failed` and the structured skill-step report.
Do not author overlays. Kernel is the parent's decision after this report.
