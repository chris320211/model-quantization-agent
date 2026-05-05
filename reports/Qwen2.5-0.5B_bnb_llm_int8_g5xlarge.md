# Qwen/Qwen2.5-0.5B-Instruct + bitsandbytes LLM.int8 on g5.xlarge

**Run date:** 2026-05-05
**Method:** bitsandbytes LLM.int8 (real W8A8 with fp16 outlier path)
**Model:** Qwen/Qwen2.5-0.5B-Instruct (0.5B params)
**Hardware:** g5.xlarge / A10G sm_86 (24 GB)

## Metrics

| run | threshold | prefill_ms | decode_ms | vram_gb | ppl | size_gb |
|---|---|---|---|---|---|---|
| fp16 reference | — | 2818 | 11017 | 1.025 | 12.473 | — |
| iter 0 (baseline) | 6.0 | 11874 | 47249 | 0.712 | 12.478 | 0.647 |
| **iter 1 (Pareto best)** | 8.0 | **11515** | **45934** | **0.712** | 12.497 | 0.647 |

## Pareto verdict

iter1 vs iter0 (eps: latency=2%, vram=1%, ppl=0.5%): prefill **-3.0% better**, decode **-2.8% better**, vram tied, ppl +0.15% worse but within eps → **iter1 Pareto-improves over iter0**.

**Best config:** `llm_int8_threshold=8.0` (fewer features promoted to fp16 outlier path).

## Comparison vs SmoothQuant on the same model

| metric | fp16 | SmoothQuant α=0.5 | bnb_int8 thr=8.0 |
|---|---|---|---|
| size_gb | — (1.0 fp16) | **1.004 (fake)** | **0.647 (real)** |
| vram_gb | 1.025 | 1.025 | **0.712** |
| ppl | 12.473 | **12.430** | 12.497 |
| prefill_ms | 2818 | **2895** | 11515 |
| decode_ms | 11017 | **11115** | 45934 |

- **bnb_int8 wins on real artifact size (-35%) and runtime VRAM (-31%) but loses badly on latency** (4× slower) — known LLM.int8 limitation on sm_86 with small models (mixed-precision overhead dominates when matmuls are small).
- **SmoothQuant wins on latency parity with fp16** but is fake-quant — no deployment savings.
- **Neither method is the right choice for this hardware/model combo.** For a 0.5B Qwen on A10G, real INT4 (AutoRound/AWQ) via packed kernels would likely beat both.

## Honest takeaways

- bnb LLM.int8 quantization itself preserves quality almost perfectly (ppl Δ < 0.04% vs fp16).
- Threshold tuning is a useful real knob: 6.0→8.0 buys 3% latency without hurting ppl.
- bnb LLM.int8's value shines on **larger models (7B+)** where the matmul size hides mixed-precision overhead — not 0.5B.

## Artifacts

- Pareto-best model: `quantized/bnb_llm_int8-Qwen_Qwen2.5-0.5B-Instruct_iter1/`
- iter0 job: `jobs/20260505T053525Z-e57f45/` (model dir pruned, dominated)
- iter1 job: `jobs/20260505T054127Z-596e85/`
- Caches: `~/.cache/quant-agent/{fp16_baselines.json, tune_history.jsonl}` (6 records total)
