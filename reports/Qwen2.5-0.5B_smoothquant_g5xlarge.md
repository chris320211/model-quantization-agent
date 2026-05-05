# Qwen/Qwen2.5-0.5B-Instruct + SmoothQuant on g5.xlarge

**Run date:** 2026-05-05
**Method:** SmoothQuant W8A8 (mit-han-lab/smoothquant repo, fake-quant)
**Model:** Qwen/Qwen2.5-0.5B-Instruct (0.5B params, Qwen2 arch)
**Hardware:** g5.xlarge / A10G sm_86 (24 GB)
**Patch applied:** Added Qwen2 dispatch to `smooth.py` and `fake_quant.py` (Llama-like)

## Metrics (wikitext-2-raw-v1 ppl, prefill 2k+128, decode 32+512)

| run | alpha | prefill_ms | decode_ms | vram_gb | ppl | size_gb |
|---|---|---|---|---|---|---|
| fp16 reference | — | 2818 | 11017 | 1.025 | 12.473 | — |
| **iter 0 (Pareto best)** | 0.5 | 2895 | 11115 | 1.025 | **12.430** | 1.004 |
| iter 1 | 0.7 | 2961 | 11358 | 1.025 | 12.424 | 1.004 |

## Pareto verdict

iter1 vs iter0 (eps: latency=2%, vram=1%, ppl=0.5%): prefill +2.3% worse, decode +2.2% worse, ppl tied (-0.05% within eps), vram tied → **iter1 dominated** → STAGNATE → loop terminated.

**Best config:** alpha=0.5, weight_quant=per_channel, act_quant=per_token.

## Honest takeaways

- This SmoothQuant research repo is **fake-quant**: weights stored as fp16 in safetensors (1.0 GB = 0.5B × 2 bytes), forward simulates W8A8 via quantize/dequantize. **No deployment savings.**
- Quantization **error** is real: ppl=12.43 reflects what real W8A8 would produce.
- Real W8A8 requires CUTLASS INT8 kernels via vLLM / TRT-LLM / torchao.

## Artifacts

- Quantized model: `quantized/smoothquant-Qwen_Qwen2.5-0.5B-Instruct/`
- iter0 job: `jobs/20260505T051532Z-e9282f/`
- iter1 job: `jobs/20260505T052432Z-c612fb/` (dominated, model dir pruned)
- Persistent caches: `~/.cache/quant-agent/{fp16_baselines.json, tune_history.jsonl}`
