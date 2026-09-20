# Contribute a successful run

See `library/README.md`. Weights stay on **your** public Hugging Face repo.

Do not commit weight files. Do not edit `catalog.json`.

1. Beneficial WikiText-2 run (`quality_ok` and better VRAM or tok/s than fp16).
2. `quant-publish --upload --repo-id <you>/<slug>`
3. `quant-catalog` (writes this JSON).
4. `quant-sync` (this GitHub remote + Hub collection). Without push access, open a PR.

```bash
PY=$(command -v python || command -v python3)
S="$PY agents/skills/_shared/scripts"
$S/library.py --catalog --job-id <job_id> --request out/requests/<slug>.json \
  --hub-url https://huggingface.co/<you>/<slug>
$S/library.py --rebuild
```

Required in the JSON: `model_id`, `gpu_instance`, `method_name`, `arxiv_id` or
`paper_url`, `repo_url` (`https://github.com/owner/repo`), `hub_repo_id`.

```json
{
  "schema_version": 1,
  "model_id": "microsoft/Phi-3-mini-4k-instruct",
  "method_name": "AWQ",
  "gpu_instance": "g5.2xlarge",
  "arxiv_id": "2306.00978",
  "paper_url": "https://arxiv.org/abs/2306.00978",
  "repo_url": "https://github.com/mit-han-lab/llm-awq",
  "hub_repo_id": "your-org/your-model",
  "quality_ok": true,
  "ppl": 7.0,
  "ppl_ratio": 1.1,
  "tokens_per_s": 4000.0,
  "peak_vram_gb": 4.0,
  "fp16_ppl": 6.36,
  "fp16_tokens_per_s": 3300.0,
  "fp16_peak_vram_gb": 9.7,
  "packed": true,
  "dataset": "wikitext-2-raw-v1",
  "max_seq_len": 2048
}
```
