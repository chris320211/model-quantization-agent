# Contribute a successful run

Do not commit weight files. Do not edit `catalog.json`.

1. **Measure** with this repo’s WikiText-2 helper (2048-token windows, fp16 snapshot baseline, `quality_ok`).
2. **Upload weights** to a **public** Hugging Face model repo (`quant-publish --upload`).
3. **Add one JSON** in this folder and open a GitHub PR.

```bash
PY=$(command -v python || command -v python3)
S="$PY .agents/skills/_shared/scripts"
$S/compare.py --job-id <job_id> --request out/requests/<slug>.json \
  --hub-url https://huggingface.co/<org>/<name> --export-contribution
$S/compare.py --accept-contribution compare/contributions/<file>.json
```

Or write the JSON by hand (filename `{model}__{gpu}__{method}.json`):

```json
{
  "schema_version": 1,
  "model_id": "microsoft/Phi-3-mini-4k-instruct",
  "method_name": "AWQ",
  "gpu_instance": "g5.2xlarge",
  "gpu_name": "NVIDIA A10G",
  "quality_ok": true,
  "ppl": 7.0,
  "ppl_ratio": 1.1,
  "tokens_per_s": 4000.0,
  "peak_vram_gb": 4.0,
  "fp16_ppl": 6.36,
  "fp16_tokens_per_s": 3300.0,
  "fp16_peak_vram_gb": 9.7,
  "packed": true,
  "hub_repo_id": "your-org/your-model",
  "dataset": "wikitext-2-raw-v1",
  "max_seq_len": 2048
}
```

`ppl_ratio` must be `ppl / fp16_ppl` and ≤ 1.5. `hub_repo_id` must be public `org/name` so others can `huggingface-cli download` it. Same `model_id` + `gpu_instance` ranks your method next to existing rows; `is_best` is assigned automatically.
