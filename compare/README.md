# Quantized method comparison

Filter `catalog.json` by `model_id`, `method_name`, `gpu_instance`.
Compare WikiText-2 metrics on the matching rows. Fetch weights yourself
with `huggingface-cli download <hub_repo_id>` (Hub stores files; this table
is the library index).

```bash
PY=$(command -v python || command -v python3)
S="$PY agents/skills/_shared/scripts"
$S/compare.py --model-id <org/model> --gpu-instance <instance>
$S/compare.py --model-id <org/model> --method <name> --gpu-instance <instance> --fetch
```

Rank: `quality_ok`, then tokens/s, then lower VRAM, packed/realquant, then lower PPL ratio.

## Contribute a run

Do not edit `catalog.json`. Upload public weights to Hugging Face, then add one
JSON file under `contributions/` and open a PR. Schema: `contributions/README.md`.

## microsoft/Phi-3-mini-4k-instruct on g5.2xlarge
GPU: NVIDIA A10G

| Rank | Method | PPL ratio | tok/s | VRAM GB | packed | quality_ok | pull |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | FlatQuant | 1.1676 | 5105.9 | 3.355 | yes | yes | quantized/flatquant-phi3-mini-4k-g52xlarge |

**Best:** FlatQuant (`flatquant-phi3-mini-4k-g52xlarge`).

