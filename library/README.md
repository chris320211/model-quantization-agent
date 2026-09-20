# Library

**One library.** Weights can live on any public Hugging Face account.
This table is how you rank them. Filter `catalog.json` by `model_id`,
`method_name`, `gpu_instance`. Fetch with `huggingface-cli download <hub_repo_id>`.

Hub collection (same rows): https://huggingface.co/collections/chris320211/quant-agent-library-6aaf22fcafd69b39eabc9230

```bash
PY=$(command -v python || command -v python3)
S="$PY agents/skills/_shared/scripts"
$S/library.py --model-id <org/model> --gpu-instance <instance>
$S/library.py --model-id <org/model> --method <name> --gpu-instance <instance> --fetch
```

Rank: `quality_ok`, then tokens/s, then lower VRAM, packed/realquant, then lower PPL ratio.

Each row is **model × GPU instance × method**, with links to the paper,
the method GitHub repo, and the Hugging Face weights.

Add a row after verify, `quality_ok`, and tok/s **or** VRAM beat fp16:
`quant-publish`, then `agents/skills/quant-catalog/SKILL.md`, then
`agents/skills/quant-sync/SKILL.md`. Do not commit checkpoints.
Do not hand-edit `catalog.json`. Without push access, PR
`library/contributions/<model>__<gpu>__<method>.json`.

## ibm-granite/granite-3.3-2b-instruct on g5.2xlarge
GPU: NVIDIA A10G

| Model | GPU | Method | Paper | Repo | Hugging Face | PPL ratio | tok/s | VRAM GB |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| ibm-granite/granite-3.3-2b-instruct | g5.2xlarge | SmoothQuant | [paper](https://arxiv.org/abs/2211.10438) | [repo](https://github.com/mit-han-lab/smoothquant) | [hub](https://huggingface.co/chris320211/smoothquant-granite33-2b-g52xlarge) | 1.0060 | 9501.7 | 3.377 |

**Best:** SmoothQuant (`smoothquant-granite33-2b-g52xlarge`).

## microsoft/Phi-3-mini-4k-instruct on g5.2xlarge
GPU: NVIDIA A10G

| Model | GPU | Method | Paper | Repo | Hugging Face | PPL ratio | tok/s | VRAM GB |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| microsoft/Phi-3-mini-4k-instruct | g5.2xlarge | FlatQuant | [paper](https://arxiv.org/abs/2410.09426) | [repo](https://github.com/ruikangliu/FlatQuant) | [hub](https://huggingface.co/chris320211/flatquant-phi3-mini-4k-g52xlarge) | 1.1676 | 5105.9 | 3.355 |

**Best:** FlatQuant (`flatquant-phi3-mini-4k-g52xlarge`).

