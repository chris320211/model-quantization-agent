# Library

One table of **beneficial** quantized runs. Weights stay on each author’s
public Hugging Face repo.

A row is always:

| Field | Meaning |
| --- | --- |
| model | Hugging Face `org/model` |
| GPU instance | e.g. `g5.2xlarge` |
| method | Quantization method name |
| paper | arXiv (from gather `arxiv_id`) |
| repo | Method GitHub repo used for the port |
| Hugging Face | Public weights from `quant-publish` |

Plus WikiText-2 metrics vs fp16 (`quality_ok`, PPL ratio, tok/s, VRAM).

## When to add a row

After verify passed, `quality_ok`, and tok/s **or** VRAM beat fp16. Then:

1. `quant-publish --upload --repo-id <you>/<slug>`
2. Follow `agents/skills/quant-catalog/SKILL.md`
3. PR only `compare/contributions/<model>__<gpu>__<method>.json`

Do not commit checkpoints. Do not hand-edit `catalog.json`.

```bash
PY=$(command -v python || command -v python3)
S="$PY agents/skills/_shared/scripts"
$S/compare.py --model-id <org/model> --gpu-instance <instance>
$S/compare.py --model-id <org/model> --method <name> --gpu-instance <instance> --fetch
```

Hub collection URL: `compare/library.json`.
