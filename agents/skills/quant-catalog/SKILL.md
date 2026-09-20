---
name: quant-catalog
description: >-
  After a successful beneficial quantization run (quality_ok and better VRAM
  or tok/s than fp16), record one library row: model, GPU instance, method,
  paper, method GitHub repo, and Hugging Face weights. Use in the parent after
  quant-publish. Not a subagent.
---

# Quant Catalog

Parent session only. Skip unless verify passed, `quality_ok`, and the run beat
fp16 **VRAM or tok/s**. Need `job_id`, the request JSON, and the Hub URL from
`quant-publish`. Do not re-quantize. Never read `.env`.

Standard row: **model**, **GPU instance**, **method**, **paper**, **method repo**,
**Hugging Face**. Same shape every time.

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

## Do

```bash
$S/compare.py --catalog --job-id <job_id> \
  --request out/requests/<slug>.json \
  --hub-url https://huggingface.co/<you>/<slug>
$S/compare.py --rebuild
$S/compare.py --sync-hf-collection
```

`--sync-hf-collection` needs `HF_TOKEN` already loaded. If unset, skip sync.

Do not edit `catalog.json` by hand. Do not commit weights. Then run
**quant-sync** in this parent session (GitHub push + Hub collection). Third
parties without push access PR
`compare/contributions/<model>__<gpu>__<method>.json`.

Docs: `compare/LIBRARY.md`.

## Return

```text
status: cataloged
model_id: <org/model>
gpu_instance: <instance>
method_name: <method>
paper_url: https://arxiv.org/abs/<id>
repo_url: https://github.com/<owner>/<repo>
hub_url: https://huggingface.co/<you>/<slug>
contribution: compare/contributions/<file>.json
```
