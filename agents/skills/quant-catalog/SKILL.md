---
name: quant-catalog
description: >-
  After a successful beneficial quantization run (quality_ok and better VRAM
  or tok/s than fp16) and quant-publish, record one library row: model, GPU
  instance, method, paper, method GitHub repo, and published weights. Use in
  the parent after quant-publish. Not a subagent. Does not refresh the Hub
  collection (quant-sync does).
disable-model-invocation: true
---

# Quant Catalog

Parent session only. Skip unless verify passed, `quality_ok`, and the run beat
fp16 **VRAM or tok/s**. Need `job_id`, the request JSON, and the Hub URL from
`quant-publish`. Do not re-quantize. Never read `.env`.

Standard row: **model**, **GPU instance**, **method**, **paper**, **method repo**,
**weights**. Same shape every time.

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

## Do

```bash
$S/library.py --catalog --job-id <job_id> \
  --request out/requests/<slug>.json \
  --hub-url https://huggingface.co/<you>/<slug>
$S/library.py --rebuild
```

Do not call `--sync-hf-collection` here. **quant-sync** refreshes the Hub
collection and pushes GitHub.

Do not edit `catalog.json` by hand. Do not commit weights. Then run
**quant-sync** in this parent session. Third parties without push access PR
`library/contributions/<model>__<gpu>__<method>.json`.

Docs: `library/LIBRARY.md`.

## Return

```text
status: cataloged
model_id: <org/model>
gpu_instance: <instance>
method_name: <method>
paper_url: https://arxiv.org/abs/<id>
repo_url: https://github.com/<owner>/<repo>
hub_url: https://huggingface.co/<you>/<slug>
contribution: library/contributions/<file>.json
```
