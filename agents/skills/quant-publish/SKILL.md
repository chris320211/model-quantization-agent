---
name: quant-publish
description: >-
  After a passed verify and WikiText-2 benchmark with quality_ok, stage the
  saved quantized artifact plus those metrics for Hugging Face Hub so others
  can snapshot_download them. Hub is the artifact store, not the method
  comparison board (`library/` ranks methods on the same model × GPU).
  Use in the parent session only (HF token).
---

# Quant Publish

Run this in the **parent** session only, like `quant-setup`. Never spawn a
subagent (the Hub token must not enter a child prompt). Need `job_id` and
the request JSON. Skip unless verify passed and `comparison.quality_ok` is
true. Do not re-quantize.

This stage is method-agnostic: upload whatever artifact that job saved, with
the WikiText-2 numbers from `benchmark.json`. Not a method catalog. One Hub
repo is one method's weights + card. Users rank methods in `library/`
(keyed by `model_id` × `gpu_instance`), not by browsing Hub model cards.

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

## Do

Stage without a token:

```bash
$S/publish.py --job-id <job_id> --request out/requests/<slug>.json
```

Writes `out/hub/<slug>/` (model card, `metrics.json`, tokenizer/config,
adapter/overlay when present). Large packed weights are referenced from
`quantized/<slug>` so they are not duplicated.

Upload (needs `HF_TOKEN` already in the parent environment from `.env`;
never paste the value into chat):

```bash
$S/publish.py --job-id <job_id> --request out/requests/<slug>.json \
  --repo-id <org>/<name> --upload --allow-unsafe-host-execution
```

`--repo-id` is `org/name` on the Hub. Default suggestion is
`<hf-username>/<slug>` (their account is fine). Create a **public model** repo.
Do not upload Hub fp16, job logs, or `.env`. After upload, this helper records
the row in `library/` and adds the repo to the **one** Hub collection
(`library.py --sync-hf-collection`). Other people still PR a contribution JSON
so the library table is reviewable.

If the token is missing, leave the staged bundle and tell the user to set
`HF_TOKEN` via `quant-setup`, then re-run with `--upload`.

## Return

```text
status: staged | uploaded
stage_dir: out/hub/<slug>/
hub_url: https://huggingface.co/<org>/<name>   # only if uploaded
metrics: out/hub/<slug>/metrics.json
library: library/catalog.json   # filter model / method / instance, then fetch hub_repo_id
```

After a successful `--upload`, run **quant-catalog** then **quant-sync** in
this same parent session so the library row has model, GPU, method, paper,
method repo, and Hub links, then GitHub and the Hub collection update.
`publish.py` also writes `hub_url` and
`library/contributions/<model>__<gpu>__<method>.json`. Docs: `library/README.md`.
