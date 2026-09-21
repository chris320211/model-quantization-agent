---
name: quant-sync
description: >-
  After a beneficial quantization run, after recording an unsuccessful
  attempt, or when asked to update GitHub and Hugging Face, push the library
  to this GitHub remote and refresh the one Hub collection. Use in the parent
  after quant-catalog. Not a subagent. Never commit checkpoints or .env.
disable-model-invocation: true
---

# Quant Sync

Parent session only. Skip unless verify passed, `quality_ok`, and the run beat
fp16 **VRAM or tok/s**, an unsuccessful attempt was just recorded, **or** the
user asked to update GitHub / Hugging Face. Need the catalog or attempts row
already written (`quant-catalog`) when syncing a new run. The Hub collection
still lists only `quality_ok` repos.
Do not re-quantize. Never read `.env`. Never `git add -f` gitignored paths.

GitHub holds the **library table**. Hugging Face holds **weights** (already
uploaded by `quant-publish`) plus **one collection** of those repos.

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

Every new shell: if `HF_TOKEN` is unset and `.env` exists,
`source agents/skills/_shared/load_env.sh .env`. Never print values.

## Do

```bash
$S/sync_remotes.py --push --allow-unsafe-host-execution
```

`sync_remotes.py` rebuilds `library/`, refreshes the Hub collection when
`HF_TOKEN` is loaded, then commits allowlisted paths. If the token is unset,
it still pushes GitHub. This is the **only** stage that refreshes the Hub
collection. It commits only `library/`, `agents/`, `tests/`, `README.md`,
and `.github/workflows/`. It never stages `quantized/`, `jobs/`, `out/`,
`.venvs/`, `.cache/`, or `.env`. `git push` uses the credential helper,
or `GITHUB_TOKEN` via `git_askpass.sh` when that key is loaded (not a remote
URL). Do not run `git config`.

Third parties without push access still PR
`library/contributions/<model>__<gpu>__<method>.json`.

Docs: `library/LIBRARY.md`, `library/ATTEMPTS.md`.

## Return

```text
status: synced
github: <origin commit url or skipped>
hf_collection: <hf_collection_url from library/library.json, or skipped>
commit: <sha or none>
```
