---
name: quant-sync
description: >-
  After a beneficial quantization run, or when asked to update GitHub and
  Hugging Face, push the compare library to this GitHub remote and refresh the
  one Hub collection. Use in the parent after quant-catalog. Not a subagent.
  Never commit checkpoints or .env.
---

# Quant Sync

Parent session only. Skip unless verify passed, `quality_ok`, and the run beat
fp16 **VRAM or tok/s**, **or** the user asked to update GitHub / Hugging Face.
Need the catalog row already written (`quant-catalog`) when syncing a new run.
Do not re-quantize. Never read `.env`. Never `git add -f` gitignored paths.

GitHub holds the **library table**. Hugging Face holds **weights** (already
uploaded by `quant-publish`) plus **one collection** of those repos.

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

Every new shell: if `HF_TOKEN` is unset and `.env` exists,
`source agents/skills/_shared/load_env.sh .env`. Never print values.

## Do

```bash
$S/compare.py --rebuild
$S/compare.py --sync-hf-collection
$S/sync_remotes.py --push --allow-unsafe-host-execution
```

`--sync-hf-collection` needs `HF_TOKEN` already loaded. If unset, skip that
step and still push GitHub. `sync_remotes.py` commits only `compare/`,
`agents/`, `tests/`, `README.md`, and `.github/workflows/`. It never stages
`quantized/`, `jobs/`, `out/`, `.venvs/`, `.cache/`, or `.env`. `git push`
uses the credential helper or `GITHUB_TOKEN` via `git_askpass.sh` (not a
remote URL). Do not run `git config`.

Third parties without push access still PR
`compare/contributions/<model>__<gpu>__<method>.json`.

Docs: `compare/LIBRARY.md`.

## Return

```text
status: synced
github: <origin commit url or skipped>
hf_collection: <compare/library.json url or skipped>
commit: <sha or none>
```
