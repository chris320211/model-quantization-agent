---
name: quant-setup
description: >-
  On first load of this repo, ask the user to create a mode-600 `.env` with
  their secrets (Hugging Face, optional GitHub). Load it every shell. Use in
  the parent session only. Never accept, reveal, or manipulate secret values
  through chat.
---

# Quant Setup

Run this in the **parent** session only. Never spawn a subagent for setup and
never paste token values into a subagent prompt. Same secret rules as
`agents/AGENTS.md`: never read, print, or edit `.env`.

Default credential store is a **local `.env`**. Create it **once per machine**
(first clone / first open). Load it **every new shell**. Do not ask the user
to recreate `.env` if it already exists.

- **Skills path:** Codex/Claude/Cursor run these skills in the active session.
  There is no `quant-agent` CLI.
- **Interpreter:** `PY=$(command -v python || command -v python3)`. Deep Learning
  AMI / Ubuntu GPU hosts often have `python3` only. Install helper deps once:
  `"$PY" -m pip install -c constraints.txt -e '.[dev]'`.
- **Model access:** gated Hugging Face models and Hub publish need `HF_TOKEN`
  in `.env`. The loader also exports `HUGGINGFACE_HUB_TOKEN` from that one value.
- **Repository access:** `GITHUB_TOKEN` is optional (API rate limits). `git push`
  uses Git's credential helper, not `.env`.

Never extract ChatGPT cookies or OAuth tokens, copy credentials from browser
storage, or put tokens in generated scripts, overlays, or job metadata.

## First load (`.env` missing)

You may `test -f .env` (existence only). Never read, print, or edit the file.

If `.env` is missing, ask the user to create it **themselves** from
`.env.example`. Point at keys by **name** only. Do not fill values. Do not
create the file with agent tools.

Tell the user to run:

```bash
cp .env.example .env
chmod 600 .env
```

Then they edit `.env` in their editor (not chat) and paste:

- `HF_TOKEN` — Hugging Face write token (gated models + `quant-publish`)
- `GITHUB_TOKEN` — optional; GitHub API rate limits for clones

Keys live in `.env.example`. `.env` is gitignored.

After they say it exists, they load it (every shell, including this one):

```bash
source agents/skills/_shared/load_env.sh .env
```

The loader parses an allowlist of keys; it does not `source` the file as shell.
Confirm only by key **names** the loader printed and by later 200/whoami
behavior. Never print values.

If `.env` already exists, do **not** ask them to recreate it. Ask them to
`source` the loader in this shell if `HF_TOKEN` is unset.

## Session-only fallback

Only if the user refuses `.env`. Must be repeated in a new shell:

```bash
read -rsp "HuggingFace token: " HF_TOKEN
export HF_TOKEN
echo
```

Do not echo the value.

## Diagnose by credential name only

- Missing `.env`: first-load steps above. Do not invent a token.
- HF `401/403`: confirm license acceptance and a write-scoped HF token for
  publish (read is enough for gated download); do not relaunch until access
  changes.
- GitHub `401`: remove or replace the optional token; public clones still use
  HTTPS GitHub URLs. `git push` 403 is Git credential-helper scope, not `.env`.
- Codex/Claude/Cursor subscription limit: wait for reset or switch backend.

## Security invariants

- Never read, print, edit, stage, or commit credential files with agent tools.
- Never place a token in a command argument, generated script, job metadata,
  logs, overlay, or request JSON.
- Installers, repository code, dry-import probes, and measurements receive no
  cloud credentials. Quantization and verify receive only HF authentication
  when needed.
- Confirm setup only by credential names and successful authenticated behavior.
