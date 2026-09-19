---
name: quant-setup
description: >-
  Guide Hugging Face and optional GitHub credential setup for gated models and
  clone rate limits. Use in the parent session only. Never accept, reveal, or
  manipulate secret values through chat.
---

# Quant Setup

Run this in the **parent** session only. Never spawn a subagent for setup and
never paste token values into a subagent prompt.

- **Skills path:** Codex/Claude/Cursor run these skills in the active session.
  There is no `quant-agent` CLI.
- **Model access:** gated Hugging Face models need `HF_TOKEN` or
  `HUGGINGFACE_HUB_TOKEN` in the parent process.
- **Repository access:** `GITHUB_TOKEN` is optional and only raises API rate limits.

Never extract ChatGPT cookies or OAuth tokens, copy credentials from browser
storage, or put tokens in generated scripts, overlays, or job metadata.

## Session-only token

Tell the user to enter it invisibly before starting the agent:

```bash
read -rsp "HuggingFace token: " HF_TOKEN
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
echo
```

Do not echo the value. Environment-only setup must be repeated in a new shell.

## Persist to `.env` (user types this; you do not)

The user may write a mode-600 `.env` themselves. You never create, read, or
edit that file. Optional loader:

```bash
source .agents/skills/_shared/load_env.sh .env
```

The loader parses an allowlist of keys; it does not `source` the file as shell.

## Diagnose by credential name only

- HF `401/403`: confirm license acceptance and a read-scoped HF token; do not
  relaunch until access changes.
- GitHub `401`: remove or replace the optional token; public clones still use
  HTTPS GitHub URLs.
- Codex/Claude/Cursor subscription limit: wait for reset or switch backend.

## Security invariants

- Never read, print, edit, stage, or commit credential files with agent tools.
- Never place a token in a command argument, generated script, job metadata,
  logs, overlay, or request JSON.
- Installers, repository code, dry-import probes, and measurements receive no
  cloud credentials. Quantization and verify receive only HF authentication
  when needed.
- Confirm setup only by credential names and successful authenticated behavior.
