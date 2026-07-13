---
name: quant-setup
description: Guide secure authentication and credential setup for subscription-backed Codex skills, the optional OpenAI API-backed Python pipeline, HuggingFace gated models, and GitHub rate limits. Use when a user needs to sign Codex in, configure tokens, choose subscription versus API authentication, or resolve 401/403/authentication failures. Never accept, reveal, or manipulate secret values through chat.
---

# Quant Setup

Separate the reasoning credential from model/repository access:

- **Codex subscription mode:** sign the Codex app/CLI into ChatGPT. The quant skills
  then use the active subscription and do not require `OPENAI_API_KEY`.
- **Python API mode:** `quant-agent ask` uses the OpenAI API backend and requires
  `OPENAI_API_KEY`.
- **Model access:** gated HuggingFace models require `HF_TOKEN` or
  `HUGGINGFACE_HUB_TOKEN` in the parent process.
- **Repository access:** `GITHUB_TOKEN` is optional and only raises API rate limits.

Never extract ChatGPT cookies or OAuth tokens, copy credentials from browser storage,
or substitute ChatGPT auth into `ChatOpenAI`.

## Subscription-backed Codex

Use the official Codex browser/device-code login on the trusted machine. Verify the
active account through Codex's status/account UI, not by reading local auth files.
On an EC2 worker, prefer device-code login. Subscription limits still apply; this
avoids Platform API billing rather than providing unlimited inference.

## API-backed Python pipeline

When the user deliberately chooses API mode, direct them to run this themselves in
an interactive terminal:

```text
quant-agent setup
```

The command uses hidden input, validates OpenAI model access unless disabled, prompts
for optional HF/GitHub tokens, refuses symlink replacement, and atomically writes a
mode-600 credential file. Never run it on the user's behalf, accept secrets in chat,
or inspect the resulting file.

## No credential file

For a session-only HF token, tell the user to enter it invisibly before starting
Codex or the Python process:

```bash
read -rsp "HuggingFace token: " HF_TOKEN
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
echo
```

Do not echo the value. Environment-only setup must be repeated in a new shell/session.

## Diagnose by credential name only

- HF `401/403`: confirm license acceptance and a read-scoped HF token; do not relaunch
  until access changes.
- OpenAI authentication/model-access error: applies only to Python API mode; rerun the
  interactive setup or use subscription-backed skills instead.
- GitHub `401`: remove or replace the optional token; public catalog clones still use
  catalog-pinned HTTPS URLs.
- Codex subscription limit: wait for reset, use eligible ChatGPT credits, select a
  lighter Codex model, or explicitly switch to another backend.

## Security invariants

- Never read, print, edit, stage, or commit credential files with agent tools.
- Never place a token in a command argument, generated script, job metadata, logs,
  Adapt trace, overlay, or reproducibility manifest.
- Installers, repository code, dry-import probes, and measurements receive no cloud
  credentials. Quantization receives only HF authentication when needed.
- Confirm setup only by credential names and successful authenticated behavior.
