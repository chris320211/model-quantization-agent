---
name: quant-setup
description: Securely write or update the model-quantization-agent .env file (Anthropic key, HuggingFace token, etc.). Use when the user asks to "set up env", "configure HF token", "store anthropic key", "add my huggingface token", "set up credentials", or reports an HF 401/403 / missing-API-key error. Strongly recommends the `quant-agent setup` CLI (hidden input via getpass) and only writes from chat as a fallback when the user explicitly insists.
---

# Quant-setup — secure credential setup for the quantization pipeline

This skill writes `/home/ubuntu/model-quantization-agent/.env` (mode 600, gitignored). It is the single source of truth that the other three skills (`quant`, `quant-execute`, `quant-tune`) read via `_shared/load_env.sh`.

## When to invoke

- User asks to "set up env / credentials", "configure HF token", "add my anthropic key", "store huggingface token".
- A sibling skill bailed because `.env` was missing or `HF_TOKEN` was not set.
- A run failed with HF `401`/`403` (gated repo) or `ANTHROPIC_API_KEY` errors.

## Decision: which path?

There are two ways to set credentials. **Always offer the CLI path first.**

### Path A — `quant-agent setup` CLI (recommended)

The Python CLI (`src/quant_agent/setup_cmd.py`) prompts for each key with `getpass`, so:

- The value never appears on screen.
- The value never lands in shell history.
- The value never lands in this Claude Code conversation.
- The file is written with mode `0600` automatically.

Tell the user, verbatim:

```
For the most secure setup, run this in your terminal (not in chat):

    quant-agent setup

It uses hidden input (getpass) so your tokens are never displayed or logged.
Re-run with `--force` to replace an existing .env, or `--no-optional` to skip
GITHUB_TOKEN and HUGGINGFACE_HUB_TOKEN prompts.
```

Then ask: **"Run that and let me know when it's done — or, if you cannot use the terminal right now, paste the value here and I'll write only that single key (with a security caveat)."** Wait for their answer.

### Path B — write a single key from chat (fallback only)

Use this only when the user explicitly insists, after you have offered Path A. Each value pasted into chat is visible in the conversation transcript. Never write a token to chat or stdout.

For each key the user provides:

1. **Confirm the key name and value** (mention the key NAME only in your reply, never the value).
2. **Warn explicitly** (one line): *"This value is now in our chat history. Consider rotating the token after this session, or using `quant-agent setup` next time for hidden input."*
3. **Update the file** with `Edit` if the key exists, or append with a Python one-liner if it does not. Do **not** use `Write` on `.env` — overwriting the whole file risks dropping other keys.

   To replace an existing key (preserves all other lines):
   ```
   Edit:
     file_path: /home/ubuntu/model-quantization-agent/.env
     old_string: HUGGINGFACE_HUB_TOKEN=<old-value-from-Read>
     new_string: HUGGINGFACE_HUB_TOKEN=<new-value>
   ```

   To append a new key, use Bash with a heredoc and immediately re-tighten perms:
   ```
   python3 -c "
   import os, pathlib
   p = pathlib.Path('/home/ubuntu/model-quantization-agent/.env')
   p.write_text(p.read_text() + 'HUGGINGFACE_HUB_TOKEN=<new-value>\n')
   os.chmod(p, 0o600)
   "
   ```

4. **Re-tighten permissions** even after `Edit` (defensive):
   ```
   chmod 600 /home/ubuntu/model-quantization-agent/.env
   ```

5. **Validate** the value worked (see Validation section below).

## Validation per key type

After writing, run a tiny live check so a typo surfaces now, not at first use.

### `HUGGINGFACE_HUB_TOKEN` / `HF_TOKEN`

```
source /home/ubuntu/model-quantization-agent/.claude/skills/_shared/load_env.sh
curl -sS -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2 | head -c 400
```

- `200` with a JSON body containing `name` / `email` → **valid**.
- `401`/`403` → token is bad or missing scope. Ask the user to regenerate at https://huggingface.co/settings/tokens with read scope.

### `ANTHROPIC_API_KEY`

The Python CLI's `_validate_anthropic` does a 1-token live call. Mirror it:

```
source /home/ubuntu/model-quantization-agent/.claude/skills/_shared/load_env.sh
python3 -c "
from anthropic import Anthropic
import os
Anthropic(api_key=os.environ['ANTHROPIC_API_KEY']).messages.create(
    model=os.environ.get('QUANT_AGENT_MODEL', 'claude-sonnet-4-6'),
    max_tokens=1, messages=[{'role':'user','content':'hi'}],
)
print('ok')
"
```

- Stdout `ok` → valid.
- Any exception → tell the user the model name and exception class; suggest re-running `quant-agent setup` with `--no-validate` skipped.

### `GITHUB_TOKEN` (optional)

```
source .../_shared/load_env.sh
curl -sS -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/user | head -c 200
```

`200` with `login` field → valid. `401` → bad token.

## Permissions audit

Whenever this skill runs, verify:

```
stat -c %a /home/ubuntu/model-quantization-agent/.env
```

Must print `600`. If not:

```
chmod 600 /home/ubuntu/model-quantization-agent/.env
```

## What this skill never does

- **Never echo a secret value back in chat.** Confirm by key name only.
- **Never `cat` or `Read` the .env and quote its contents back to the user.** Only emit key NAMES.
- **Never check the .env into git.** It is in `.gitignore`; if you accidentally `git add` it, run `git rm --cached .env` and re-confirm.
- **Never write tokens to stdout, log files, or skill output.**
- **Never paste a token into a Bash command line where it would land in shell history.** Always go through the .env file or the CLI's getpass prompt.

## Reference: keys this pipeline reads

| Key | Required? | Used by | Notes |
|-----|-----------|---------|-------|
| `ANTHROPIC_API_KEY` | yes (Python CLI) | quant-agent CLI | Skills call Claude through the Claude Code session, so skill-only flows do not strictly need this. The Python CLI does. |
| `HUGGINGFACE_HUB_TOKEN` | for gated models | all skills + CLI | Loader aliases this to `HF_TOKEN` automatically. |
| `GITHUB_TOKEN` | optional | all | Lifts GitHub API rate limits during README/example fetching. |
| `QUANT_AGENT_MODEL` | optional | Python CLI | Model id override; defaults to `claude-sonnet-4-6`. |

## Cross-skill contract

Sibling skills (`quant`, `quant-execute`, `quant-tune`) load credentials by sourcing the shared helper at the top of every Bash subprocess that touches HF or runs a generated script:

```
source /home/ubuntu/model-quantization-agent/.claude/skills/_shared/load_env.sh
```

The helper:

- Refuses to source a `.env` whose mode is not `600` (auto-tightens).
- Exports every `KEY=value` line.
- Aliases `HUGGINGFACE_HUB_TOKEN` ↔ `HF_TOKEN` so libraries reading either name see the same value.
- Echoes only the loaded key NAMES to stderr, never values.

If a skill encounters a missing key at runtime, it should bail and direct the user back here — not silently retry.
