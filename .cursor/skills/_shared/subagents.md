# Subagents

Each stage is a **separate subagent**. The parent (`quant`) only collects the
three inputs, launches one subagent at a time, and passes the previous return
value. It does not search, clone, adapt, or run GPU jobs itself.

## Rules

- One stage per subagent. Fresh context.
- Pass only paths and the three user inputs. Never paste tokens, `.env`, or credential files.
- `quant-setup` stays in the **parent**. Do not spawn a subagent for secrets.
- Wait until a subagent returns before starting the next **stage**.
- **Port is the exception:** the port coordinator launches up to three *named*
  strategy subagents at once (author + validate only). The parent still runs
  **one** GPU job, using the winner, then the next ranked overlay if verify fails.
- If a subagent fails, stop or retry **that** stage. Do not silently skip ahead.
- Scripts: `python .agents/skills/_shared/scripts/<name>.py`. Same path from
  `.claude` or `.cursor` skill trees.

## How to launch

Use the host’s subagent tool (Cursor `Task`, Claude/Codex agent). Point it at the
stage skill and paste the prompt template from that skill.

## Example (gather)

User: `AWQ`, `Qwen/Qwen2.5-0.5B-Instruct`, `g5.xlarge`

Parent launches a subagent with:

```text
Read .agents/skills/quant-gather/SKILL.md and follow it.

Inputs:
- method_name: AWQ
- model_name: Qwen/Qwen2.5-0.5B-Instruct
- gpu_instance: g5.xlarge

Do only gather. Return the path to out/requests/<slug>.json.
```

When that returns `out/requests/awq-qwen25-05b-g5xlarge.json`, the parent launches
**port**, then run, then verify.

## Example (port — three strategies, one GPU winner)

Parent launches **one coordinator**:

```text
Read .agents/skills/quant-port/SKILL.md and follow it.
Request JSON: out/requests/awq-qwen25-05b-g5xlarge.json
You are the port coordinator. Spawn workers for dispatch, llama_alias, and
adapter_only. Validate only. Return a ranked winner. Do not launch a GPU job.
```

Each worker:

```text
Read .agents/skills/quant-port/SKILL.md and follow it.
Request JSON: out/requests/awq-qwen25-05b-g5xlarge.json
strategy: llama_alias
Do only this strategy. Write out/overlays/<slug>/llama_alias/. Return validate ok/fail.
```

Run **one** `quant-run` on the winner. If verify fails with an arch/loader bug,
run the second-ranked overlay. Do not start three quantize jobs at once.
