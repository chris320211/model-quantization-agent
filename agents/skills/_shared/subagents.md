# Subagents

Each stage is a **separate subagent**. The parent (`quant`) only collects the
three inputs, launches one subagent at a time, and passes the previous return
value. It does not search, clone, adapt, or run GPU jobs itself.

## Rules

- One stage per subagent. Fresh context.
- Pass only paths and the three user inputs. Never paste tokens, `.env`, or credential files.
  Secrets: `agents/AGENTS.md` — never read or print `.env`.
- `quant-setup`, `quant-publish`, and `quant-catalog` stay in the **parent**.
  Do not spawn a subagent for secrets, Hub upload, or the library row.
  `quant-setup` asks for a local `.env` once (copy `.env.example`); load it
  every shell. Never read `.env`. Method comparison is `compare/` (written by
  `quant-catalog` / `compare.py`), not a Hub leaderboard and not a subagent.
- Wait until a subagent returns before starting the next **stage**.
- **Port is the exception:** the port coordinator launches up to three *named*
  strategy subagents at once (author + validate only). The parent still runs
  **one** GPU job, using the winner, then the next ranked overlay if verify fails.
- If a subagent fails, stop or retry **that** stage. Do not silently skip ahead.
- Scripts: `"$PY" agents/skills/_shared/scripts/<name>.py` with
  `PY=$(command -v python || command -v python3)`.

## How to launch

Use the host’s subagent tool (Cursor `Task`, Claude/Codex agent). Point it at the
stage skill and paste the prompt template from that skill.

## Example (gather)

User: `AWQ`, `Qwen/Qwen2.5-0.5B-Instruct`, `g5.xlarge`

Parent launches a subagent with:

```text
Read agents/skills/quant-gather/SKILL.md and follow it.

Inputs:
- method_name: AWQ
- model_name: Qwen/Qwen2.5-0.5B-Instruct
- gpu_instance: g5.xlarge

Do only gather. Return the path to out/requests/<slug>.json.
```

When that returns `out/requests/awq-qwen25-05b-g5xlarge.json`, the parent launches
**port**, then run, then verify, then **always** benchmark, then the **retry
loop** (diagnose → `recommended_action`) for **that** slug. Same loop if the
user named a different method, model, or GPU instance. Do not spawn a retry
skill. Do not paste WikiText-2 numbers into port workers; pass `issue:` codes
and `jobs/<job_id>/diagnose.json`.

## Example (port — three strategies, one GPU winner)

Parent launches **one coordinator**:

```text
Read agents/skills/quant-port/SKILL.md and follow it.
Request JSON: out/requests/awq-qwen25-05b-g5xlarge.json
You are the port coordinator. Spawn workers for dispatch, llama_alias, and
adapter_only. Validate only. Return a ranked winner. Do not launch a GPU job.
```

Each worker:

```text
Read agents/skills/quant-port/SKILL.md and follow it.
Request JSON: out/requests/awq-qwen25-05b-g5xlarge.json
strategy: llama_alias
Do only this strategy. Write out/overlays/<slug>/llama_alias/. Return validate ok/fail.
```

Run **one** `quant-run` on the winner. If verify fails, diagnose first.
Next ranked overlay is only for unpacked `loader_arch`. Packed CUDA dtype
asserts stay on the packed path. Do not start three quantize jobs at once.

## Example (benchmark — always after verify)

```text
Read agents/skills/quant-benchmark/SKILL.md and follow it.
Request JSON: out/requests/awq-qwen25-05b-g5xlarge.json
job_id: <job_id>
Do only benchmark. Return quantized vs fp16 WikiText-2 metrics.
```

## Example (diagnose — after a failed benefit check)

```text
Read agents/skills/quant-diagnose/SKILL.md and follow it.
Request JSON: out/requests/awq-qwen25-05b-g5xlarge.json
job_id: <job_id>
Do only diagnose. Return root_cause, issue_codes, action, and next overlay if any.
```

## Example (parent retry — typed diagnose feedback)

```text
Read agents/skills/quant-run/SKILL.md and follow it.
Request JSON: out/requests/awq-qwen25-05b-g5xlarge.json
Overlay: <next_overlay_dir from diagnose.json>
Script: <next_script>
parent_job_id: <failed_or_unimproved_job_id>
issue: transform_or_runtime_dropped_on_save
diagnose_json: jobs/<job_id>/diagnose.json
Do only run. Return job_id.
```

Then verify + (if passed) benchmark that new `job_id`. If verify failed,
diagnose without benchmark. **Always** diagnose after a passed benchmark
(including success) and switch only on `recommended_action`. Stop on `none`.
Do not inspect overlays for SDPA or fused-kernel names. Packed-path
`kernel` / `author_fix` may run when `retry.remaining` is 0; the helper
owns the cap. `kernel` is efficiency after quality is OK, for any method.
A packed kernel that failed verify with `cuda_kernel_dtype_mismatch` is still
a packed fix, not a stop.
