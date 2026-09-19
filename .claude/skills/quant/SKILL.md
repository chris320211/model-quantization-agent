---
name: quant
description: >-
  Dispatcher for autonomous quantization porting. Use when the user names a
  quantization method, a Hugging Face model, and a GPU instance (for example
  AWQ, Qwen2.5 0.5B, g5.xlarge). Launches one subagent per stage.
---

# Quant

Read [_shared/pipeline_contract.md](../_shared/pipeline_contract.md) and
[_shared/subagents.md](../_shared/subagents.md) first.

You are the **parent**. Collect `method_name`, `model_name`, `gpu_instance`.
If a gated model needs HF auth and it is missing, follow `quant-setup` **in this
session**. Then launch **one subagent per stage**, in order. Do not do stage work
yourself. Helpers: `python .agents/skills/_shared/scripts/<name>.py`.

## Order

1. Gather → returns `out/requests/<slug>.json`
2. Port → returns ranked overlay dirs under `out/overlays/<slug>/<strategy>/`
3. Run → returns `job_id`
4. Verify → passed / failed (must be the saved artifact, not Hub fp16)
5. Baseline compare (parent may relaunch verify with `--baseline`)
6. Kernel subagent **only if** quantized is not better, or the user asked to go faster
7. Tell the user where weights (`quantized/<slug>`) and job metrics live

## Example

User: “port AWQ to Qwen/Qwen2.5-0.5B-Instruct on g5.xlarge”

Launch gather (see [_shared/subagents.md](../_shared/subagents.md)).

Then launch **port** as a coordinator (it fans out up to three strategy
subagents). Use the **winner** overlay for run. If verify fails on an arch/loader
bug, try the next ranked overlay — do not launch three GPU jobs at once.

Then run:

```text
Read .agents/skills/quant-run/SKILL.md and follow it.
Request JSON: out/requests/awq-qwen25-05b-g5xlarge.json
Overlay: <winner_overlay_dir>
Script: <winner_script>
Do only run. Return job_id.
```

Then verify with that `job_id` and the same request JSON. Stop on verify failure
unless it is a loader bug (then the next ranked port overlay). Kernel subagent
only after a failed benefit check.
