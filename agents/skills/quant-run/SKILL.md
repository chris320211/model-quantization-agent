---
name: quant-run
description: >-
  Launch one validated quantization script on this GPU instance and monitor it
  to completion. Use only as a quant-run subagent launched by the parent quant
  skill after quant-port (or after diagnose/kernel). Does not patch overlays,
  change methods, or rewrite kernels.
disable-model-invocation: true
---

# Quant Run

You are a **subagent**. Do only run. Need request JSON + script path + overlay dir.
Do not ask the parent mid-stage. If `HF_TOKEN` is unset and `.env` exists,
`source agents/skills/_shared/load_env.sh .env` (never print values).

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

## Do

1. Launch **once** (isolated box; flag required):

   ```bash
   $S/launch.py <script.py> --request out/requests/<slug>.json \
     --overlay-dir <overlay_dir> --allow-unsafe-host-execution
   ```

   Parent retry (optional flags from diagnose, not a quality loop of your own):

   ```bash
   $S/launch.py <script.py> --request out/requests/<slug>.json \
     --overlay-dir <next_overlay_dir> --parent-job-id <job_id> \
     --attempt <n> --fix-note <issue_code> \
     --allow-unsafe-host-execution
   ```

   Prints job metadata JSON including `job_id`.

2. Monitor:

   ```bash
   $S/jobs.py status <job_id>
   $S/jobs.py logs <job_id> -n 200
   ```

3. On failure: return `failed` and the `job_id`. Tail stderr in the return
   so the parent can pass it to diagnose. Do **not** invent an overlay patch.
   Do not call `overlay.py write`. Do not relaunch. Process failures are
   parent diagnose (`author_fix` / `retry_ranked_overlay`) so they count
   against `retry_gpu_jobs_max` and `tried_overlays`. Do not retry
   gated-model auth, OOM at the same config, disk full, or wrong GPU.

4. Do not rewrite kernels here. Do not start a second GPU job while one is
   running. Do not interpret WikiText-2; that is benchmark + parent diagnose.
   The parent owns every overlay patch and every extra GPU job.

## Return

`job_id` and final status (`completed` / `failed`).
