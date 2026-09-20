---
name: quant-run
description: >-
  Launch a validated quantization script on this GPU instance, monitor it, and
  retry with one fix per failed attempt. Use as a quant-run subagent after
  quant-port. Does not change methods or rewrite kernels.
---

# Quant Run

You are a **subagent**. Do only run. Need request JSON + script path + overlay dir.
Do not ask the parent mid-stage. If `HF_TOKEN` is unset and `.env` exists,
`source agents/skills/_shared/load_env.sh .env` (never print values).

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

## Do

1. Launch (isolated box; flag required):

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

3. On failure: one new fix per attempt, same method and overlay strategy.
   Re-validate with `validate_script.py`, then launch again. Bounded retries (3).
   Do not retry gated-model auth, OOM at the same config, disk full, or wrong GPU.

4. Do not rewrite kernels here. Do not start a second GPU job while one is
   running. Do not interpret WikiText-2; that is benchmark + parent diagnose.
   Your launch retries are process failures only (same overlay strategy). The
   parent owns the method-agnostic quality/efficiency retry loop.

## Return

`job_id` and final status (`completed` / `failed`).
