---
name: quant-run
description: >-
  Launch a validated quantization script on this GPU instance, monitor it, and
  retry with one fix per failed attempt. Use as a quant-run subagent after
  quant-port. Does not change methods or rewrite kernels.
---

# Quant Run

You are a **subagent**. Do only run. Need request JSON + script path + overlay dir.

`S=python .agents/skills/_shared/scripts`

## Do

1. Launch (isolated box; flag required):

   ```bash
   $S/launch.py <script.py> --request out/requests/<slug>.json \
     --overlay-dir <winner_overlay_dir> --allow-unsafe-host-execution
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

4. Do not rewrite kernels here. Do not start a second GPU job while one is running.

## Return

`job_id` and final status (`completed` / `failed`).
