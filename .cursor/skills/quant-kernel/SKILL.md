---
name: quant-kernel
description: >-
  After a verified run that did not beat the baseline, rewrite hot paths to
  Triton using the paper, repo, and this GPU type. Use as a quant-kernel
  subagent. Keep the rewrite only if generate still works and speed improves.
---

# Quant Kernel

You are a **subagent**. Do only kernel optimize. Need request JSON, `job_id`,
and baseline metrics. Skip unless the parent said the run did not help or the
user asked to go faster.

`S=python .agents/skills/_shared/scripts`

## Do

1. Profile hot Python/CUDA ops from the completed job logs (do not edit the clone).
2. Reread the paper (`paper_path`) and repo for what the method is supposed to do,
   plus `gpu_arch` from the request JSON.
3. Rewrite those ops to **Triton** (small CUDA only if Triton cannot express the
   kernel) **in a new overlay** with strategy `kernel_triton`.
4. Write and validate like a port worker:

   ```bash
   $S/overlay.py write --slug <slug> --strategy kernel_triton --model-id <id> \
     --base-commit <sha> --patch-file /tmp/kernel.patch --rationale "..."
   $S/overlay.py apply-check --overlay-dir <bundle> --repo .venvs/<slug>/repo
   $S/validate_script.py <script.py> --model-id <id> --output-dir ./quantized/<slug> \
     --overlay-dir <bundle>
   ```

5. Return the new overlay path. The parent will `quant-run` + `quant-verify`
   again. Keep the rewrite only if generate still works **and** speed/VRAM
   improves. Otherwise return `revert`.

Never edit `.venvs/<slug>/repo`.

## Return

New overlay path, or `revert` if you cannot improve safely.
