---
name: quant-gather
description: >-
  Find a quantization method’s paper and GitHub from the method name, clone the
  repo, download the Hugging Face model, record GPU facts, and install one
  method venv. Use as a quant-gather subagent. No method catalog. Does not
  adapt or launch GPU jobs.
---

# Quant Gather

You are a **subagent**. Do only gather. Read
[_shared/pipeline_contract.md](../_shared/pipeline_contract.md).

`S=python .agents/skills/_shared/scripts`

Host-mutating steps need `--allow-unsafe-host-execution` on an isolated box.

## Do

1. Web-search `method_name` for the official paper and official GitHub.
   Prefer the authors’ repo. If two repos look official, stop and ask once.
   GitHub URL must be `https://github.com/owner/repo` (no git@, no other hosts).
2. Download the paper:

   ```bash
   $S/paper.py --arxiv-id <id>
   $S/refuse.py .cache/papers/<id>.txt
   ```

   Stop if refuse returns `qat_training_only`, `kv_cache_only`, or `bitnet_1_58`.
3. Clone into `.venvs/<slug>/repo`:

   ```bash
   $S/clone.py --slug <slug> --repo-url https://github.com/owner/repo \
     --allow-unsafe-host-execution
   ```

4. Snapshot the Hugging Face model (uses `HF_TOKEN` from the environment; never print it):

   ```bash
   $S/snapshot.py --model-id <org/model>
   ```

5. GPU facts:

   ```bash
   $S/gpu.py --gpu-instance <instance>
   ```

6. **One** venv install for this slug (before any port worker):

   ```bash
   $S/install_venv.py --slug <slug> --allow-unsafe-host-execution \
     --step 'pip install -e .'
   ```

   Add extra `--step` values from the repo README (python/pip only; no shell
   operators). Torch pins are applied automatically.

7. Write the handoff:

   ```bash
   $S/request.py --write-json '{"method_name":"...","model_id":"...","gpu_instance":"...","arxiv_id":"...","paper_path":"...","repo_url":"...","repo_path":"...","repo_commit":"...","hf_snapshot_path":"...","slug":"..."}'
   ```

   Include optional GPU fields from `gpu.py` when present.

Slug: lowercase method + short model + instance, e.g. `awq-qwen25-05b-g5xlarge`.

## Do not

Adapt, write overlays, launch jobs, rewrite kernels, or read `.env`.

## Return

The request JSON path only.
