# quant-agent

Skills-first porter: name a **quantization method**, a **Hugging Face model**,
and a **GPU instance**. An agent finds the paper and GitHub repo, clones and
snapshots onto the box, ports in a reviewable overlay, runs one GPU job,
verifies the **saved** artifact still generates, and **always** benchmarks LLM
metrics against the fp16 snapshot.

## Find, compare, fetch

Library users do not run the agent. They filter one table, compare metrics,
then download weights themselves from Hugging Face.

1. **Table:** `compare/catalog.json` — one row per method × model × instance.
2. **Filter** any of `model_id`, `method_name`, `gpu_instance`.
3. **Compare** WikiText-2 PPL / tok/s / VRAM on the matching rows (`is_best` is the pick).
4. **Fetch** with `huggingface-cli download <hub_repo_id>` (or `snapshot_download`).

```bash
PY=$(command -v python || command -v python3)
S="$PY .agents/skills/_shared/scripts"
$S/compare.py --model-id microsoft/Phi-3-mini-4k-instruct
$S/compare.py --model-id microsoft/Phi-3-mini-4k-instruct --gpu-instance g5.2xlarge --best
$S/compare.py --model-id microsoft/Phi-3-mini-4k-instruct --method FlatQuant --fetch
```

`--fetch` prints the download command. It does not download. If `hub_repo_id`
is null, that row is local-only until `quant-publish --upload`.

## Contribute a run

Others post successful runs the same way: **public Hub weights** + **one JSON PR**.
Do not commit checkpoints and do not edit `catalog.json`.

1. Run the agent (or the same WikiText-2 helper) until `quality_ok`.
2. Upload weights: `quant-publish --upload --repo-id <you>/<slug>`.
3. Add `compare/contributions/<model>__<gpu>__<method>.json` and open a PR.

```bash
PY=$(command -v python || command -v python3)
S="$PY .agents/skills/_shared/scripts"
$S/compare.py --job-id <job_id> --request out/requests/<slug>.json \
  --hub-url https://huggingface.co/<you>/<slug> --export-contribution
```

Details: `compare/contributions/README.md`. After merge, the row shows up in
`catalog.json` for the same model × instance, ranked against other methods.

## Agent workflow

Codex, Claude, and Cursor load `.agents/skills` (mirrored under
`.claude/skills` and `.cursor/skills`). Helpers:
`.agents/skills/_shared/scripts/`. No method catalog inside the agent loop.

`port AWQ to Qwen/Qwen2.5-0.5B-Instruct on g5.xlarge`

1. **quant** (parent) collects the three inputs. `quant-setup` stays in the parent
   if a gated model needs `HF_TOKEN`.
2. **quant-gather** — paper + GitHub, clone, HF snapshot, GPU facts, one venv.
   Writes `out/requests/<slug>.json`.
3. **quant-port** — `dispatch` / `llama_alias` / `adapter_only`, validate only,
   one ranked winner under `out/overlays/`.
4. **quant-run** — launch with `--allow-unsafe-host-execution`.
5. **quant-verify** — reload saved weights (not Hub fp16) and require generation.
6. **quant-benchmark** — always compare WikiText-2 perplexity (and throughput/VRAM)
   against the original fp16 snapshot.
7. **quant-diagnose** — classify into typed issue codes. The **parent** runs the
   same retry loop for every method × model × GPU (`retry_ranked_overlay` /
   `author_fix` / `kernel`). Default two extra GPU jobs for quality retries;
   packed-path dtype/eval/prefill follow-ups are helper-capped, not
   method-specific. Stop only when diagnose says `none`.
8. **quant-kernel** — only if diagnose says `kernel`. First overlay is complete
   (pack if the repo has it, plus prefill SDPA/flash).
9. **quant-publish** — parent only. Stages `out/hub/<slug>/` (weights + WikiText-2
   `metrics.json` + model card) and uploads to Hugging Face Hub when `HF_TOKEN`
   is set so others can `snapshot_download` the artifact. Hub is **not** how
   methods are compared. `compare/` ranks every method on the same `model_id`
   × `gpu_instance` (quality_ok, then tok/s, then VRAM) so a user can pick
   the best and pull that row's weights.
10. **compare** — written by `benchmark.py` after every passed WikiText-2 run.
    `compare/index.json` lists boards; `compare/groups/<model>__<gpu>.json`
    has one row per method and a `best` pick. Optional Hub URL on the row
    after publish upload.

## Setup

Isolated NVIDIA GPU host (Deep Learning AMI GPU PyTorch, Ubuntu 22.04, or any
box with driver + CUDA 12.x, Python ≥ 3.10, git). Suggested: `g5.xlarge` (≤13B),
`g6e.xlarge` (≤34B), `g5.12xlarge` (≤70B). Torch wheels match this host's nvcc/toolkit
(`QUANT_AGENT_TORCH_SPEC=torch==X.Y.Z|cuZZZ` to override).

```bash
git clone <this repo> && cd model-quantization-agent
PY=$(command -v python || command -v python3)
"$PY" -m pip install -c constraints.txt -e '.[dev]'
```

Gated models — export a token in the parent shell, never in chat:

```bash
read -rsp "HuggingFace token: " HF_TOKEN
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
echo
```

Optional mode-600 `.env`, then `source .agents/skills/_shared/load_env.sh .env`.
Open the repo in Codex, Claude Code, or Cursor and invoke the `quant` skill.

`--allow-unsafe-host-execution` runs third-party and generated code. Use it only
on a disposable host. Installers get no cloud credentials; quantize/verify get
the HF token only when needed.

## Jobs

```bash
PY=$(command -v python || command -v python3)
"$PY" .agents/skills/_shared/scripts/jobs.py list
"$PY" .agents/skills/_shared/scripts/jobs.py status <job_id>
"$PY" .agents/skills/_shared/scripts/jobs.py logs <job_id> -n 200
"$PY" .agents/skills/_shared/scripts/jobs.py kill <job_id>
"$PY" .agents/skills/_shared/scripts/compare.py \
  --model-id <org/model> --gpu-instance <instance> --best
```

State: `jobs/<id>/`. Weights: `quantized/`. Comparison: `compare/`. Clone:
`.venvs/<slug>/repo` (never edited). Workspace override: `QUANT_AGENT_WORKSPACE`.
