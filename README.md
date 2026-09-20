# quant-agent

Skills-first porter: name a **quantization method**, a **Hugging Face model**,
and a **GPU instance**. An agent finds the paper and GitHub repo, clones and
snapshots onto the box, ports in a reviewable overlay, runs one GPU job,
verifies the **saved** artifact still generates, and **always** benchmarks LLM
metrics against the fp16 snapshot.

## Find, compare, fetch

**One library.** Each row is model × GPU instance × method, with links to the
paper, the method GitHub repo, and Hugging Face weights. Details:
`compare/LIBRARY.md`.

1. Filter `compare/catalog.json` by `model_id`, `method_name`, `gpu_instance`.
2. Compare WikiText-2 PPL / tok/s / VRAM (`is_best` is the pick).
3. Open paper / method repo / Hub from the row, or `huggingface-cli download <hub_repo_id>`.
4. Hub collection URL lives in `compare/library.json`.

```bash
PY=$(command -v python || command -v python3)
S="$PY agents/skills/_shared/scripts"
$S/compare.py --model-id microsoft/Phi-3-mini-4k-instruct
$S/compare.py --model-id microsoft/Phi-3-mini-4k-instruct --gpu-instance g5.2xlarge --best
$S/compare.py --model-id microsoft/Phi-3-mini-4k-instruct --method FlatQuant --fetch
```

`--fetch` prints the download command. It does not download.

## Contribute a run

After a **beneficial** run (quality_ok and better VRAM or tok/s than fp16):
`quant-publish`, then `quant-catalog`. Do not commit checkpoints.

1. Upload: `quant-publish --upload --repo-id <you>/<slug>`
2. Record the standard row (model, GPU, method, paper, repo, Hub):
   `agents/skills/quant-catalog/SKILL.md`
3. PR `compare/contributions/<model>__<gpu>__<method>.json`

```bash
PY=$(command -v python || command -v python3)
S="$PY agents/skills/_shared/scripts"
$S/compare.py --catalog --job-id <job_id> --request out/requests/<slug>.json \
  --hub-url https://huggingface.co/<you>/<slug>
```

Details: `compare/LIBRARY.md`. After merge, the row ranks against other methods
on the same model × instance.

## Agent workflow

Cursor, Claude, and Codex read `agents/AGENTS.md`, then `agents/skills/`.
Helpers: `agents/skills/_shared/scripts/`. No method catalog inside the agent loop.

`port AWQ to Qwen/Qwen2.5-0.5B-Instruct on g5.xlarge`

1. **quant** (parent) collects the three inputs. **quant-setup** stays in the
   parent on first load: create `.env` once from `.env.example`, load it every
   shell. Do not recreate `.env` if it already exists.
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
   is set.
10. **quant-catalog** — parent only, after a beneficial run. Writes the standard
    library row (model, GPU instance, method, paper, method repo, Hugging Face)
    into `compare/catalog.json`. See `compare/LIBRARY.md`.

## Setup

Isolated NVIDIA GPU host (Deep Learning AMI GPU PyTorch, Ubuntu 22.04, or any
box with driver + CUDA 12.x, Python ≥ 3.10, git). Suggested: `g5.xlarge` (≤13B),
`g6e.xlarge` (≤34B), `g5.12xlarge` (≤70B). Torch wheels match this host's nvcc/toolkit
(`QUANT_AGENT_TORCH_SPEC=torch==X.Y.Z|cuZZZ` to override).

```bash
git clone <this repo> && cd model-quantization-agent
PY=$(command -v python || command -v python3)
"$PY" -m pip install -c constraints.txt -e '.[dev]'
cp .env.example .env
chmod 600 .env
# edit .env in your editor; put HF_TOKEN=... (and optional GITHUB_TOKEN=)
source agents/skills/_shared/load_env.sh .env
```

Create `.env` **once** on this machine. Every new terminal, only `source` the
loader. Never paste token values into chat. `.env` is gitignored.

Open the repo in Codex, Claude Code, or Cursor and invoke the `quant` skill.

`--allow-unsafe-host-execution` runs third-party and generated code. Use it only
on a disposable host. Installers get no cloud credentials; quantize/verify get
the HF token only when needed.

## Jobs

```bash
PY=$(command -v python || command -v python3)
"$PY" agents/skills/_shared/scripts/jobs.py list
"$PY" agents/skills/_shared/scripts/jobs.py status <job_id>
"$PY" agents/skills/_shared/scripts/jobs.py logs <job_id> -n 200
"$PY" agents/skills/_shared/scripts/jobs.py kill <job_id>
"$PY" agents/skills/_shared/scripts/compare.py \
  --model-id <org/model> --gpu-instance <instance> --best
```

State: `jobs/<id>/`. Weights: `quantized/`. Comparison: `compare/`. Clone:
`.venvs/<slug>/repo` (never edited). Workspace override: `QUANT_AGENT_WORKSPACE`.
