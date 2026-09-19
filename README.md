# quant-agent

Skills-first porter: name a **quantization method**, a **Hugging Face model**,
and a **GPU instance**. An agent finds the paper and GitHub repo, clones and
snapshots onto the box, ports in a reviewable overlay, runs one GPU job, and
verifies the **saved** artifact still generates.

No method catalog and no CLI. Codex, Claude, and Cursor load
`.agents/skills` (mirrored under `.claude/skills` and `.cursor/skills`).
Helpers: `.agents/skills/_shared/scripts/`.

## Workflow

`port AWQ to Qwen/Qwen2.5-0.5B-Instruct on g5.xlarge`

1. **quant** (parent) collects the three inputs. `quant-setup` stays in the parent
   if a gated model needs `HF_TOKEN`.
2. **quant-gather** — paper + GitHub, clone, HF snapshot, GPU facts, one venv.
   Writes `out/requests/<slug>.json`.
3. **quant-port** — `dispatch` / `llama_alias` / `adapter_only`, validate only,
   one ranked winner under `out/overlays/`.
4. **quant-run** — launch with `--allow-unsafe-host-execution`.
5. **quant-verify** — reload saved weights (not Hub fp16) and require generation.
6. **quant-kernel** — Triton overlay only if the run did not help, or the user asked.

## Setup

Isolated NVIDIA GPU host (Deep Learning AMI GPU PyTorch, Ubuntu 22.04, or any
box with driver + CUDA 12.x, Python ≥ 3.10, git). Suggested: `g5.xlarge` (≤13B),
`g6e.xlarge` (≤34B), `g5.12xlarge` (≤70B). Torch wheels are chosen at install
time (`QUANT_AGENT_TORCH_SPEC=torch==X.Y.Z|cuZZZ` to override).

```bash
git clone <this repo> && cd model-quantization-agent
python -m pip install -c constraints.txt -e '.[dev]'
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
python .agents/skills/_shared/scripts/jobs.py list
python .agents/skills/_shared/scripts/jobs.py status <job_id>
python .agents/skills/_shared/scripts/jobs.py logs <job_id> -n 200
python .agents/skills/_shared/scripts/jobs.py kill <job_id>
```

State: `jobs/<id>/`. Weights: `quantized/`. Clone: `.venvs/<slug>/repo` (never
edited). Workspace override: `QUANT_AGENT_WORKSPACE`.
