# quant-agent

LangChain agent that ports quantization to HuggingFace LLMs. Given a model and hardware constraints, it:

1. Detects local GPU VRAM (`nvidia-smi`).
2. Looks up the model's architecture and parameter count.
3. Scores candidate quantization methods from a curated catalog.
4. Grounds its pick in recent arxiv papers + method repos via RAG over Chroma.
5. Either emits a script, or launches the quantization end-to-end in the background.

## Two ways to run

### A. Laptop-only (recommend + generate script)

```bash
pip install -e .
cp .env.example .env              # fill ANTHROPIC_API_KEY
python -m quant_agent.ingest
quant-agent "Port meta-llama/Llama-3-8B to 4-bit for 16GB VRAM, vLLM, quality first"
# -> writes out/quantize_..._awq.py
```

### B. On a CUDA EC2 instance (end-to-end execution)

Recommended instance types:
- `g5.xlarge` (24 GB A10G)    — up to 13B at 4-bit
- `g5.2xlarge` (24 GB A10G)   — same VRAM, more CPU/RAM
- `g5.12xlarge` (4×24 GB)     — up to 70B with device_map=auto
- `g6e.xlarge` (48 GB L40S)   — larger models single-GPU
- `p4d.24xlarge` (8×40 GB A100) — for anything bigger

Use the AWS **Deep Learning AMI GPU PyTorch** (Ubuntu 22.04, CUDA 12.1). Then:

```bash
# once, on the EC2 box:
git clone <this repo> && cd model-quantization-agent
bash scripts/bootstrap_ec2.sh           # creates .venvs/{awq,gptq,hqq,bnb}
pip install -e .
cp .env.example .env                    # fill ANTHROPIC_API_KEY
python -m quant_agent.ingest

# then, any time:
quant-agent "Port meta-llama/Llama-3-8B end-to-end at 4-bit, vLLM target"
# -> starts a detached job, prints a job_id
```

#### Monitoring jobs

```bash
quant-agent jobs list
quant-agent jobs status <job_id>
quant-agent jobs logs   <job_id> -n 200
quant-agent jobs kill   <job_id>
```

Jobs run under `setsid` and survive SSH disconnects. State lives in `./jobs/<id>/` (meta.json, script.py, stdout/stderr logs, exit_code sentinel). Quantized weights land in `./quantized/<method>-<model>/`.

## Methods covered

GPTQ, AWQ, SmoothQuant, QuIP#, SpinQuant, HQQ, bitsandbytes (LLM.int8 + NF4), GGUF/llama.cpp k-quants, FP8, OmniQuant, SqueezeLLM, KIVI (KV-cache), LLM-QAT, Marlin, Atom. See `seed/methods.yaml`.

Runnable templates exist for: **AWQ, GPTQ, HQQ, bnb-NF4** (the methods wired into `bootstrap_ec2.sh`). Adding a new template means dropping a Jinja file into `src/quant_agent/templates/` and setting `template: <id>` in the YAML entry.

## Architecture

```
gpu_info ──┐
hf_model_info ─┼─> recommend_quantization ──> generate_script ──> execute_quantization
              │         (deterministic)         (Jinja -> .py)     (setsid, per-method venv)
rag_search    │                                                           │
arxiv_fetch   ┘                                                           ▼
github_readme                                                    check_job / tail_job_logs
```

All of these are LangGraph tools bound to a `ChatAnthropic` ReAct agent (`claude-sonnet-4-6` by default; override via `QUANT_AGENT_MODEL`).

## Files

- `seed/methods.yaml` — curated method catalog (arxiv IDs, repos, bit widths, backends, quality/speed/maturity).
- `src/quant_agent/tools/recommender.py` — deterministic scorer. Pure Python, unit-tested.
- `src/quant_agent/executor.py` — background job launcher + registry.
- `scripts/bootstrap_ec2.sh` — sets up per-method venvs with matching torch/CUDA pins.
