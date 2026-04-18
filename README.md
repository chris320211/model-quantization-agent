# quant-agent

Two-subagent LangChain pipeline that ports quantization to HuggingFace LLMs. Given a free-form request like `"port llama2 7b to g5.xlarge"`, it:

1. **Research agent** resolves the model + instance, surveys the catalog + RAG index, and returns **3–8 candidate methods** (no winner picked).
2. **You pick one** from stdin.
3. **Adapt agent** reads the chosen repo's actual README + source via the GitHub API, writes a script, and validates it (`ast.parse` + top-level dry-import in the method's venv; up to 3 retries).
4. The validated script is launched end-to-end on the box (skip with `--dry`).

## Setup

After cloning, install the package and run the interactive setup:

```bash
pip install -e .
quant-agent setup
```

`quant-agent setup` prompts for `ANTHROPIC_API_KEY` (required) and optionally `GITHUB_TOKEN` / `HUGGINGFACE_HUB_TOKEN`. Input is hidden (via `getpass`), so nothing lands in shell history. It writes `.env` to the repo root with mode `0600` and is gitignored. By default it makes a 1-token call to Anthropic to verify the key; pass `--no-validate` to skip, `--force` to overwrite an existing file, or `--no-optional` to skip the optional tokens.

## Run it

### Laptop (dry mode — research + script only)

```bash
python -m quant_agent.ingest
quant-agent ask --dry "port llama2 7b to g5.xlarge"
# -> lists 3-8 candidates with tradeoffs
# -> you type "1" (or 2..N, or q)
# -> writes out/quantize_<model>_<method>.py  (no job launched)
```

### On a CUDA EC2 instance (end-to-end)

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
quant-agent setup                       # interactive; writes .env (chmod 600)
python -m quant_agent.ingest

# then, any time:
quant-agent ask "port llama2 7b to g5.xlarge"
# -> pick a method from the list
# -> detached job launches; prints job_id
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

End-to-end execution venvs are wired for: **AWQ, GPTQ, HQQ, bnb-NF4** (the methods in `scripts/bootstrap_ec2.sh`). Other methods surface in Research and are adapted by the Adapt agent; add a new venv in `bootstrap_ec2.sh` and a `METHOD_TO_VENV` entry in `executor.py` to enable end-to-end launch.

## Architecture

```
                ┌──────────────────────────────┐
user input ──► │  Research agent (1 LLM call) │ ──► ResearchReport (3-8 candidates, tradeoffs)
                │  resolve model + instance    │
                │  rag_survey + catalog        │
                └──────────────────────────────┘
                              │
                        stdin: pick 1..N
                              ▼
                ┌──────────────────────────────┐
                │  Adapt agent (ReAct loop)    │
                │  github_readme / list_dir /  │ ──► out/quantize_<model>_<method>.py
                │  github_file / rag_search /  │      (validated: ast + dry-import)
                │  hf_model_info / write_script│
                └──────────────────────────────┘
                              │
                              ▼
                  execute_quantization (setsid, per-method venv)
                              │
                              ▼
                  check_job / tail_job_logs
```

The Research agent uses `ChatAnthropic.with_structured_output(ResearchReport)` — not a ReAct loop. The Adapt agent is a `create_react_agent` with the GitHub + RAG + `write_script` toolset. Default model `claude-sonnet-4-6`; override with `QUANT_AGENT_MODEL`.

## Files

- `seed/methods.yaml` — curated method catalog (arxiv IDs, repos, bit widths, backends, quality/speed/maturity).
- `seed/model_aliases.yaml` — fuzzy model name → canonical HuggingFace id.
- `seed/aws_instances.yaml` — EC2 GPU instance types → VRAM / GPU count / GPU model.
- `src/quant_agent/research_agent.py` — context loader + structured-output call.
- `src/quant_agent/adapt_agent.py` — ReAct loop with GitHub-fetching tools + validated `write_script`.
- `src/quant_agent/orchestrator.py` — Research → select → Adapt → execute.
- `src/quant_agent/tools/script_io.py` — `ValidationSession` + ast/dry-import validation.
- `src/quant_agent/executor.py` — background job launcher + registry.
- `scripts/bootstrap_ec2.sh` — sets up per-method venvs with matching torch/CUDA pins.
