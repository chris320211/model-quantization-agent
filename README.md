# quant-agent

Two-subagent LangChain pipeline that ports quantization to HuggingFace LLMs. Given a free-form request like `"port llama2 7b to g5.xlarge"`, it:

1. **Research agent** resolves the model + instance, walks the catalog, and returns **3–8 candidate methods** (no winner picked).
2. **You pick one** from stdin.
3. **Adapt agent** clones the chosen method's repo, builds its venv, learns the target model's exact architecture (full `config.json` + a meta-device module-tree introspection), consults the method's paper when needed, writes a script, and validates it (`ast.parse` + top-level dry-import in the method's venv; up to 3 retries).
4. The validated script is atomically promoted and launched end-to-end on the box
   (skip execution with `--dry`; Adapt still requires the host-execution acknowledgement).

## Setup

After cloning, install the package and run the interactive setup:

```bash
pip install -c constraints.txt -e .
quant-agent setup
```

`quant-agent setup` prompts for `ANTHROPIC_API_KEY` (the only required credential) plus optional `GITHUB_TOKEN` (raises GitHub rate limits for repo clones/README fetches) and `HUGGINGFACE_HUB_TOKEN` (gated models). Input is hidden (via `getpass`), so nothing lands in shell history. It atomically writes the configured workspace's `.env` with mode `0600`; symlinks are refused. By default it makes a 1-token call to Anthropic to verify the key; pass `--no-validate` to skip, `--force` to overwrite an existing regular file, or `--no-optional` to skip optional tokens.

## Run it

### Laptop (dry mode — research + script only)

```bash
quant-agent ask --dry --allow-unsafe-host-execution "port llama2 7b to g5.xlarge"
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
bash scripts/bootstrap_ec2.sh           # creates .venvs/{awq,gptq,bnb_nf4,bnb_llm_int8}
pip install -c constraints.txt -e .
quant-agent setup                       # interactive; writes .env (chmod 600)

# then, any time:
quant-agent ask --allow-unsafe-host-execution "port llama2 7b to g5.xlarge"
# -> pick a method from the list
# -> detached job launches; prints job_id
```

The acknowledgement flag is required because Adapt installs and imports mutable
third-party quantization repositories and the executor runs generated code. Use it
only on an isolated, disposable host. Install/build subprocesses do not receive API
credentials; the quantization runtime receives only the Hugging Face token when needed.

#### Monitoring jobs

```bash
quant-agent jobs list
quant-agent jobs status <job_id>
quant-agent jobs logs   <job_id> -n 200
quant-agent jobs kill   <job_id>
```

Jobs run under `setsid` and survive SSH disconnects. State lives in the configured
workspace's `jobs/<id>/` (meta.json, script.py, stdout/stderr logs, exit-code
sentinel). Quantized weights land under `quantized/`. Editable checkouts default to
the repository; installed packages default to `~/.local/share/quant-agent`. Override
with `QUANT_AGENT_WORKSPACE`.

## Methods covered

GPTQ, AWQ, SmoothQuant, QuIP#, SpinQuant, bitsandbytes (LLM.int8 + NF4), FP8, OmniQuant, SqueezeLLM, QuaRot, FlatQuant, AutoRound, AQLM, VPTQ, KIVI (KV-cache), LLM-QAT, GGUF, and more — 35 methods total. See `src/quant_agent/data/methods.yaml` for the authoritative list of ids.

End-to-end execution venvs are wired for: **AWQ, GPTQ, bnb-NF4, bnb-LLM.int8** (the methods in `scripts/bootstrap_ec2.sh`). Other methods surface in Research and are adapted by the Adapt agent, which builds a matching `.venvs/<method_id>/` on demand via `install_method_venv`. Venvs resolve purely by the `.venvs/<method_id>/` naming convention (see `executor.venv_python`); to pre-provision one, add a case to `bootstrap_ec2.sh` keyed by the catalog id.

## Architecture

```
                ┌──────────────────────────────┐
user input ──► │  Research agent (1 LLM call) │ ──► ResearchReport (3-8 candidates, tradeoffs)
                │  resolve model + instance    │
                │  catalog walk (no RAG)       │
                └──────────────────────────────┘
                              │
                        stdin: pick 1..N
                              ▼
                ┌──────────────────────────────────┐
                │  Adapt agent (ReAct loop)         │
                │  clone_method_repo / read_repo /  │ ──► out/quantize_<model>_<method>.py
                │  fetch_model_config /             │      (validated: ast + dry-import)
                │  inspect_model_architecture /     │
                │  read_paper / write_script        │
                └──────────────────────────────────┘
                              │
                              ▼
                  execute_quantization (setsid, per-method venv)
                              │
                              ▼
                  check_job / tail_job_logs
```

The Research agent uses `ChatAnthropic.with_structured_output(ResearchReport)` — not a ReAct loop, and grounds purely on the structured catalog (no RAG). The Adapt agent is a `create_react_agent` that clones the method repo, introspects the target model's architecture, reads the method's paper, and writes a validated script. Default model `claude-sonnet-4-6`; override with `QUANT_AGENT_MODEL`.

## Files

- `src/quant_agent/data/methods.yaml` — packaged method catalog (arxiv IDs, repos, bit widths, backends, quality/speed/maturity).
- `src/quant_agent/data/model_aliases.yaml` — packaged fuzzy model name → canonical HuggingFace id.
- `src/quant_agent/data/aws_instances.yaml` — packaged EC2 GPU instance types → VRAM / GPU count / GPU model.
- `src/quant_agent/research_agent.py` — context loader + structured-output call.
- `src/quant_agent/adapt_agent.py` — ReAct loop: clone repo + read paper + introspect model architecture + validated `write_script`.
- `src/quant_agent/tools/paper.py` — `read_paper` (arXiv full-text fetch + cache) for the Adapt agent.
- `src/quant_agent/tools/model_arch.py` — `fetch_model_config` + meta-device `inspect_model_architecture`.
- `src/quant_agent/orchestrator.py` — Research → select → Adapt → execute.
- `src/quant_agent/tools/script_io.py` — `ValidationSession` + ast/dry-import validation.
- `src/quant_agent/executor.py` — background job launcher + registry.
- `scripts/bootstrap_ec2.sh` — sets up per-method venvs with matching torch/CUDA pins.
