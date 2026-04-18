# quant-agent

A LangChain agent that ports quantization to HuggingFace LLMs. Given a model ID and your hardware/quality constraints, it:

1. Looks up the model's architecture and size on the HF Hub.
2. Scores candidate quantization methods from a curated catalog.
3. Grounds its choice in recent literature via RAG over arxiv papers and method repos.
4. Emits a ready-to-run Python quantization script.

## Setup

```bash
pip install -e .
cp .env.example .env       # then fill in ANTHROPIC_API_KEY
python -m quant_agent.ingest
```

## Usage

```bash
quant-agent "Port meta-llama/Llama-3-8B to 4-bit for a 16GB RTX 4080, prioritize quality, vLLM"
```

The generated script lands in `./out/`.

## Methods covered

GPTQ, AWQ, SmoothQuant, QuIP#, SpinQuant, HQQ, bitsandbytes (LLM.int8 + NF4), GGUF/llama.cpp k-quants, FP8, OmniQuant, SqueezeLLM, KIVI, LLM-QAT, Marlin, Atom. See `seed/methods.yaml`.

New methods can be added by appending to `seed/methods.yaml` and re-running ingest, or fetched on demand via the agent's `arxiv_fetch` tool.
