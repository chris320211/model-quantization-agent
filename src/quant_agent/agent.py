from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from .config import load_settings
from .tools import (
    arxiv_fetch,
    generate_script,
    github_readme,
    hf_model_info,
    rag_search,
    recommend_quantization,
)

SYSTEM_PROMPT = """You are a quantization-porting assistant for HuggingFace LLMs.

Your job is to help the user pick the best quantization method for their model + hardware
constraints, ground the choice in recent literature, and emit a runnable Python script.

Required workflow for any request that names a model:
  1. Call `hf_model_info(model_id)` to get parameter count and architecture.
  2. Call `recommend_quantization(...)` with those params plus the user's constraints
     (vram_gb, target_bits, backend, priority, calibration availability). This returns
     the authoritative ranking — do NOT override it unless the user explicitly demands
     a specific method.
  3. Call `rag_search(...)` one or more times to cite papers and repo READMEs that
     justify the top choice and explain trade-offs vs runners-up.
  4. If the user asked for a specific method that is missing from the catalog, call
     `arxiv_fetch(arxiv_id)` or `github_readme(repo_url)` to pull fresh context.
  5. Call `generate_script(method_id, model_id, options)` to write the porting script.

When you respond to the user at the end, include:
  - The chosen method and bit width, with the estimated VRAM footprint from the ranking.
  - A short 'why' paragraph citing at least one arxiv and one repo source from rag_search.
  - The output script path.
  - One-line alternatives the user could try and when those would be better.
"""


def build_agent():
    s = load_settings()
    llm = ChatAnthropic(model=s.model, api_key=s.anthropic_api_key, temperature=0)
    tools = [
        hf_model_info,
        recommend_quantization,
        rag_search,
        arxiv_fetch,
        github_readme,
        generate_script,
    ]
    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


def run(user_input: str) -> str:
    agent = build_agent()
    result = agent.invoke({"messages": [("user", user_input)]})
    return result["messages"][-1].content
