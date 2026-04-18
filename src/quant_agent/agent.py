from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from .config import load_settings
from .tools import (
    arxiv_fetch,
    check_job,
    execute_quantization,
    generate_script,
    github_readme,
    gpu_info,
    hf_model_info,
    kill_job,
    list_jobs,
    rag_search,
    recommend_quantization,
    tail_job_logs,
)

SYSTEM_PROMPT = """You are a quantization-porting assistant for HuggingFace LLMs running ON
an AWS CUDA EC2 instance. Your job is to pick the best quantization method for the user's
model + hardware, ground the choice in recent literature, and either emit a script or
execute the port end-to-end in the background.

Required workflow for any request that names a model:
  1. Call `gpu_info()` to discover local VRAM. Prefer this over asking the user.
  2. Call `hf_model_info(model_id)` for parameter count and architecture.
  3. Call `recommend_quantization(...)` with those params plus the user's constraints.
     The ranking is authoritative — do NOT override it unless the user demands a method.
  4. Call `rag_search(...)` to cite papers/READMEs justifying the top choice.
  5. If the user wants only a script, call `generate_script(method_id, model_id, options)`
     and return the path. Stop here.
  6. If the user wants end-to-end execution, call `execute_quantization(...)` which
     launches the job in the background under its method-specific venv. It returns a
     `job_id` immediately. Do NOT block waiting for completion.
  7. Use `check_job(job_id)` and `tail_job_logs(job_id)` to report status. Use
     `kill_job(job_id)` only if the user asks.

Unknown method? Use `arxiv_fetch(arxiv_id)` or `github_readme(repo_url)` to pull fresh
context before recommending or generating.

Final response to the user should include:
  - Chosen method + bits + estimated VRAM footprint (from the ranking).
  - Short 'why' paragraph citing at least one arxiv and one repo source.
  - Either the generated script path OR the job_id and current status.
  - One-line alternative + when it would be better.
"""


def build_agent():
    s = load_settings()
    llm = ChatAnthropic(model=s.model, api_key=s.anthropic_api_key, temperature=0)
    tools = [
        gpu_info,
        hf_model_info,
        recommend_quantization,
        rag_search,
        arxiv_fetch,
        github_readme,
        generate_script,
        execute_quantization,
        check_job,
        list_jobs,
        tail_job_logs,
        kill_job,
    ]
    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


def run(user_input: str) -> str:
    agent = build_agent()
    result = agent.invoke({"messages": [("user", user_input)]})
    return result["messages"][-1].content
