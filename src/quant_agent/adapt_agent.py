"""Adapt agent: read the chosen method's real repo and author a quantization script.

This is a ReAct loop (`create_react_agent`) with a narrow toolset. The agent:
  1. Reads the repo README and a few entry-point files via GitHub contents API.
  2. Grounds the library API in what it actually sees there (no Jinja templates).
  3. Writes a script via `write_script`, which validates (ast.parse + dry-import in the
     method venv) before committing to disk. On failure it retries up to 3 times.

Returns (script_path, script_code) on success — the orchestrator then hands the code
off to execute_quantization.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from .config import load_settings
from .schemas import MethodCandidate
from .tools import (
    github_file,
    github_list_dir,
    github_readme,
    hf_model_info,
)
from .tools.rag import rag_search
from .tools.script_io import ValidationSession, make_write_script_tool

log = logging.getLogger(__name__)

_PROMPT = """You are the Adapt agent. You have been handed a quantization method chosen
from a catalog and a target HuggingFace model. Your job is to produce a single Python
script that quantizes the user's model using the REAL API of the chosen repo — not a
guess, not a template.

Chosen method:
  id:        {method_id}
  name:      {method_name}
  repo_url:  {repo_url}
  bits:      {bits}

Target model: {model_id}
Output script path: {script_path}

Workflow (follow in order):
  1. Call `github_readme(repo_url)` to learn the intended usage.
  2. Call `github_list_dir(repo_url, path="")` to see the layout. If helpful, drill into
     likely directories (examples/, scripts/, src/, <package>/).
  3. Call `github_file(repo_url, path=...)` on 1-3 files that look like the quantization
     entry point (common names: quantize.py, main.py, examples/basic_usage.py,
     auto_gptq/, awq/quantize/quantizer.py, hqq/core/quantize.py).
  4. If you need background on the method itself, `rag_search(query, method_id="{method_id}")`
     returns chunks filtered to this method.
  5. Write the full script via `write_script(path="{script_path}", code=...)`. The tool
     validates the code (ast.parse + top-level dry-import against the method venv) and
     returns status="ok" on success, or status="error" with attempts_left on failure.
     Fix reported errors and retry. Stop as soon as status="ok".

Script requirements:
  - Must be a standalone Python script runnable by the method venv's python.
  - Must load the model from HF by id `{model_id}`.
  - Must write the quantized model to the CWD under ./quantized/<method>-<model>/ using
    the library's native save API (e.g. save_quantized / save_pretrained).
  - Must import the real library API surface you saw in the repo files — no invented names.
  - No CLI argparse is needed; hardcode the model id and output dir.
  - HuggingFace auth: if the target model is gated (meta-llama/*, mistralai/Mixtral-*,
    google/gemma-*, and similar), the script MUST read the token from the environment
    (`HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")`)
    and pass it to every loader/tokenizer call that accepts an auth kwarg (e.g.
    `from_pretrained(..., token=HF_TOKEN)`, `login(token=HF_TOKEN)`, or the repo's own
    `hf_token=` argument). Do NOT hardcode the token and do NOT pass `None` if the
    environment variable is available at runtime.

Stop calling tools once `write_script` returns status="ok".
"""


def _safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")


def _read_output_file(path: Path) -> str:
    text = path.read_text()
    # Strip the validation-failure banner if write_script had to write in exhausted mode.
    if text.startswith("# WARNING: failed validation"):
        lines = text.splitlines()
        return "\n".join(lines[1:])
    return text


def run(model_id: str, method: MethodCandidate) -> tuple[str, str]:
    """Run the Adapt ReAct loop. Returns (script_path, script_code)."""
    s = load_settings()

    out_dir = s.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / f"quantize_{_safe_slug(model_id)}_{method.id}.py"

    session = ValidationSession(method_id=method.id)
    write_script = make_write_script_tool(session)

    @tool
    def rag_search_for_method(query: str, k: int = 6) -> str:
        """Search the local quantization-literature index, scoped to the chosen method."""
        return rag_search.invoke({"query": query, "k": k, "method_id": method.id})

    tools = [
        github_readme,
        github_list_dir,
        github_file,
        rag_search_for_method,
        hf_model_info,
        write_script,
    ]

    llm = ChatAnthropic(model=s.model, api_key=s.anthropic_api_key, temperature=0)
    prompt = _PROMPT.format(
        method_id=method.id,
        method_name=method.name,
        repo_url=method.repo_url,
        bits=method.bits,
        model_id=model_id,
        script_path=str(script_path),
    )
    agent = create_react_agent(llm, tools, prompt=prompt)

    agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Port {model_id} to {method.name} ({method.bits}-bit). "
                    f"Write the script to {script_path}.",
                )
            ]
        }
    )

    if not script_path.exists():
        raise RuntimeError(
            f"Adapt agent finished without writing {script_path}. Check the agent's last tool call."
        )

    code = _read_output_file(script_path)
    return str(script_path), code
