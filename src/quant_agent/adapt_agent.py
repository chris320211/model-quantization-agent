"""Adapt agent: clone the method's repo, build its venv, and author a script.

ReAct loop that:
  1. Reads the repo README to learn install + usage.
  2. Clones the repo into .venvs/<method_id>/repo/.
  3. Builds a method-specific venv at .venvs/<method_id>/ and runs the repo's
     install steps (torch + transformers baseline installed automatically).
  4. Inspects the cloned source locally to find an example entry point.
  5. Writes either a standalone script (importing the installed library) or a
     wrapper that subprocess-invokes the repo's example with filled-in args.
  6. Validates via ast.parse + dry-import (stdlib-only wrappers pass trivially).

Returns (script_path, script_code) — the orchestrator hands the code off to the
executor, which launches it with .venvs/<method_id>/bin/python.

When the tune loop calls back in with a non-empty ``hyperparameters`` dict, those
values are baked into the prompt as a TUNE-LOCKED block. The script must reflect
them, and downstream fix_agent invocations must not silently alter them.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from .config import load_settings
from .schemas import MethodCandidate
from .tools import (
    clone_method_repo,
    github_file,
    github_list_dir,
    github_readme,
    hf_model_info,
    install_method_venv,
    list_repo_dir,
    read_repo_file,
    run_in_venv,
)
from .tools.rag import rag_search
from .tools.script_io import ValidationSession, make_write_script_tool

log = logging.getLogger(__name__)

_PROMPT = """You are the Adapt agent. A quantization method has been chosen from the catalog
and you must author a Python script that quantizes the user's model using that method's
real implementation — cloned from its GitHub repo.

Chosen method:
  id:        {method_id}
  name:      {method_name}
  repo_url:  {repo_url}
  bits:      {bits}

Hyperparameters (TUNE-LOCKED — must appear verbatim in the generated script as the
configured values for this method's quantizer; do not substitute defaults). When this
block reads "(none — use method defaults)", omit explicit hyperparameter args and let
the method's own defaults apply.
{hyperparameters_block}

Target model: {model_id}
Output script path: {script_path}

Per-method paths (created by the tools below):
  venv:       .venvs/{method_id}/            (python at .venvs/{method_id}/bin/python)
  cloned repo: .venvs/{method_id}/repo/

Workflow — follow in order:

  1. github_readme(repo_url="{repo_url}") — learn install + usage.

  2. clone_method_repo(method_id="{method_id}", repo_url="{repo_url}")
     Clones to .venvs/{method_id}/repo/ (idempotent).

  3. Identify `install_steps` from the README. Examples:
        Research repos:     ["pip install -r requirements.txt", "pip install -e ."]
                            or ["pip install -e ."]
        Pip-packaged libs:  ["pip install autoawq"]   (for awq)
                            ["pip install gptqmodel datasets"]   (for gptq)
                            ["pip install hqq"]   (for hqq)
                            ["pip install bitsandbytes"]   (for bnb_nf4/bnb_llm_int8)
     A baseline of torch==2.3.1 (cu121) + transformers + accelerate + safetensors +
     sentencepiece is ALWAYS installed first. Do NOT duplicate those in install_steps.

  4. install_method_venv(method_id="{method_id}", install_steps=[...])
     Creates the venv and runs the steps. If it errors:
       - Read the last stderr lines carefully.
       - Adjust install_steps (add a missing dep, drop an optional broken step).
       - Retry ONCE. If it still fails, write a best-effort script anyway — the
         executor will surface the real error at runtime.

  5. Find the example entry point in the cloned repo:
       list_repo_dir(method_id="{method_id}", path="examples")
       list_repo_dir(method_id="{method_id}", path="scripts")
       list_repo_dir(method_id="{method_id}", path="")   (if needed)
     Then read_repo_file on 1–2 most-likely entry files to see their API.

  6. (Optional but recommended) run_in_venv(method_id="{method_id}",
        command="python <entry_path> --help") to confirm the entry is invokable
     and to discover the real arg flag names.

  7. Decide the script style and write it via
     write_script(path="{script_path}", code=...):

     STYLE A — Standalone import (preferred when the method has a stable Python API,
     e.g. autoawq, gptqmodel, hqq, bitsandbytes, or an importable module in the
     cloned repo). Write a single Python file that imports the library and runs
     quantization in-process. Must load the model from HF by id `{model_id}` and
     save output to ./quantized/{method_id}-<safe_model>/.

     STYLE B — Wrapper subprocess (preferred for research repos whose usage is
     "python examples/foo.py --model ..."). Write a Python file that:
        - import sys, os, subprocess, pathlib
        - Reads HF_TOKEN from env (HUGGINGFACE_HUB_TOKEN or HF_TOKEN) and passes it
          into the child env.
        - Computes REPO = pathlib.Path(".venvs/{method_id}/repo").resolve()
        - subprocess.run([sys.executable, "<entry_path_relative_to_repo>",
                          "<flag_name>", "{model_id}", "<output_flag>", str(OUT)],
                          cwd=REPO, env={{**os.environ, ...}}, check=True)
        - OUT = pathlib.Path("./quantized/{method_id}-<safe_model>").resolve()
        - The EXACT flag names MUST come from the --help output or the example source
          you read — do NOT invent flag names.

     Both styles must:
       - Quantize the real model at `{model_id}`.
       - Handle HF gated models: read HF_TOKEN from env and pass to every loader/
         tokenizer call (or login()) and into the child env for wrapper style.
       - Print a final line with the quantized model's on-disk size so success is
         visible in stdout.
       - Begin the script with a header comment block:
            # TUNE-LOCKED HYPERPARAMETERS (do not modify in fix_agent):
            # <one line per name=value from the hyperparameters block above>
         When no hyperparameters are supplied, omit the block.

Stop as soon as write_script returns status="ok". Do not call run_in_venv on the
wrapper itself — the executor will launch it.
"""


def _format_hyperparameters_block(hyperparameters: dict | None) -> str:
    if not hyperparameters:
        return "(none — use method defaults)"
    return json.dumps(hyperparameters, indent=2, sort_keys=True)


def _safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")


def _read_output_file(path: Path) -> str:
    text = path.read_text()
    if text.startswith("# WARNING: failed validation"):
        lines = text.splitlines()
        return "\n".join(lines[1:])
    return text


def run(
    model_id: str,
    method: MethodCandidate,
    previous_error: Exception | str | None = None,
    *,
    hyperparameters: dict | None = None,
    script_suffix: str | None = None,
) -> tuple[str, str]:
    """Run the Adapt ReAct loop. Returns (script_path, script_code).

    ``previous_error`` is set by the orchestrator on retry attempts so the agent
    sees its prior failure and can diagnose (e.g. pick a different install step
    or entry file) instead of repeating the same tool sequence.

    ``hyperparameters`` is the (possibly tune-locked) flat name->value dict the
    generated script must use. None or empty means use the method's defaults.

    ``script_suffix`` lets the orchestrator land successive tune iterations at
    distinct paths (e.g. ``_iter2``) so prior iterations' scripts aren't overwritten.
    """
    s = load_settings()

    if hyperparameters is None:
        hyperparameters = method.hyperparameters

    out_dir = s.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{script_suffix}" if script_suffix else ""
    script_path = out_dir / f"quantize_{_safe_slug(model_id)}_{method.id}{suffix}.py"

    session = ValidationSession(method_id=method.id)
    write_script = make_write_script_tool(session)

    @tool
    def rag_search_for_method(query: str, k: int = 6) -> str:
        """Search the quantization-literature index, scoped to the chosen method."""
        return rag_search.invoke({"query": query, "k": k, "method_id": method.id})

    tools = [
        github_readme,
        github_list_dir,
        github_file,
        clone_method_repo,
        install_method_venv,
        run_in_venv,
        list_repo_dir,
        read_repo_file,
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
        hyperparameters_block=_format_hyperparameters_block(hyperparameters),
        model_id=model_id,
        script_path=str(script_path),
    )
    agent = create_react_agent(llm, tools, prompt=prompt)

    user_msg = (
        f"Port {model_id} to {method.name} ({method.bits}-bit). "
        f"Clone the repo, build the venv, write the script to {script_path}."
    )
    if previous_error is not None:
        user_msg += (
            f"\n\nThis is a RETRY. The previous attempt failed with: {previous_error}. "
            "Diagnose the root cause and change your approach — do not repeat the "
            "same tool sequence."
        )

    agent.invoke(
        {"messages": [("user", user_msg)]},
        config={"recursion_limit": 60},
    )

    if not script_path.exists():
        raise RuntimeError(
            f"Adapt agent finished without writing {script_path}. Check the agent's last tool call."
        )

    code = _read_output_file(script_path)
    return str(script_path), code
