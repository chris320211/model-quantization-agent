"""Adapt agent: clone the method's repo, build its venv, and author a script.

ReAct loop that:
  1. Clones the method repo into .venvs/<method_id>/repo/ and reads its README
     locally (full tree, no GitHub API rate limits).
  2. Builds a method-specific venv at .venvs/<method_id>/ and runs the repo's
     install steps (torch + transformers baseline installed automatically).
  3. Learns the TARGET model's exact architecture: full config.json
     (fetch_model_config) + the meta-device module tree (inspect_model_architecture,
     run in the method venv). This is what the quantizer's layer map is written against.
  4. Consults the method's source paper (read_paper) when the repo is thin.
  5. Inspects the cloned source locally to find an example entry point.
  6. Writes either a standalone script (importing the installed library) or a
     wrapper that subprocess-invokes the repo's example with filled-in args.
  7. Validates via ast.parse + dry-import (stdlib-only wrappers pass trivially).

Returns (script_path, script_code) — the orchestrator hands the code off to the
executor, which launches it with .venvs/<method_id>/bin/python.

When the tune loop calls back in with a non-empty ``hyperparameters`` dict, those
values are baked into the prompt as a TUNE-LOCKED block. The script must reflect
them, and downstream fix_agent invocations must not silently alter them.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from . import executor
from .config import load_settings
from .schemas import MethodCandidate
from .tools import (
    clone_method_repo,
    fetch_model_config,
    hf_model_info,
    install_method_venv,
    list_repo_dir,
    read_repo_file,
    run_in_venv,
)
from .tools.model_arch import inspect_architecture_core
from .tools.paper import read_paper_text
from .tools.recommender import load_catalog
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
  paper:     {arxiv_id}

Hyperparameters (TUNE-LOCKED — must appear verbatim in the generated script as the
configured values for this method's quantizer; do not substitute defaults). When this
block reads "(none — use method defaults)", omit explicit hyperparameter args and let
the method's own defaults apply.
{hyperparameters_block}

Target model: {model_id}
Output script path: {script_path}
Output model directory (save the quantized weights EXACTLY here — nowhere else): {output_dir}

Per-method paths (created by the tools below):
  venv:       .venvs/{method_id}/            (python at .venvs/{method_id}/bin/python)
  cloned repo: .venvs/{method_id}/repo/

Workflow — follow in order:

  1. clone_method_repo(method_id="{method_id}", repo_url="{repo_url}")
     Clones the FULL repo to .venvs/{method_id}/repo/ (idempotent). Then read the
     README locally to learn install + usage:
       read_repo_file(method_id="{method_id}", path="README.md")
     (try README.rst / docs/ if there is no README.md).

  2. Identify `install_steps` from the README. Examples:
        Research repos:     ["pip install -r requirements.txt", "pip install -e ."]
                            or ["pip install -e ."]
        Pip-packaged libs:  ["pip install autoawq"]   (for awq)
                            ["pip install gptqmodel datasets"]   (for gptq)
                            ["pip install hqq"]   (for hqq)
                            ["pip install bitsandbytes"]   (for bnb_nf4/bnb_llm_int8)
     A baseline of torch==2.3.1 (cu121) + transformers + accelerate + safetensors +
     sentencepiece is ALWAYS installed first. Do NOT duplicate those in install_steps.

  3. install_method_venv(method_id="{method_id}", install_steps=[...])
     Creates the venv and runs the steps. If it errors:
       - Read the last stderr lines carefully.
       - Adjust install_steps (add a missing dep, drop an optional broken step).
       - Retry ONCE. If it still fails, write a best-effort script anyway — the
         executor will surface the real error at runtime.

  4. Learn the TARGET model's architecture BEFORE writing the quantizer config:
       fetch_model_config(model_id="{model_id}")
         Full config.json — read num_key_value_heads (GQA), intermediate_size,
         tie_word_embeddings, MoE/rope fields, and trust_remote_code_required.
       inspect_model_architecture()
         Instantiates {model_id} on the meta device (no weight download) inside
         THIS method's venv and returns the exact module tree (Linear/Embedding
         names + shapes, collapsed across repeated layers). Use these REAL module
         names whenever the method needs a layer target/skip map — do not guess
         layer names. (Requires the venv from step 3; if trust_remote_code is
         required and not granted it returns a config-only summary.)

  5. If the repo's README/examples are thin or the algorithm/hyperparameters are
     unclear, consult the method's paper:
       read_paper(section="method")   (or read_paper() for the whole paper;
       other useful sections: "quantization", "experiments", "algorithm")

  6. Find the example entry point in the cloned repo:
       list_repo_dir(method_id="{method_id}", path="examples")
       list_repo_dir(method_id="{method_id}", path="scripts")
       list_repo_dir(method_id="{method_id}", path="")   (if needed)
     Then read_repo_file on 1–2 most-likely entry files to see their API.

  7. (Optional but recommended) run_in_venv(method_id="{method_id}",
        command="python <entry_path> --help") to confirm the entry is invokable
     and to discover the real arg flag names.

  8. Decide the script style and write it via
     write_script(path="{script_path}", code=...):

     STYLE A — Standalone import (preferred when the method has a stable Python API,
     e.g. autoawq, gptqmodel, hqq, bitsandbytes, or an importable module in the
     cloned repo). Write a single Python file that imports the library and runs
     quantization in-process. Must load the model from HF by id `{model_id}` and
     save output to EXACTLY this directory: {output_dir}

     STYLE B — Wrapper subprocess (preferred for research repos whose usage is
     "python examples/foo.py --model ..."). Write a Python file that:
        - import sys, os, subprocess, pathlib
        - Reads HF_TOKEN from env (HUGGINGFACE_HUB_TOKEN or HF_TOKEN) and passes it
          into the child env.
        - Computes REPO = pathlib.Path(".venvs/{method_id}/repo").resolve()
        - Builds the child env by forwarding ONLY PATH plus the HF token — do NOT
          splat the whole os.environ into the child:
            hf = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or ""
            child = {{"PATH": os.environ.get("PATH", "")}}
            if hf: child["HUGGINGFACE_HUB_TOKEN"] = child["HF_TOKEN"] = hf
        - subprocess.run([sys.executable, "<entry_path_relative_to_repo>",
                          "<flag_name>", "{model_id}", "<output_flag>", str(OUT)],
                          cwd=REPO, env=child, check=True)
        - OUT = pathlib.Path("{output_dir}").resolve()
        - The EXACT flag names MUST come from the --help output or the example source
          you read — do NOT invent flag names.

     Both styles must:
       - Quantize the real model at `{model_id}`.
       - Handle HF gated models: read HF_TOKEN from env and pass to every loader/
         tokenizer call (or login()) and into the child env for wrapper style.
       - trust_remote_code: pass trust_remote_code={trust_remote_code} to every
         from_pretrained / AutoConfig / AutoModel loader. This model's config
         {trust_hint}. Do not override what fetch_model_config reported.
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


def _catalog_arxiv_id(method_id: str) -> str | None:
    """Look up the method's arxiv_id from the catalog (read_paper is scoped to it)."""
    for m in load_catalog():
        if m.get("id") == method_id:
            aid = m.get("arxiv_id")
            return str(aid) if aid else None
    return None


def _safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")


def run(
    model_id: str,
    method: MethodCandidate,
    previous_error: Exception | str | None = None,
    *,
    hyperparameters: dict | None = None,
    script_suffix: str | None = None,
    output_dir: str | None = None,
    trust_remote_code: bool = False,
) -> tuple[str, str]:
    """Run the Adapt ReAct loop. Returns (script_path, script_code).

    ``previous_error`` is set by the orchestrator on retry attempts so the agent
    sees its prior failure and can diagnose (e.g. pick a different install step
    or entry file) instead of repeating the same tool sequence.

    ``hyperparameters`` is the (possibly tune-locked) flat name->value dict the
    generated script must use. None or empty means use the method's defaults.

    ``script_suffix`` lets the orchestrator land successive tune iterations at
    distinct paths (e.g. ``_iter2``) so prior iterations' scripts aren't overwritten.

    ``output_dir`` is the exact on-disk directory the generated script must save the
    quantized weights to. When None it defaults to the canonical location; the tune
    loop passes a per-iteration dir so iterations don't overwrite/prune each other.

    ``trust_remote_code`` gates execution of a model's custom modeling code (models
    whose config has ``auto_map``). Default False. When False, architecture
    introspection falls back to a config-only summary for such models and the
    generated script must not set ``trust_remote_code=True``.
    """
    s = load_settings()

    if hyperparameters is None:
        hyperparameters = method.hyperparameters
    if output_dir is None:
        output_dir = executor.default_output_dir(method.id, model_id)

    out_dir = s.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{script_suffix}" if script_suffix else ""
    script_path = out_dir / f"quantize_{_safe_slug(model_id)}_{method.id}{suffix}.py"
    attempt_path = out_dir / f".{script_path.name}.{secrets.token_hex(6)}.tmp"

    session = ValidationSession(method_id=method.id, allowed_root=out_dir)
    write_script = make_write_script_tool(session)

    arxiv_id = _catalog_arxiv_id(method.id)

    @tool
    def read_paper(section: str | None = None) -> str:
        """Read the chosen method's source paper (arXiv). Pass `section` (e.g. 'method',
        'quantization', 'experiments', 'algorithm') to focus, or omit for the whole paper.
        Use this when the repo README/examples don't make the API or hyperparameters clear."""
        return read_paper_text(arxiv_id, section=section)

    @tool
    def inspect_model_architecture() -> str:
        """Instantiate the target model on the meta device (no weight download) inside this
        method's venv and return its exact module tree — Linear/Embedding layer names and
        shapes, collapsed across repeated blocks. This is the ground truth for which layers
        the quantizer targets; call it AFTER install_method_venv and use the real names."""
        return inspect_architecture_core(
            model_id, method.id, trust_remote_code=trust_remote_code
        )

    tools = [
        clone_method_repo,
        install_method_venv,
        run_in_venv,
        list_repo_dir,
        read_repo_file,
        fetch_model_config,
        inspect_model_architecture,
        read_paper,
        hf_model_info,
        write_script,
    ]

    llm = ChatAnthropic(model=s.model, api_key=s.anthropic_api_key, temperature=0)
    trust_hint = (
        "ships custom modeling code (auto_map present), so trust_remote_code IS required"
        if trust_remote_code
        else "uses a standard transformers architecture, so trust_remote_code is NOT needed"
    )
    prompt = _PROMPT.format(
        method_id=method.id,
        method_name=method.name,
        repo_url=method.repo_url,
        bits=method.bits,
        arxiv_id=arxiv_id or "(none on file — rely on the repo)",
        hyperparameters_block=_format_hyperparameters_block(hyperparameters),
        model_id=model_id,
        script_path=str(attempt_path),
        output_dir=output_dir,
        trust_remote_code=trust_remote_code,
        trust_hint=trust_hint,
    )
    agent = create_react_agent(llm, tools, prompt=prompt)

    user_msg = (
        f"Port {model_id} to {method.name} ({method.bits}-bit). "
        f"Clone the repo, build the venv, write the script to {attempt_path}."
    )
    if previous_error is not None:
        user_msg += (
            f"\n\nThis is a RETRY. The previous attempt failed with: {previous_error}. "
            "Diagnose the root cause and change your approach — do not repeat the "
            "same tool sequence."
        )

    try:
        agent.invoke(
            {"messages": [("user", user_msg)]},
            config={"recursion_limit": 60},
        )
    except Exception:
        if attempt_path.exists():
            attempt_path.unlink()
        raise

    expected = attempt_path.resolve()
    if session.validated_path is None or session.validated_path.resolve() != expected:
        if attempt_path.exists():
            attempt_path.unlink()
        raise RuntimeError(
            "Adapt agent finished without producing a validated artifact in this session. "
            "Check the agent's final write_script result."
        )
    if not attempt_path.exists():
        raise RuntimeError(f"validated Adapt artifact disappeared before promotion: {attempt_path}")

    # Atomic promotion ensures a failed attempt cannot corrupt a previous good script.
    os.replace(attempt_path, script_path)
    code = script_path.read_text()
    return str(script_path), code
