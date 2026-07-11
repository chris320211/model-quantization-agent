"""Staged Adapt pipeline: acquire, plan, build, inspect, author, and validate.

Bounded stages:
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
  7. Validates syntax, imports, exact model/output references, and tune locks.

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
from .adapt_stages import AdaptPlanSession, AdaptTrace, make_write_adapt_plan_tool
from .config import load_settings
from .schemas import MethodCandidate
from .tools import (
    clone_method_repo,
    fetch_model_config,
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

_PLAN_PROMPT = """You are the repository-planning stage of a quantization adapter.
The repository has already been cloned. Inspect its README and the minimum relevant
source files, then call write_adapt_plan exactly once.

Method: {method_name} ({method_id})
Repository: .venvs/{method_id}/repo/
Target model: {model_id}

Determine only:
1. method-specific pip/python install steps (baseline Torch/Transformers packages are
   installed separately; never repeat them),
2. whether the final script should use a stable standalone Python API or wrap a real
   repository entrypoint,
3. the repository-relative entrypoint when wrapper style is selected,
4. the files that prove these choices.

Commands must be accepted by the restricted installer: python/python3/pip/pip3 only;
no shell operators, redirects, curl, wget, arbitrary executables, or environment
assignments other than TORCH_CUDA_ARCH_LIST/MAX_JOBS. Do not install or execute code in
this stage. Stop after write_adapt_plan returns status=ok.
"""

_AUTHOR_PROMPT = """You are the script-authoring stage of a quantization adapter.
Acquisition, environment construction, and model architecture inspection have already
completed. Do not repeat them. Use the repository-derived plan and architecture facts
below to write one executable script via write_script.

Method: {method_name} ({method_id}), {bits}-bit
Repository: .venvs/{method_id}/repo/
Paper: {arxiv_id}
Target model: {model_id}
Output script: {script_path}
Exact output model directory: {output_dir}
trust_remote_code={trust_remote_code}

Repository plan:
{adapt_plan}

Model config and architecture evidence:
{architecture_evidence}

Tune-locked hyperparameters:
{hyperparameters_block}

For standalone style, use the real installed API verified in repository source. For
wrapper style, invoke only the plan's repository-relative entrypoint with exact flags
verified from source or `python <entrypoint> --help`. Forward only PATH and an optional
HF token to child processes. Every Hugging Face loader must use the target model and
trust_remote_code value above. Save weights exactly to the requested output directory,
print final on-disk size, and include tune-locked values in code and header comments.
Stop immediately after write_script returns status=ok.
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


def _tool_payload(raw: str, operation: str) -> dict:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{operation} returned invalid JSON") from exc
    if payload.get("status") != "ok":
        raise RuntimeError(f"{operation} failed: {payload.get('error') or payload}")
    return payload


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

    arxiv_id = _catalog_arxiv_id(method.id)
    trace = AdaptTrace(model_id=model_id, method_id=method.id)
    trace_path = script_path.with_suffix(".adapt.json")
    trace.record("prepare", "completed", script_path=str(script_path), output_dir=output_dir)

    @tool
    def read_paper(section: str | None = None) -> str:
        """Read the chosen method's source paper (arXiv). Pass `section` (e.g. 'method',
        'quantization', 'experiments', 'algorithm') to focus, or omit for the whole paper.
        Use this when the repo README/examples don't make the API or hyperparameters clear."""
        return read_paper_text(arxiv_id, section=section)

    llm = ChatAnthropic(model=s.model, api_key=s.anthropic_api_key, temperature=0)
    try:
        trace.record("acquire", "started")
        clone = _tool_payload(clone_method_repo.invoke({
            "method_id": method.id, "repo_url": method.repo_url,
        }), "repository acquisition")
        trace.record(
            "acquire", "completed", commit_sha=clone.get("commit_sha"),
            already_present=clone.get("already_present", False),
        )

        trace.record("plan", "started")
        plan_session = AdaptPlanSession()
        write_adapt_plan = make_write_adapt_plan_tool(plan_session)
        plan_agent = create_react_agent(
            llm,
            [list_repo_dir, read_repo_file, read_paper, write_adapt_plan],
            prompt=_PLAN_PROMPT.format(
                method_id=method.id, method_name=method.name, model_id=model_id,
            ),
        )
        plan_message = "Inspect the cloned repository and finalize the Adapt plan."
        if previous_error is not None:
            plan_message += (
                f" Previous attempt failed with {previous_error}; choose a materially "
                "different repository-supported plan."
            )
        plan_agent.invoke(
            {"messages": [("user", plan_message)]},
            config={"recursion_limit": 60},
        )
        if plan_session.plan is None:
            raise RuntimeError("planning stage finished without write_adapt_plan")
        plan = plan_session.plan
        trace.record("plan", "completed", plan=plan.model_dump())

        trace.record("environment", "started")
        install = _tool_payload(install_method_venv.invoke({
            "method_id": method.id,
            "install_steps": plan.install_steps,
        }), "environment construction")
        trace.record("environment", "completed", python=install.get("python"))

        trace.record("architecture", "started")
        config_raw = fetch_model_config.invoke({"model_id": model_id})
        try:
            config_payload = json.loads(config_raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("model config stage returned invalid JSON") from exc
        if config_payload.get("error"):
            raise RuntimeError(f"model config stage failed: {config_payload['error']}")
        architecture_raw = inspect_architecture_core(
            model_id, method.id, trust_remote_code=trust_remote_code
        )
        try:
            architecture_payload = json.loads(architecture_raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("architecture stage returned invalid JSON") from exc
        trace.record(
            "architecture", "completed",
            architectures=config_payload.get("architectures"),
            trust_remote_code_required=config_payload.get("trust_remote_code_required"),
            introspection_status=architecture_payload.get("status"),
        )

        trace.record("generate", "started")
        session = ValidationSession(
            method_id=method.id,
            allowed_root=out_dir,
            expected_model_id=model_id,
            expected_output_dir=output_dir,
            locked_hyperparameters=hyperparameters,
        )
        write_script = make_write_script_tool(session)
        architecture_evidence = json.dumps({
            "config": config_payload,
            "module_tree": architecture_payload,
        }, indent=2)
        author_agent = create_react_agent(
            llm,
            [run_in_venv, list_repo_dir, read_repo_file, read_paper, write_script],
            prompt=_AUTHOR_PROMPT.format(
                method_id=method.id,
                method_name=method.name,
                bits=method.bits,
                arxiv_id=arxiv_id or "(none)",
                model_id=model_id,
                script_path=str(attempt_path),
                output_dir=output_dir,
                trust_remote_code=trust_remote_code,
                adapt_plan=plan.model_dump_json(indent=2),
                architecture_evidence=architecture_evidence,
                hyperparameters_block=_format_hyperparameters_block(hyperparameters),
            ),
        )
        author_agent.invoke(
            {"messages": [("user", "Author and validate the quantization script now.")]},
            config={"recursion_limit": 40},
        )
        trace.record("generate", "completed")

        expected = attempt_path.resolve()
        if session.validated_path is None or session.validated_path.resolve() != expected:
            raise RuntimeError(
                "authoring stage finished without producing a validated artifact in this session"
            )
        if not attempt_path.exists():
            raise RuntimeError(f"validated Adapt artifact disappeared: {attempt_path}")
        trace.record("validate", "completed", path=str(expected))

        os.replace(attempt_path, script_path)
        trace.record("promote", "completed", path=str(script_path))
        trace.persist(trace_path)
    except Exception as exc:
        # A failed run never promotes a script, but its bounded-stage trace remains
        # available next to the intended output for diagnosis and reproducibility.
        failed_stage = next(
            (r.name for r in reversed(trace.stages) if r.status == "started"), "prepare"
        )
        trace.record(failed_stage, "failed", error_type=type(exc).__name__, error=str(exc))
        trace.persist(trace_path)
        if attempt_path.exists():
            attempt_path.unlink()
        raise
    code = script_path.read_text()
    return str(script_path), code
