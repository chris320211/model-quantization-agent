"""Fix agent: diagnose a runtime-failed quantization job and relaunch it.

Mirrors ``adapt_agent.py``: a ReAct loop over ``run_in_venv``, ``install_method_venv``,
repo-inspection tools, and three job-scoped tools (``read_job_logs``, ``edit_script``,
``relaunch_job``). The loop terminates when ``relaunch_job`` succeeds, at which point
this module returns the new ``job_id`` for the orchestrator's supervisor to wait on.

If the agent classifies the failure as non-retryable (HF auth, gated-model access,
OOM on a too-small GPU) it returns ``None`` without relaunching.
"""
from __future__ import annotations

import json
import logging

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from . import executor
from .config import load_settings
from .schemas import MethodCandidate
from .tools import (
    edit_script,
    github_readme,
    install_method_venv,
    list_repo_dir,
    read_job_logs,
    read_repo_file,
    read_script,
    relaunch_job,
    run_in_venv,
)

log = logging.getLogger(__name__)

_PROMPT = """You are the Fix agent. A quantization job failed at runtime. Diagnose the
failure from its logs, apply ONE targeted fix, and relaunch the same script on the same
method. You must NOT switch methods — the user already chose {method_name}.

Failed job:
  job_id:     {job_id}
  method_id:  {method_id}      venv: .venvs/{method_id}/   (python at .venvs/{method_id}/bin/python)
  model_id:   {model_id}
  script:     {script_path}
  attempt:    {attempt} of {max_attempts}

Workflow — follow in order:

  1. read_job_logs(job_id="{job_id}") — read the stdout + stderr tail. Identify the
     failing step and the root-cause error line.

  1b. If you plan to call edit_script, call read_script(job_id="{job_id}") FIRST
      so you know the exact text in the script. edit_script's `old` arg must match
      the file's content exactly once — guessing from the traceback is unreliable.

  2. Classify the failure and pick ONE fix:
       - ImportError / ModuleNotFoundError → `install_method_venv` with a targeted
         pip step (e.g. ["pip install transformers==4.46.3"]) or `run_in_venv`
         with `pip install <pkg>`.
       - Version conflict ("requires X but found Y") → pin via `install_method_venv`.
       - Missing CUDA extension (e.g. "No module named 'awq_inference_engine'") →
         `run_in_venv` with `cd <subdir> && python setup.py install`. Set
         TORCH_CUDA_ARCH_LIST to the GPU's compute capability if the kernel needs it.
       - Wrong script flag / arg mismatch → `edit_script(job_id, old, new)` to patch
         the saved script. `old` must match exactly once.
       - Non-retryable (HF 401/403 on a gated model, OOM on too-small VRAM, missing
         HF_TOKEN, disk full) → STOP. Do NOT call relaunch_job. End your turn with
         a one-sentence explanation of why this is unfixable so the orchestrator
         can surface it to the user.

  3. If you applied a fix in step 2, call relaunch_job(job_id="{job_id}"). This
     launches a NEW job with parent_job_id="{job_id}" and terminates this loop.

Apply exactly one fix per attempt. Prefer venv surgery (run_in_venv /
install_method_venv) over script edits. If unsure, inspect the cloned repo via
list_repo_dir / read_repo_file / github_readme before editing.

Stop as soon as relaunch_job returns status="ok" — do not continue iterating.
"""


def _extract_new_job_id(final_state: dict) -> str | None:
    """Walk the ReAct final state to find the relaunch_job tool result, if any."""
    for msg in reversed(final_state.get("messages", [])):
        name = getattr(msg, "name", None)
        if name != "relaunch_job":
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if payload.get("status") == "ok" and payload.get("new_job_id"):
            return payload["new_job_id"]
    return None


def run(
    job_id: str,
    method: MethodCandidate,
    model_id: str,
    attempt: int,
    max_attempts: int,
    script_path: str | None = None,
) -> str | None:
    """Run the Fix ReAct loop. Returns the new job_id on a successful relaunch, else None."""
    s = load_settings()

    meta = executor.refresh_status(job_id)
    resolved_script_path = script_path or meta.script_path

    tools = [
        read_job_logs,
        read_script,
        run_in_venv,
        install_method_venv,
        list_repo_dir,
        read_repo_file,
        github_readme,
        edit_script,
        relaunch_job,
    ]

    prompt = _PROMPT.format(
        job_id=job_id,
        method_id=method.id,
        method_name=method.name,
        model_id=model_id,
        script_path=resolved_script_path,
        attempt=attempt,
        max_attempts=max_attempts,
    )

    llm = ChatAnthropic(model=s.model, api_key=s.anthropic_api_key, temperature=0)
    agent = create_react_agent(llm, tools, prompt=prompt)

    final_state = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Job {job_id} ({method.name} / {model_id}) failed on attempt "
                    f"{attempt}. Diagnose and repair.",
                )
            ]
        },
        config={"recursion_limit": 40},
    )

    return _extract_new_job_id(final_state)
