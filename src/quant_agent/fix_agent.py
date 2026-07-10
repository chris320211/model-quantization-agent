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

Prior repair attempts on this failure chain (oldest first):
{prior_attempts_block}

Workflow — follow in order:

  1. read_job_logs(job_id="{job_id}") — read the stdout + stderr tail. Identify the
     failing step and the root-cause error line.

  1b. If you plan to call edit_script, call read_script(job_id="{job_id}") FIRST
      so you know the exact text in the script. edit_script's `old` arg must match
      the file's content exactly once — guessing from the traceback is unreliable.

  2. Classify the failure and pick ONE fix — it must be DIFFERENT from every fix
     listed under "Prior repair attempts" above (those already failed):
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

  3. If you applied a fix in step 2, call
     relaunch_job(job_id="{job_id}", fix_description="<one sentence: what you changed>").
     This launches a NEW job with parent_job_id="{job_id}" and terminates this loop.
     The fix_description is recorded so the next repair attempt (if any) knows what
     was already tried — make it specific (package + version, exact flag change).

Apply exactly one fix per attempt. Prefer venv surgery (run_in_venv /
install_method_venv) over script edits. If unsure, inspect the cloned repo via
list_repo_dir / read_repo_file / github_readme before editing.

Stop as soon as relaunch_job returns status="ok" — do not continue iterating.
"""


def _format_prior_attempts(
    prior_attempts: list[dict] | None, same_error: bool
) -> str:
    """Render the repair history block for the prompt.

    Each entry describes one earlier relaunch: the fix that was applied (the
    agent's own fix_description, persisted as JobMeta.fix_note), the job it
    produced, and how that job ended. ``same_error`` is set by the supervisor
    when the latest failed job's root error is identical to its parent's —
    i.e. the last fix demonstrably changed nothing.
    """
    if not prior_attempts:
        return "  (none — this is the first repair attempt on this chain)"
    lines: list[str] = []
    for i, a in enumerate(prior_attempts, start=1):
        fix = a.get("fix") or "(no fix description recorded)"
        outcome = a.get("status") or "failed"
        if a.get("exit_code") is not None:
            outcome += f" (exit {a['exit_code']})"
        lines.append(f"  {i}. fix applied: {fix}")
        lines.append(f"     -> relaunched as job {a.get('job_id', '?')}: {outcome}")
        if a.get("error_line"):
            lines.append(f"     root error: {a['error_line']}")
    if same_error:
        lines.append("")
        lines.append(
            "  WARNING: the most recent fix did NOT change the failure — the relaunched"
        )
        lines.append(
            "  job died with the SAME root error as its parent. Do NOT repeat that fix."
        )
        lines.append(
            "  Diagnose deeper (different package/version, different entry point, build"
        )
        lines.append(
            "  the CUDA extension) or classify the failure as non-retryable and stop."
        )
    return "\n".join(lines)


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
    prior_attempts: list[dict] | None = None,
    same_error: bool = False,
) -> str | None:
    """Run the Fix ReAct loop. Returns the new job_id on a successful relaunch, else None.

    ``prior_attempts`` carries the repair history for this failure chain (one dict
    per earlier relaunch: fix, job_id, status, exit_code, error_line) so the agent
    doesn't re-apply a fix that already failed. ``same_error`` flags that the latest
    relaunch died with the identical root error as its parent.
    """
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
        prior_attempts_block=_format_prior_attempts(prior_attempts, same_error),
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
