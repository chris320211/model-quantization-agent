#!/usr/bin/env python3
"""Launch a quantization script in its method venv, detached via setsid."""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import env
import jobs as jobs_mod
import overlay as overlay_mod
import paths
from paths import venv_python
from request import load_request


def default_output_dir(slug: str) -> str:
    """Canonical on-disk location for a run's quantized weights."""
    paths.require_slug(slug)
    return f"./quantized/{slug}"


def _allowed_overlay(detected: Path) -> bool:
    detected = detected.resolve()
    overlays = paths.OVERLAYS_ROOT.resolve()
    jobs_root = paths.JOBS_ROOT.resolve()
    from_generated = overlays == detected or overlays in detected.parents
    from_prior_job = (
        jobs_root in detected.parents
        and detected.name == "overlay"
        and jobs_mod.valid_job_id(detected.parent.name)
    )
    return from_generated or from_prior_job


def launch(
    *,
    request: dict,
    script_code: str,
    overlay_dir: Path | None = None,
    output_dir: str | None = None,
    parent_job_id: str | None = None,
    attempt: int = 1,
    fix_note: str | None = None,
    tune_iter: int = 0,
    hyperparameters: dict | None = None,
) -> jobs_mod.JobMeta:
    """Spawn the quantization script in its method venv, detached from the agent."""
    env.require_host_execution("quantization launch")
    if overlay_dir is None:
        raise RuntimeError("launch requires an overlay directory")
    slug = paths.require_slug(str(request["slug"]))
    model_id = str(request["model_id"])
    py = venv_python(slug)
    if not py.exists():
        raise RuntimeError(
            f"Venv python not found at {py}. Run install_venv for this slug first."
        )

    job_id = jobs_mod.new_job_id()
    job_directory = jobs_mod.job_dir(job_id)
    job_directory.mkdir(parents=True, exist_ok=True)

    script_path = job_directory / "script.py"
    script_path.write_text(script_code)

    overlay_snapshot: Path | None = None
    overlay_sha256: str | None = None
    overlay_manifest = None
    method_worktree: Path | None = None
    repo = Path(request["repo_path"]).expanduser().resolve()

    if overlay_dir is not None:
        detected_overlay = overlay_dir.expanduser().resolve()
        overlay_mod._assert_not_venv_write(detected_overlay)
        if not _allowed_overlay(detected_overlay):
            raise RuntimeError(
                "port overlay must come from generated output or a prior job snapshot: "
                f"{detected_overlay}"
            )
        overlay_manifest = overlay_mod.validate_overlay_bundle(detected_overlay)
        if overlay_manifest.method_id != slug or overlay_manifest.model_id != model_id:
            raise RuntimeError(
                "port overlay identity does not match launch request: "
                f"{overlay_manifest.method_id}/{overlay_manifest.model_id}"
            )
        overlay_snapshot = job_directory / "overlay"
        overlay_mod._assert_not_venv_write(overlay_snapshot)
        shutil.copytree(detected_overlay, overlay_snapshot)
        overlay_mod.validate_overlay_bundle(overlay_snapshot)
        overlay_sha256 = overlay_mod.directory_sha256(overlay_snapshot)
        method_worktree = overlay_mod.prepare_overlay_worktree(
            repo=repo,
            overlay_manifest=overlay_manifest,
            overlay_snapshot=overlay_snapshot,
            worktree=job_directory / "method-repo",
        )

    output = output_dir or default_output_dir(slug)
    command_argv = [str(py), str(script_path)]
    started_at = datetime.now(timezone.utc).isoformat()

    stdout = (job_directory / "stdout.log").open("wb")
    stderr = (job_directory / "stderr.log").open("wb")
    exit_sentinel = job_directory / "exit_code"

    quoted_command = " ".join(shlex.quote(arg) for arg in command_argv)
    if method_worktree is not None:
        cleanup = " ".join(shlex.quote(part) for part in [
            "git", "-C", str(repo), "worktree", "remove", "--force", str(method_worktree),
        ])
        wrapper = (
            f"{quoted_command}; code=$?; {cleanup}; "
            f"echo $code > {shlex.quote(str(exit_sentinel))}"
        )
    else:
        wrapper = f"{quoted_command}; echo $? > {shlex.quote(str(exit_sentinel))}"

    extra_env = None
    if overlay_snapshot is not None and method_worktree is not None:
        extra_env = {
            "QUANT_AGENT_OVERLAY_DIR": str(overlay_snapshot),
            "QUANT_AGENT_METHOD_REPO": str(method_worktree),
            "PYTHONPATH": str(method_worktree),
        }
    try:
        proc = subprocess.Popen(
            ["bash", "-c", wrapper],
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            cwd=str(paths.REPO_ROOT),
            start_new_session=True,  # setsid: survives SSH disconnect (SIGHUP)
            env=env.child_env(extra_env, include_hf=True),
        )
    except Exception:
        if method_worktree is not None:
            overlay_mod.cleanup_overlay_worktree(method_worktree, repo)
        raise
    finally:
        stdout.close()
        stderr.close()

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = proc.pid

    meta = jobs_mod.JobMeta(
        job_id=job_id,
        method_id=slug,
        model_id=model_id,
        venv=slug,
        script_path=str(script_path),
        output_dir=output,
        pid=proc.pid,
        started_at=started_at,
        pgid=pgid,
        parent_job_id=parent_job_id,
        attempt=attempt,
        fix_note=fix_note,
        tune_iter=tune_iter,
        hyperparameters=hyperparameters,
        execution_mode="host",
        overlay_path=str(overlay_snapshot) if overlay_snapshot else None,
        overlay_sha256=overlay_sha256,
        worktree_path=str(method_worktree) if method_worktree else None,
    )
    jobs_mod.write_meta(meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a quantization script as a detached job")
    parser.add_argument("script", type=Path)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--allow-unsafe-host-execution", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--parent-job-id")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--fix-note", help="Typed issue code from diagnose.json")
    args = parser.parse_args()

    payload = load_request(args.request)
    with env.host_execution_policy(args.allow_unsafe_host_execution):
        meta = launch(
            request=payload,
            script_code=args.script.read_text(),
            overlay_dir=args.overlay_dir,
            output_dir=args.output_dir,
            parent_job_id=args.parent_job_id,
            attempt=args.attempt,
            fix_note=args.fix_note,
        )
    print(meta.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
