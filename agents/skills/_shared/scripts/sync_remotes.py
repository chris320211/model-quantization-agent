#!/usr/bin/env python3
"""Rebuild the compare library, sync the Hub collection, push allowlisted git paths."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import os
import subprocess

from env import host_execution_policy, require_host_execution
from paths import COMPARE_ROOT, REPO_ROOT, contained_in

ALLOW_PREFIXES = ("compare/", "agents/", "tests/", ".github/workflows/")
ALLOW_FILES = ("README.md",)
BLOCK_PREFIXES = ("quantized/", "jobs/", "out/", ".venvs/", ".cache/")
BLOCK_NAMES = {".env", ".env.local"}
ASKPASS = Path(__file__).resolve().parent / "git_askpass.sh"
_GIT_TIMEOUT = 120

_DEFAULT_MESSAGE = "Sync the compare library to GitHub and Hugging Face."


def rel_posix(path: Path) -> str:
    resolved = path.expanduser().resolve()
    return resolved.relative_to(REPO_ROOT).as_posix()


def is_blocked(rel_path: str) -> bool:
    name = Path(rel_path).name
    if name in BLOCK_NAMES or name.startswith(".env"):
        return True
    return any(rel_path == p.rstrip("/") or rel_path.startswith(p) for p in BLOCK_PREFIXES)


def is_allowed(rel_path: str) -> bool:
    if is_blocked(rel_path) or ".." in Path(rel_path).parts:
        return False
    if rel_path in ALLOW_FILES:
        return True
    return any(rel_path == p.rstrip("/") or rel_path.startswith(p) for p in ALLOW_PREFIXES)


def _git(argv: list[str], *, env: dict[str, str] | None = None, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    require_host_execution("git")
    return subprocess.run(
        argv,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env or os.environ.copy(),
        check=False,
    )


def _push_env() -> dict[str, str]:
    env = os.environ.copy()
    if env.get("GITHUB_TOKEN") and ASKPASS.is_file():
        env["GIT_ASKPASS"] = str(ASKPASS)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GCM_INTERACTIVE"] = "Never"
    return env


def porcelain_paths() -> list[str]:
    proc = _git(["git", "status", "--porcelain", "-uall"])
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "git status failed").strip())
    paths: list[str] = []
    for line in (proc.stdout or "").splitlines():
        if len(line) < 4:
            continue
        raw = line[3:]
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        rel_path = raw.strip().strip('"')
        if rel_path:
            paths.append(rel_path)
    return paths


def stage_allowlisted() -> list[str]:
    staged: list[str] = []
    for rel_path in porcelain_paths():
        if not is_allowed(rel_path):
            continue
        abs_path = (REPO_ROOT / rel_path).resolve()
        if not contained_in(abs_path, REPO_ROOT) or is_blocked(rel_posix(abs_path)):
            raise RuntimeError(f"refusing to stage blocked path: {rel_path}")
        add = _git(["git", "add", "--", rel_path])
        if add.returncode != 0:
            raise RuntimeError((add.stderr or add.stdout or f"git add failed: {rel_path}").strip())
        staged.append(rel_path)
    return staged


def commit_and_push(*, message: str) -> dict:
    staged = stage_allowlisted()
    payload: dict = {"staged": staged, "commit": None, "pushed": False, "github": None}
    if staged:
        commit = _git(["git", "commit", "-m", message])
        if commit.returncode != 0:
            err = (commit.stderr or commit.stdout or "git commit failed").strip()
            raise RuntimeError(err)
        sha = _git(["git", "rev-parse", "HEAD"])
        payload["commit"] = (sha.stdout or "").strip() or None
    else:
        payload["commit_skipped"] = "nothing allowlisted to commit"

    ahead = _git(["git", "status", "-sb"])
    branch_line = (ahead.stdout or "").splitlines()[0] if ahead.stdout else ""
    needs_push = bool(staged) or "ahead" in branch_line
    if not needs_push:
        payload["push_skipped"] = "nothing to push"
        return payload

    push = _git(["git", "push", "-u", "origin", "HEAD"], env=_push_env(), timeout=_GIT_TIMEOUT)
    if push.returncode != 0:
        err = (push.stderr or push.stdout or "git push failed").strip()
        if os.environ.get("GITHUB_TOKEN"):
            err = err.replace(os.environ["GITHUB_TOKEN"], "***")
        raise RuntimeError(err)
    payload["pushed"] = True
    remote = _git(["git", "remote", "get-url", "origin"])
    url = (remote.stdout or "").strip().rstrip(".git")
    sha = payload["commit"] or (_git(["git", "rev-parse", "HEAD"]).stdout or "").strip()
    if url.startswith("https://github.com/") and sha:
        payload["github"] = f"{url}/commit/{sha}"
    else:
        payload["github"] = url or None
    return payload


def sync_remotes(*, push: bool, message: str, skip_hf: bool) -> dict:
    import compare as compare_mod

    payload: dict = {
        "status": "synced",
        "rebuild": compare_mod.rebuild_derived(),
        "hf_collection": None,
        "github": None,
        "commit": None,
    }
    library = COMPARE_ROOT / "library.json"
    if not skip_hf and (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        payload["hf_collection"] = compare_mod.sync_hf_collection()
    elif library.is_file():
        try:
            payload["hf_collection"] = {
                "status": "skipped",
                "hf_collection_url": json.loads(library.read_text()).get("hf_collection_url"),
            }
        except json.JSONDecodeError:
            payload["hf_collection"] = {"status": "skipped"}
    else:
        payload["hf_collection"] = {"status": "skipped"}

    if push:
        git_payload = commit_and_push(message=message)
        payload.update(git_payload)
    else:
        payload["push_skipped"] = "pass --push to commit allowlisted paths and git push"
        try:
            payload["would_stage"] = [p for p in porcelain_paths() if is_allowed(p)]
        except RuntimeError:
            payload["would_stage"] = []
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild compare/, sync the Hub collection, optionally git push allowlisted paths"
    )
    parser.add_argument("--push", action="store_true", help="Commit allowlisted paths and git push origin HEAD")
    parser.add_argument("--message", default=_DEFAULT_MESSAGE, help="git commit message")
    parser.add_argument("--skip-hf", action="store_true", help="Do not call Hugging Face collection sync")
    parser.add_argument("--allow-unsafe-host-execution", action="store_true")
    args = parser.parse_args()
    try:
        if args.push and not args.allow_unsafe_host_execution:
            raise RuntimeError("git push requires --allow-unsafe-host-execution")
        with host_execution_policy(args.allow_unsafe_host_execution):
            print(json.dumps(sync_remotes(push=args.push, message=args.message, skip_hf=args.skip_hf), indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
