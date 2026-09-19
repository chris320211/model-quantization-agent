#!/usr/bin/env python3
"""Clone a GitHub repo into .venvs/<slug>/repo."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
from contextlib import contextmanager

from env import child_env, host_execution_policy, require_host_execution
from paths import VENV_ROOT, require_slug

_GITHUB_URL_RE = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_CLONE_TIMEOUT = 300
_GIT_TIMEOUT = 15


def _normalize_url(repo_url: str) -> str:
    return repo_url.strip().rstrip("/")


def _validate_slug(slug: str) -> str | None:
    if not _SLUG_RE.fullmatch(slug):
        return f"invalid slug: {slug!r}"
    return None


@contextmanager
def _slug_lock(slug: str):
    lock_dir = VENV_ROOT / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{slug}.lock"
    with lock_path.open("a+") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _git(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    require_host_execution("git")
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=child_env(include_hf=False),
    )


def _head_sha(repo: Path) -> str | None:
    head = _git(["git", "-C", str(repo), "rev-parse", "HEAD"], _GIT_TIMEOUT)
    sha = (head.stdout or "").strip().lower()
    if head.returncode != 0 or not _SHA_RE.fullmatch(sha):
        return None
    return sha


def clone_repo(slug: str, repo_url: str) -> dict:
    try:
        require_slug(slug)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}

    raw = repo_url.strip()
    if not _GITHUB_URL_RE.match(raw):
        return {
            "status": "error",
            "error": f"repo_url is not a plain GitHub repo URL: {repo_url!r}",
        }
    expected = _normalize_url(raw)

    with _slug_lock(slug):
        venv_dir = VENV_ROOT / slug
        repo = venv_dir / "repo"
        venv_dir.mkdir(parents=True, exist_ok=True)

        if repo.exists() and (repo / ".git").exists():
            origin = _git(["git", "-C", str(repo), "remote", "get-url", "origin"], _GIT_TIMEOUT)
            actual = _normalize_url((origin.stdout or "").strip())
            if origin.returncode != 0 or actual != expected:
                return {
                    "status": "error",
                    "error": f"existing checkout origin mismatch: expected {expected!r}, got {actual!r}",
                }
            sha = _head_sha(repo)
            if sha is None:
                return {"status": "error", "error": "existing checkout HEAD is not a 40-char SHA"}
            return {
                "status": "ok",
                "path": str(repo),
                "commit_sha": sha,
                "repo_url": expected,
                "already_present": True,
            }

        if repo.exists():
            shutil.rmtree(repo)

        try:
            result = _git(
                ["git", "-c", "core.symlinks=false", "clone", "--depth", "1", raw, str(repo)],
                _CLONE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"git clone timed out after {_CLONE_TIMEOUT}s"}
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "clone failed").strip()
            return {"status": "error", "error": err}

        origin = _git(["git", "-C", str(repo), "remote", "get-url", "origin"], _GIT_TIMEOUT)
        actual = _normalize_url((origin.stdout or "").strip())
        if origin.returncode != 0 or actual != expected:
            return {
                "status": "error",
                "error": f"cloned origin mismatch: expected {expected!r}, got {actual!r}",
            }
        sha = _head_sha(repo)
        if sha is None:
            return {"status": "error", "error": "cloned HEAD is not a 40-char SHA"}
        return {
            "status": "ok",
            "path": str(repo),
            "commit_sha": sha,
            "repo_url": expected,
            "already_present": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone a GitHub repo into .venvs/<slug>/repo")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--allow-unsafe-host-execution", action="store_true")
    args = parser.parse_args()
    try:
        with host_execution_policy(args.allow_unsafe_host_execution):
            payload = clone_repo(args.slug, args.repo_url)
    except Exception as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
