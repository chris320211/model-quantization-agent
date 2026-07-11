from __future__ import annotations

import base64
import json
import re

import requests
from langchain_core.tools import tool

from ..config import load_settings

_MAX_BODY = 12_000
_REPO_URL_RE = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/?$"
)


def _parse_owner_repo(repo_url: str) -> tuple[str, str]:
    match = _REPO_URL_RE.fullmatch(repo_url.strip())
    if match is None:
        raise ValueError(f"Could not parse repo URL: {repo_url}")
    return match.group("owner"), match.group("repo")


def _headers(accept: str = "application/vnd.github+json") -> dict:
    s = load_settings()
    h = {"Accept": accept}
    if s.github_token:
        h["Authorization"] = f"Bearer {s.github_token}"
    return h


def _truncate(text: str) -> str:
    if len(text) <= _MAX_BODY:
        return text
    return text[:_MAX_BODY] + "\n…[truncated]"


@tool
def github_readme(repo_url: str) -> str:
    """Fetch the raw README of a GitHub repository URL (e.g. 'https://github.com/mit-han-lab/llm-awq').

    Use this first when adapting a quantization repo — the README usually shows the
    intended API. Returns raw markdown (truncated to 12 KB).
    """
    try:
        owner, repo = _parse_owner_repo(repo_url)
    except ValueError as e:
        return str(e)
    api = f"https://api.github.com/repos/{owner}/{repo}/readme"
    try:
        r = requests.get(api, headers=_headers("application/vnd.github.raw"), timeout=20)
    except requests.RequestException as e:
        return f"GitHub request failed: {e}"
    if r.status_code != 200:
        return f"GitHub returned {r.status_code} for {repo_url}"
    return _truncate(r.text)


@tool
def github_list_dir(repo_url: str, path: str = "") -> str:
    """List files and directories at a path inside a GitHub repo.

    Use after reading the README to discover quantization entry points
    (typical names: quantize.py, main.py, examples/, auto_gptq/, awq/quantize/).

    Args:
        repo_url: e.g. 'https://github.com/casper-hansen/AutoAWQ'.
        path:     Path within the repo; empty string = repo root.

    Returns a JSON list of {name, type, path, size} entries, or an error string.
    """
    try:
        owner, repo = _parse_owner_repo(repo_url)
    except ValueError as e:
        return str(e)
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{path.lstrip('/')}"
    try:
        r = requests.get(api, headers=_headers(), timeout=20)
    except requests.RequestException as e:
        return f"GitHub request failed: {e}"
    if r.status_code != 200:
        return f"GitHub returned {r.status_code} for {repo_url}/{path}"
    data = r.json()
    if isinstance(data, dict):
        return f"Path is a file, not a directory: {path}. Use github_file to fetch it."
    entries = [
        {
            "name": e.get("name"),
            "type": e.get("type"),
            "path": e.get("path"),
            "size": e.get("size"),
        }
        for e in data
    ]
    return json.dumps(entries, indent=2)


@tool
def github_file(repo_url: str, path: str) -> str:
    """Fetch a specific file from a GitHub repo (text only, 12 KB truncation).

    Use this on 1-3 files that look like quantization entry points — NOT
    indiscriminately. Binary files return an error.
    """
    try:
        owner, repo = _parse_owner_repo(repo_url)
    except ValueError as e:
        return str(e)
    if not path:
        return "path must be non-empty"
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{path.lstrip('/')}"
    try:
        r = requests.get(api, headers=_headers(), timeout=20)
    except requests.RequestException as e:
        return f"GitHub request failed: {e}"
    if r.status_code != 200:
        return f"GitHub returned {r.status_code} for {repo_url}/{path}"
    data = r.json()
    if isinstance(data, list):
        return f"Path is a directory, not a file: {path}. Use github_list_dir."
    if data.get("encoding") != "base64":
        return f"Unsupported encoding: {data.get('encoding')}"
    try:
        raw = base64.b64decode(data["content"])
        text = raw.decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return f"File {path} appears to be binary; skipping."
    return _truncate(text)
