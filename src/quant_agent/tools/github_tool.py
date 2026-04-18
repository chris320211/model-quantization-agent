from __future__ import annotations

import requests
from langchain_core.tools import tool

from ..config import load_settings


@tool
def github_readme(repo_url: str) -> str:
    """Fetch the raw README of a GitHub repository URL (e.g. 'https://github.com/mit-han-lab/llm-awq').

    Use this when the user names a repo not in the catalog, or to cross-reference API details
    for generating a quantization script.
    """
    s = load_settings()
    try:
        parts = repo_url.rstrip("/").replace("https://github.com/", "").split("/")
        owner, repo = parts[0], parts[1]
    except (IndexError, ValueError):
        return f"Could not parse repo URL: {repo_url}"
    api = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {"Accept": "application/vnd.github.raw"}
    if s.github_token:
        headers["Authorization"] = f"Bearer {s.github_token}"
    try:
        r = requests.get(api, headers=headers, timeout=20)
    except requests.RequestException as e:
        return f"GitHub request failed: {e}"
    if r.status_code != 200:
        return f"GitHub returned {r.status_code} for {repo_url}"
    text = r.text
    if len(text) > 12000:
        text = text[:12000] + "\n…[README truncated]"
    return text
