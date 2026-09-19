"""Workspace paths for skill scripts. No quant_agent package."""
from __future__ import annotations

import os
import re
from pathlib import Path

_SKILLS_ROOT = Path(__file__).resolve().parents[1]  # .agents/skills/_shared
_REPO_CANDIDATE = _SKILLS_ROOT.parents[2]  # repo root from .agents/skills/_shared/scripts

REPO_ROOT = Path(os.environ.get("QUANT_AGENT_WORKSPACE", _REPO_CANDIDATE)).expanduser().resolve()
OUT_ROOT = REPO_ROOT / "out"
REQUESTS_ROOT = OUT_ROOT / "requests"
OVERLAYS_ROOT = OUT_ROOT / "overlays"
VENV_ROOT = REPO_ROOT / ".venvs"
JOBS_ROOT = REPO_ROOT / "jobs"
PAPER_CACHE = REPO_ROOT / ".cache" / "papers"
HF_SNAP_ROOT = REPO_ROOT / ".cache" / "hf-snapshots"

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")


def require_slug(slug: str) -> str:
    if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
        raise ValueError(f"invalid slug: {slug!r}")
    return slug


def contained_in(path: Path, root: Path) -> bool:
    resolved = path.expanduser().resolve()
    base = root.expanduser().resolve()
    return resolved == base or base in resolved.parents


def venv_python(slug: str) -> Path:
    require_slug(slug)
    py = (VENV_ROOT / slug / "bin" / "python").resolve()
    if not contained_in(py, VENV_ROOT):
        raise ValueError(f"venv python escaped .venvs: {py}")
    return py
