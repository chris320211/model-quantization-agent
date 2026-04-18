from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import yaml
from huggingface_hub import HfApi

from ..config import REPO_ROOT, load_settings

log = logging.getLogger(__name__)

_ALIAS_PATH = REPO_ROOT / "seed" / "model_aliases.yaml"


@dataclass
class ResolveResult:
    model_id: str | None          # resolved canonical id, or None if ambiguous/unknown
    candidates: list[str]         # HfApi fallback suggestions (up to 3)
    source: str                   # "alias" | "hf_search" | "unresolved"


@lru_cache(maxsize=1)
def _load_aliases() -> dict[str, str]:
    if not _ALIAS_PATH.exists():
        return {}
    with _ALIAS_PATH.open() as f:
        raw = yaml.safe_load(f) or {}
    return {_normalize(k): v for k, v in raw.items()}


def _normalize(s: str) -> str:
    # lowercase, collapse all non-alphanumeric runs to single space
    return re.sub(r"[^a-z0-9.]+", " ", s.lower()).strip()


def _hf_search(query: str, limit: int = 3) -> list[str]:
    s = load_settings()
    try:
        api = HfApi(token=s.hf_token)
        results = api.list_models(search=query, limit=limit, sort="downloads", direction=-1)
        return [m.modelId for m in results]
    except Exception as e:  # noqa: BLE001 — HF Hub can fail for many reasons
        log.warning("HfApi.list_models search failed for %r: %s", query, e)
        return []


def resolve(name: str) -> ResolveResult:
    """Resolve a fuzzy model name to a canonical HuggingFace id.

    Order of preference:
      1. Exact (normalized) alias hit in seed/model_aliases.yaml.
      2. HfApi.list_models(search=name) — returns up to 3 candidates for the user to pick.
      3. Unresolved — empty candidates list.
    """
    norm = _normalize(name)
    aliases = _load_aliases()
    if norm in aliases:
        return ResolveResult(model_id=aliases[norm], candidates=[aliases[norm]], source="alias")

    hits = _hf_search(name)
    if hits:
        return ResolveResult(
            model_id=hits[0] if len(hits) == 1 else None,
            candidates=hits,
            source="hf_search",
        )
    return ResolveResult(model_id=None, candidates=[], source="unresolved")
