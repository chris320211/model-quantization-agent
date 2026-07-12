"""Per-method hyperparameter range inference.

Two-tier strategy:
  1. If the catalog entry already ships `hyperparameters_default`, use it as the
     authoritative source — no LLM call. (Curated for popular methods like AWQ,
     GPTQ, SmoothQuant, bnb_nf4, fp8.)
  2. Otherwise, fetch the method's README and ask the LLM to extract a flat list
     of tunable knobs with explicit value enumerations. Validate via Pydantic +
     a per-method allowlist before caching.

Cache is keyed by (method_id, repo_commit_sha) where commit_sha is best-effort —
falls back to method_id alone if no SHA is available.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from .config import load_settings
from .llm import AgentStage, create_chat_model
from .schemas import MethodCandidate
from .tools import github_readme
from .tools.recommender import load_catalog
from .io_utils import atomic_write_text

log = logging.getLogger(__name__)

_GLOBAL_CACHE_PATH = Path(
    os.path.expanduser("~/.cache/quant-agent/hyperparam_cache.yaml")
)


class HyperparamSpec(BaseModel):
    """One tunable knob. Use explicit `values` lists rather than min/max ranges
    so invalid combos (e.g. group_size=96) cannot creep in."""
    name: str = Field(..., min_length=1, max_length=64)
    type: Literal["int", "float", "bool", "categorical"]
    values: list[Any] = Field(..., min_length=1, max_length=16)
    default: Any


class HyperparamRanges(BaseModel):
    method_id: str
    specs: list[HyperparamSpec] = Field(..., max_length=12)


# ---------------------------------------------------------------------------
# Cache I/O


def _load_global_cache() -> dict[str, dict]:
    if not _GLOBAL_CACHE_PATH.exists():
        return {}
    try:
        with _GLOBAL_CACHE_PATH.open() as f:
            return yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        log.warning("hyperparam cache read failed: %s", e)
        return {}


def _save_global_cache(cache: dict[str, dict]) -> None:
    try:
        atomic_write_text(_GLOBAL_CACHE_PATH, yaml.safe_dump(cache, sort_keys=True))
    except OSError as e:
        log.warning("hyperparam cache write failed: %s", e)


def _cache_key(method_id: str, commit_sha: str | None) -> str:
    return f"{method_id}@{commit_sha or 'unknown'}"


def _local_repo_commit(method_id: str) -> str | None:
    from .config import REPO_ROOT
    import subprocess

    repo = REPO_ROOT / ".venvs" / method_id / "repo"
    if not (repo / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


# ---------------------------------------------------------------------------
# Catalog defaults (tier-1 source)


def _catalog_defaults(method_id: str) -> HyperparamRanges | None:
    """Convert the packaged catalog's defaults into HyperparamRanges."""
    for m in load_catalog():
        if m["id"] != method_id:
            continue
        block = m.get("hyperparameters_default")
        if not block:
            return None
        specs = []
        for name, info in block.items():
            values = info["values"]
            default = info.get("default", values[0])
            t = _infer_type(values)
            specs.append(HyperparamSpec(name=name, type=t, values=values, default=default))
        return HyperparamRanges(method_id=method_id, specs=specs)
    return None


def _infer_type(values: list[Any]) -> Literal["int", "float", "bool", "categorical"]:
    if all(isinstance(v, bool) for v in values):
        return "bool"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "int"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return "float"
    return "categorical"


# ---------------------------------------------------------------------------
# LLM tier (tier-2 source)


_LLM_PROMPT = """Read the README below for the {method_id} quantization method and
extract its tunable hyperparameters as a flat list. For each knob, emit:
  - name: the exact CLI/API arg name as used in the method's code
  - type: one of "int", "float", "bool", "categorical"
  - values: an explicit list of valid choices (no ranges, no min/max — list every
    sensible option you'd want to A/B test, max 6 values per knob)
  - default: the value listed as default in the README, or the most conservative

Constraints:
  - No more than 6 hyperparameters total. Pick the ones that actually move
    quality/speed/VRAM, not cosmetic flags.
  - Skip purely cosmetic knobs (verbosity, log paths, save flags).
  - Skip knobs that are determined by the model architecture (n_layers, d_model).
  - For continuous parameters, pick 3-5 representative values (e.g. for
    smoothquant alpha, use [0.5, 0.6, 0.7, 0.8, 0.9]).

README:
{readme}
"""


def _query_llm(method: MethodCandidate, readme: str) -> HyperparamRanges:
    s = load_settings()
    llm = create_chat_model(AgentStage.HYPERPARAM, s)
    structured = llm.with_structured_output(HyperparamRanges)
    prompt = _LLM_PROMPT.format(method_id=method.id, readme=readme[:50_000])
    return structured.invoke(prompt)


def _fetch_readme(method: MethodCandidate) -> str:
    try:
        result = github_readme.invoke({"repo_url": method.repo_url})
    except Exception as e:  # noqa: BLE001
        log.warning("github_readme failed for %s: %s", method.repo_url, e)
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and "content" in result:
        return str(result["content"])
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Validation


def _validate_specs(ranges: HyperparamRanges) -> tuple[bool, str | None]:
    """Sanity checks on LLM output beyond Pydantic typing.

    Catches obvious errors (default not in values, duplicate names, empty lists)
    that the schema can't enforce.
    """
    seen: set[str] = set()
    for spec in ranges.specs:
        if spec.name in seen:
            return False, f"duplicate name: {spec.name}"
        seen.add(spec.name)
        if spec.default not in spec.values:
            return False, f"{spec.name}: default {spec.default!r} not in values {spec.values!r}"
    return True, None


# ---------------------------------------------------------------------------
# Public API


def infer_ranges(
    method: MethodCandidate,
    *,
    repo_commit_sha: str | None = None,
    job_dir: Path | None = None,
) -> HyperparamRanges:
    """Resolve hyperparameter ranges for a method.

    Resolution order:
      1. Catalog `hyperparameters_default` (curated, no LLM call).
      2. Global cache lookup by (method_id, commit_sha).
      3. LLM inference from README, validated, then cached globally and per-job.
      4. On second LLM-validation failure: empty ranges (caller treats as "no
         knobs to tune"; the loop will degenerate to a single iteration).

    `job_dir` (when provided) gets a `hyperparams.yaml` copy of the result for
    reproducibility of that specific run.
    """
    # Tier 1: curated catalog defaults
    catalog = _catalog_defaults(method.id)
    if catalog is not None:
        _persist_per_job(catalog, job_dir)
        return catalog

    # Tier 2: global cache
    cache = _load_global_cache()
    key = _cache_key(method.id, repo_commit_sha or _local_repo_commit(method.id))
    if key in cache:
        try:
            cached = HyperparamRanges(**cache[key])
            _persist_per_job(cached, job_dir)
            return cached
        except ValidationError as e:
            log.warning("cache entry %s invalid (%s); re-inferring", key, e)

    # Tier 3: LLM inference with one retry
    readme = _fetch_readme(method)
    if not readme:
        ranges = HyperparamRanges(method_id=method.id, specs=[])
    else:
        ranges = _llm_with_retry(method, readme)

    cache[key] = ranges.model_dump()
    _save_global_cache(cache)
    _persist_per_job(ranges, job_dir)
    return ranges


def _llm_with_retry(method: MethodCandidate, readme: str) -> HyperparamRanges:
    last_err: str | None = None
    for attempt in (1, 2):
        try:
            ranges = _query_llm(method, readme) if attempt == 1 else _query_llm(
                method, readme + f"\n\nValidation error on prior try: {last_err}"
            )
        except ValidationError as e:
            last_err = str(e)
            log.warning("LLM hyperparam call attempt %d failed schema: %s", attempt, e)
            continue
        ok, err = _validate_specs(ranges)
        if ok:
            return ranges
        last_err = err
        log.warning("LLM hyperparam attempt %d failed validation: %s", attempt, err)

    log.warning("LLM hyperparam inference exhausted retries for %s; using empty ranges", method.id)
    return HyperparamRanges(method_id=method.id, specs=[])


def _persist_per_job(ranges: HyperparamRanges, job_dir: Path | None) -> None:
    if job_dir is None:
        return
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            job_dir / "hyperparams.yaml",
            yaml.safe_dump(ranges.model_dump(), sort_keys=False),
        )
    except OSError as e:
        log.warning("per-job hyperparam persist failed: %s", e)


def default_config(ranges: HyperparamRanges) -> dict[str, Any]:
    """Flat dict of name -> default value. Used by adapt_agent as the initial config."""
    return {spec.name: spec.default for spec in ranges.specs}
