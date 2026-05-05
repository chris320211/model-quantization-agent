"""Cross-run tuning history. Append-only JSONL at ~/.cache/quant-agent/tune_history.jsonl.

Each entry records a *successful* tune iteration so future runs on the same
(model, instance, method) tuple can warm-start from a known-good config.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_CACHE_DIR = Path(os.path.expanduser("~/.cache/quant-agent"))
_HISTORY_PATH = _CACHE_DIR / "tune_history.jsonl"


@dataclass(frozen=True)
class HistoryEntry:
    model_id: str
    instance_type: str | None
    method_id: str
    hyperparameters: dict
    metrics: dict
    timestamp: str
    note: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _sanitize(d: dict) -> dict:
    """Strip absolute paths and HOME-leaking values before persistence."""
    home = os.path.expanduser("~")

    def _scrub(v):
        if isinstance(v, str):
            return v.replace(home, "~")
        if isinstance(v, dict):
            return {k: _scrub(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_scrub(x) for x in v]
        return v

    return _scrub(d)


def append(
    *,
    model_id: str,
    instance_type: str | None,
    method_id: str,
    hyperparameters: dict,
    metrics: dict,
    note: str | None = None,
) -> HistoryEntry:
    """Record one tuning result. Failure-tolerant: returns the entry even if
    the file write fails (logged), so loop progress isn't blocked by disk issues.
    """
    entry = HistoryEntry(
        model_id=model_id,
        instance_type=instance_type,
        method_id=method_id,
        hyperparameters=_sanitize(hyperparameters),
        metrics=_sanitize(metrics),
        timestamp=datetime.now(timezone.utc).isoformat(),
        note=note,
    )
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with _HISTORY_PATH.open("a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
    except OSError as e:
        log.warning("tune_history append failed: %s", e)
    return entry


def _read_all() -> list[HistoryEntry]:
    if not _HISTORY_PATH.exists():
        return []
    entries: list[HistoryEntry] = []
    with _HISTORY_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                entries.append(HistoryEntry(**d))
            except (json.JSONDecodeError, TypeError) as e:
                log.warning("skipping malformed history line: %s", e)
    return entries


def query_prior_wins(
    *,
    model_id: str,
    instance_type: str | None,
    method_id: str,
    limit: int = 5,
) -> list[HistoryEntry]:
    """Most-recent-first prior wins for the same (model, instance, method) tuple.

    Used as warm-start hints fed into the tune_agent prompt. Family-level fallback
    (e.g. "llama-2-7b" matching "llama-2-13b") is intentionally NOT done here —
    different param counts have different optimal configs; cross-family transfer
    is the LLM's job to reason about, not a heuristic file lookup's.
    """
    matches = [
        e for e in _read_all()
        if e.model_id == model_id
        and e.instance_type == instance_type
        and e.method_id == method_id
    ]
    matches.sort(key=lambda e: e.timestamp, reverse=True)
    return matches[:limit]


def history_path() -> Path:
    """Expose for tests and CLI 'show history' commands."""
    return _HISTORY_PATH


_FAMILY_RE = re.compile(r"^(?P<org>[^/]+)/(?P<name>.+)$")


def model_family(model_id: str) -> str | None:
    """Best-effort family slug, e.g. 'meta-llama/Llama-2-7b-hf' -> 'llama-2'.

    Used for diagnostic comparison only. Never used as a primary cache key —
    different param sizes inside the same family tune to different configs.
    """
    m = _FAMILY_RE.match(model_id)
    if not m:
        return None
    name = m.group("name").lower()
    name = re.sub(r"-?(7b|13b|70b|405b|1\.1b|1b|3b)$", "", name)
    name = re.sub(r"-(hf|chat|instruct|base)$", "", name)
    return name or None
