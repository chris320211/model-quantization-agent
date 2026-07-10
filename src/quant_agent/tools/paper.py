"""arXiv paper fetch + cache for the Adapt agent's ``read_paper`` tool.

This replaces the RAG paper channel. Instead of similarity-retrieving 6 chunks,
the Adapt agent reads the *whole* paper (or a named section) for the chosen
method — which is what porting a novel/under-documented method actually needs.

Source priority: ar5iv HTML (clean section structure) -> arXiv PDF (pypdf text).
Fetched text is cached under ``.cache/papers/<arxiv_id>.txt`` so repeated adapt /
tune iterations don't re-download.
"""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import requests

from ..config import REPO_ROOT

log = logging.getLogger(__name__)

_CACHE_DIR = REPO_ROOT / ".cache" / "papers"
_HTTP_TIMEOUT = 60
_DEFAULT_MAX_CHARS = 16_000
_UA = {"User-Agent": "quant-agent/0.1 (+https://github.com/)"}


def _cache_path(arxiv_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", arxiv_id)
    return _CACHE_DIR / f"{safe}.txt"


def _strip_html(html: str) -> str:
    """Coarse HTML -> text. ar5iv markup is clean enough that this reads well."""
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|h[1-6]|li|section|tr)>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", "", text)
    for ent, ch in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&#39;", "'")):
        text = text.replace(ent, ch)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fetch_ar5iv(arxiv_id: str) -> str | None:
    url = f"https://ar5iv.org/abs/{arxiv_id}"
    try:
        r = requests.get(url, timeout=_HTTP_TIMEOUT, headers=_UA)
    except requests.RequestException as e:  # noqa: BLE001
        log.warning("ar5iv fetch failed for %s: %s", arxiv_id, e)
        return None
    if r.status_code == 200 and "<html" in r.text.lower():
        return _strip_html(r.text) or None
    return None


def _fetch_pdf_text(arxiv_id: str) -> str | None:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        r = requests.get(url, timeout=_HTTP_TIMEOUT, headers=_UA)
    except requests.RequestException as e:  # noqa: BLE001
        log.warning("arXiv PDF fetch failed for %s: %s", arxiv_id, e)
        return None
    if r.status_code != 200:
        log.warning("arXiv PDF %s: HTTP %s", arxiv_id, r.status_code)
        return None
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(r.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() or None
    except Exception as e:  # noqa: BLE001 — pypdf raises a zoo of errors on odd PDFs
        log.warning("arXiv PDF parse failed for %s: %s", arxiv_id, e)
        return None


def fetch_paper_text(arxiv_id: str, *, use_cache: bool = True) -> str | None:
    """Return full plain text of an arXiv paper (cached to disk). None if unavailable."""
    cache = _cache_path(arxiv_id)
    if use_cache and cache.exists():
        return cache.read_text(errors="replace")
    text = _fetch_ar5iv(arxiv_id) or _fetch_pdf_text(arxiv_id)
    if text:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    return text


def _slice_section(text: str, section: str, max_chars: int) -> str:
    """Return text starting at the first heading-ish line matching ``section``."""
    low = section.lower()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s and len(s) < 120 and low in s.lower():
            return "\n".join(lines[i:])[:max_chars]
    idx = text.lower().find(low)
    if idx >= 0:
        return text[idx : idx + max_chars]
    return f"(section {section!r} not found; showing the start of the paper)\n\n" + text[:max_chars]


def read_paper_text(
    arxiv_id: str | None,
    section: str | None = None,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """Core logic behind the Adapt agent's ``read_paper`` tool (unit-testable, no @tool)."""
    if not arxiv_id:
        return (
            "No paper on file for this method (no arxiv_id in the catalog). "
            "Rely on the cloned repo + README."
        )
    text = fetch_paper_text(arxiv_id)
    if not text:
        return (
            f"Could not fetch arXiv paper {arxiv_id} (network or parse error). "
            "Rely on the cloned repo + README."
        )
    if section:
        return _slice_section(text, section, max_chars)
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + f"\n\n…[truncated at {max_chars} chars — call read_paper with a `section` "
        "like 'method', 'quantization', or 'experiments' to focus]"
    )
