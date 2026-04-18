"""Research agent: resolve inputs + produce a ResearchReport via structured output.

Not a ReAct loop. The orchestrator pre-loads the deterministic context (model alias,
AWS instance spec, HF model info, catalog, RAG survey) and hands the LLM a single
`.with_structured_output(ResearchReport)` call. The LLM's job is narrow: pick 3-8
catalog-backed methods and write a tradeoffs paragraph. It does NOT recommend a winner.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic

from .config import load_settings
from .schemas import ResearchReport
from .tools import hf_model_info
from .tools.aws_instance import UnknownInstanceType, lookup as instance_lookup
from .tools.model_alias import resolve as resolve_model
from .tools.rag import rag_survey
from .tools.recommender import load_catalog

log = logging.getLogger(__name__)

_INSTANCE_RE = re.compile(
    r"\b([a-z]\d+[a-z]*(?:n|e|de)?\.\d*x?large)\b", re.IGNORECASE
)


@dataclass
class Parsed:
    model_phrase: str
    instance_phrase: str | None


def _parse_input(text: str) -> Parsed:
    """Best-effort split: pull out the instance type if it's recognizable, treat the rest
    as the model phrase. Deliberately simple — ambiguity surfaces downstream."""
    m = _INSTANCE_RE.search(text)
    instance = m.group(1).lower() if m else None
    model_phrase = text
    if instance:
        model_phrase = _INSTANCE_RE.sub("", text).strip()
    model_phrase = re.sub(r"\b(port|quantize|to|on|for|using)\b", " ", model_phrase, flags=re.I)
    model_phrase = re.sub(r"\s+", " ", model_phrase).strip()
    return Parsed(model_phrase=model_phrase, instance_phrase=instance)


def _hf_info(model_id: str) -> dict:
    try:
        raw = hf_model_info.invoke({"model_id": model_id})
        return json.loads(raw)
    except Exception as e:  # noqa: BLE001
        log.warning("hf_model_info failed for %s: %s", model_id, e)
        return {"model_id": model_id, "params_b": None, "architectures": None}


def _catalog_context() -> str:
    """JSON-serialise the catalog fields the LLM needs to copy verbatim."""
    rows = []
    for m in load_catalog():
        rows.append(
            {
                "id": m["id"],
                "name": m["name"],
                "repo_url": (m.get("repos") or [None])[0],
                "bits": m.get("bits") or [],
                "quality_score": m.get("quality", 0),
                "speed_score": m.get("speedup", 0),
                "needs_calibration": bool(m.get("needs_calibration", False)),
                "notes": m.get("notes", ""),
            }
        )
    return json.dumps(rows, indent=2)


_PROMPT = """You are the Research agent in a quantization-porting pipeline. Your ONLY job is
to produce a ResearchReport that surfaces 3-8 candidate quantization methods for the user's
model and hardware. You do NOT pick a winner — the user picks.

Resolved inputs (authoritative — copy verbatim into the report):
- resolved_model_id: {model_id}
- params_b: {params_b}
- instance_type: {instance_type}
- vram_gb: {vram_gb}

HuggingFace model info:
{hf_info}

Catalog (seed/methods.yaml, authoritative — you MUST pick ids, repo_urls, and scores from here):
{catalog}

Retrieved literature (broad survey across the catalog):
{rag}

Rules:
- Each method.id MUST be one of the catalog ids above.
- repo_url MUST equal the catalog's repos[0] for that id.
- quality_score and speed_score MUST be copied from the catalog (quality / speedup).
- Pick between 3 and 8 methods; prefer those that plausibly fit {vram_gb} GB VRAM for this model.
- For each method.summary: 2-3 sentences on when this method is the right fit, why, and what
  it costs (calibration, quality, supported bits). Cite the RAG context when useful.
- For bits: pick one supported bit width that best fits the user's VRAM budget.
- est_vram_gb: estimate weight footprint (params_b * bits / 8) plus ~40% headroom.
- Write a tradeoffs paragraph comparing the surfaced options (no ranking — just the axes).
"""


def run(user_input: str) -> ResearchReport:
    """Resolve inputs from a free-form user request and return a ResearchReport.

    Raises:
        ValueError: if the model can't be resolved unambiguously and the orchestrator
            hasn't supplied a disambiguation yet. The orchestrator handles this path.
    """
    parsed = _parse_input(user_input)

    # Model resolution
    resolved = resolve_model(parsed.model_phrase or user_input)
    if resolved.model_id is None:
        # Ambiguous or unknown — orchestrator will prompt the user.
        hint = ", ".join(resolved.candidates) if resolved.candidates else "no candidates"
        raise ValueError(
            f"Could not resolve model from {parsed.model_phrase!r}. Candidates: {hint}"
        )
    model_id = resolved.model_id

    # Instance / VRAM resolution
    instance_type: str | None = None
    vram_gb: float | None = None
    if parsed.instance_phrase:
        try:
            spec = instance_lookup(parsed.instance_phrase)
            instance_type = spec.instance_type
            vram_gb = spec.vram_gb
        except UnknownInstanceType:
            instance_type = parsed.instance_phrase

    # HF info
    info = _hf_info(model_id)
    params_b = info.get("params_b")

    prompt = _PROMPT.format(
        model_id=model_id,
        params_b=params_b,
        instance_type=instance_type,
        vram_gb=vram_gb,
        hf_info=json.dumps(info, indent=2),
        catalog=_catalog_context(),
        rag=rag_survey("quantization methods overview tradeoffs", k=20),
    )

    s = load_settings()
    llm = ChatAnthropic(model=s.model, api_key=s.anthropic_api_key, temperature=0)
    report: ResearchReport = llm.with_structured_output(ResearchReport).invoke(prompt)

    # Defend against the LLM drifting from the resolved facts.
    return report.model_copy(
        update={
            "resolved_model_id": model_id,
            "params_b": params_b,
            "instance_type": instance_type,
            "vram_gb": vram_gb,
        }
    )
