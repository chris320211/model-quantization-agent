"""Research agent: per-method RAG survey + structured verdict walk.

For each catalog method, fan out a rag_search query that injects the user's model
architecture and GPU context. The LLM then walks every catalog id, emits include/reject
with a one-line reason grounded in the retrieved chunks, and picks 3-8 finalists.

Not a ReAct loop. Single `.with_structured_output(ResearchReport)` call. The LLM does
NOT pick a winner — the user picks.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from pydantic import ValidationError

from .config import load_settings
from .hardware_probe import probe_live
from .schemas import ResearchReport
from .tools import hf_model_info, rag_search
from .tools.aws_instance import UnknownInstanceType, lookup as instance_lookup
from .tools.model_alias import resolve as resolve_model
from .tools.recommender import load_catalog

log = logging.getLogger(__name__)

_INSTANCE_RE = re.compile(
    r"\b([a-z]\d+[a-z\d\-]*\.(?:\d*x?large|metal))\b", re.IGNORECASE
)

_PER_METHOD_RAG_K = 3
_RAG_FANOUT_WORKERS = 8


@dataclass
class Parsed:
    model_phrase: str
    instance_phrase: str | None


def _parse_input(text: str) -> Parsed:
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
    rows = []
    for m in load_catalog():
        rows.append(
            {
                "id": m["id"],
                "name": m["name"],
                "repo_url": (m.get("repos") or [None])[0],
                "bits": m.get("bits") or [],
                "quantizes": m.get("quantizes") or [],
                "needs_calibration": bool(m.get("needs_calibration", False)),
                "qat": bool(m.get("qat", False)),
                "inference_backends": m.get("inference_backends") or [],
                "quality_score": m.get("quality", 0),
                "speed_score": m.get("speedup", 0),
                "notes": m.get("notes", ""),
                "hyperparameters_default": m.get("hyperparameters_default"),
            }
        )
    return json.dumps(rows, indent=2)


def _build_query(
    name: str, arch: str | None, gpu: str | None, gpu_arch: str | None
) -> str:
    parts = [name]
    if arch:
        parts.append(arch)
    if gpu:
        parts.append(gpu)
    if gpu_arch:
        parts.append(gpu_arch)
    parts.append("support compatibility kernel")
    return " ".join(parts)


def _fan_out_rag(
    catalog: list[dict],
    architecture: str | None,
    gpu: str | None,
    gpu_arch: str | None,
) -> dict[str, str]:
    """Parallel rag_search, one query per catalog method. Returns {method_id: chunks_text}."""
    def _one(method: dict) -> tuple[str, str]:
        mid = method["id"]
        query = _build_query(method["name"], architecture, gpu, gpu_arch)
        try:
            chunks = rag_search.invoke(
                {"query": query, "k": _PER_METHOD_RAG_K, "method_id": mid}
            )
        except Exception as e:  # noqa: BLE001
            chunks = f"(rag_search failed: {e})"
        return mid, chunks

    with ThreadPoolExecutor(max_workers=_RAG_FANOUT_WORKERS) as ex:
        return dict(ex.map(_one, catalog))


def _format_rag_bundle(per_method: dict[str, str]) -> str:
    return "\n\n".join(
        f"=== method_id={mid} ===\n{per_method[mid]}" for mid in sorted(per_method)
    )


_PROMPT = """You are the Research agent in a quantization-porting pipeline. Your job is to
survey the user's options and produce a ResearchReport the user can choose from. You do
NOT pick a winner — the user picks.

Resolved inputs (authoritative — copy verbatim into the report):
- resolved_model_id: {model_id}
- params_b: {params_b}
- instance_type: {instance_type}
- vram_gb: {vram_gb}
- compute_capability: {compute_capability}
- gpu_arch: {gpu_arch}
- memory_bandwidth_gb_s: {memory_bandwidth_gb_s}
- peak_fp16_tflops: {peak_fp16_tflops}
- int8_tops: {int8_tops}

Hardware profile (live nvidia-smi merged with static specs; probe_ok=false means no GPU
was visible at planning time and the live fields are null — treat with appropriate caveat):
{hw_profile}

HuggingFace model info (pay attention to `architectures`):
{hf_info}

Catalog (seed/methods.yaml, authoritative — ids, repo_urls, scores MUST come from here.
The `hyperparameters_default` field, when present, lists tunable knobs and their valid
values; use it to set the candidate's `hyperparameters` to a sensible starting config):
{catalog}

Retrieved literature per method (one RAG query per catalog id, grouped by id):
{rag}

Task — do these steps in order.

1. Per-method walk. For EVERY catalog id above, emit one `considered` entry with:
   - verdict: "include" if the method plausibly supports this model's architecture AND
     runs on this GPU (compute capability / kernel availability / bit-width support) AND
     fits VRAM at a supported bit width. Otherwise "reject".
   - reason: one line. Cite the specific axis that drove the decision: architecture
     compatibility (hf_info architectures vs retrieved chunks), GPU/compute-capability
     fit (e.g. FP8 needs Hopper sm_90, Marlin kernels need Ampere sm_80+), VRAM math
     (params_b * bits / 8 * 1.4 <= vram_gb), bandwidth/TOPS bottleneck reasoning where
     relevant, or calibration/QAT fit. Ground in the retrieved chunks — do not
     hallucinate support chunks do not corroborate.
   You MUST produce exactly one `considered` entry per catalog id. No duplicates, no omissions.

2. Finalists. From the "include" verdicts, pick 3-8 and emit them as `methods`:
   - id MUST be one of the catalog ids.
   - repo_url MUST equal the catalog's repo_url for that id.
   - quality_score and speed_score MUST be copied verbatim from the catalog.
   - bits: pick one supported bit width that fits the VRAM budget.
   - est_vram_gb: params_b * bits / 8 * 1.4 (weight footprint + ~40% headroom).
   - hyperparameters: when the catalog entry has a `hyperparameters_default` block,
     emit a flat dict of {{name: default_value}} as your initial recommendation
     (e.g. {{"group_size": 128, "sym": true}}). When no defaults are listed, omit
     this field — the tune loop will infer ranges later from the method's README.
   - summary: 2-3 sentences on when this method is the right fit and what it costs.
     Cite retrieved chunks where useful.

3. Tradeoffs. One paragraph comparing the finalists across the axes that matter for
   THIS model and GPU: quality vs speed, calibration cost, bit-width options, activation
   vs weight-only, kernel maturity, and how bandwidth vs compute headroom on this GPU
   shapes which methods are likely to be Pareto-optimal. No ranking.
"""


def run(user_input: str) -> ResearchReport:
    """Resolve inputs + run a per-method RAG survey + return a ResearchReport.

    Raises:
        ValueError: if the model can't be resolved unambiguously. The orchestrator
            handles this path.
    """
    parsed = _parse_input(user_input)

    resolved = resolve_model(parsed.model_phrase or user_input)
    if resolved.model_id is None:
        hint = ", ".join(resolved.candidates) if resolved.candidates else "no candidates"
        raise ValueError(
            f"Could not resolve model from {parsed.model_phrase!r}. Candidates: {hint}"
        )
    model_id = resolved.model_id

    instance_type: str | None = None
    vram_gb: float | None = None
    compute_capability: float | None = None
    gpu: str | None = None
    gpu_arch: str | None = None
    memory_bandwidth_gb_s: float | None = None
    peak_fp16_tflops: float | None = None
    int8_tops: float | None = None
    spec = None
    if parsed.instance_phrase:
        try:
            spec = instance_lookup(parsed.instance_phrase)
            instance_type = spec.instance_type
            vram_gb = spec.vram_gb
            compute_capability = spec.compute_capability
            gpu = spec.gpu
            gpu_arch = spec.gpu_arch
            memory_bandwidth_gb_s = spec.memory_bandwidth_gb_s
            peak_fp16_tflops = spec.peak_fp16_tflops
            int8_tops = spec.int8_tops
        except UnknownInstanceType:
            instance_type = parsed.instance_phrase

    hw_profile = probe_live(spec)

    info = _hf_info(model_id)
    params_b = info.get("params_b")
    arches = info.get("architectures") or []
    arch_hint = arches[0] if arches else None

    catalog = load_catalog()
    per_method = _fan_out_rag(catalog, arch_hint, gpu, gpu_arch)

    prompt = _PROMPT.format(
        model_id=model_id,
        params_b=params_b,
        instance_type=instance_type,
        vram_gb=vram_gb,
        compute_capability=compute_capability,
        gpu_arch=gpu_arch,
        memory_bandwidth_gb_s=memory_bandwidth_gb_s,
        peak_fp16_tflops=peak_fp16_tflops,
        int8_tops=int8_tops,
        hw_profile=json.dumps(hw_profile.to_dict(), indent=2),
        hf_info=json.dumps(info, indent=2),
        catalog=_catalog_context(),
        rag=_format_rag_bundle(per_method),
    )

    s = load_settings()
    llm = ChatAnthropic(model=s.model, api_key=s.anthropic_api_key, temperature=0)
    structured = llm.with_structured_output(ResearchReport)

    try:
        report: ResearchReport = structured.invoke(prompt)
    except ValidationError as first_err:
        retry_prompt = (
            prompt
            + "\n\nYour previous response failed schema validation:\n"
            + str(first_err)
            + "\n\nRe-emit the ResearchReport. Every catalog id above must appear "
            "exactly once in `considered` with a verdict; every id in `methods` "
            "must have a matching 'include' verdict."
        )
        report = structured.invoke(retry_prompt)

    return report.model_copy(
        update={
            "resolved_model_id": model_id,
            "params_b": params_b,
            "instance_type": instance_type,
            "vram_gb": vram_gb,
            "compute_capability": compute_capability,
            "gpu_arch": gpu_arch,
            "memory_bandwidth_gb_s": memory_bandwidth_gb_s,
            "peak_fp16_tflops": peak_fp16_tflops,
            "int8_tops": int8_tops,
        }
    )
