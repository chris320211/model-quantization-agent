"""Research agent: structured verdict walk over the catalog.

Walks every catalog method, emits include/reject with a one-line reason grounded in
the structured catalog cards + the resolved model/GPU facts, and picks 3-8 finalists.

Not a ReAct loop. Single `.with_structured_output(ResearchReport)` call. The LLM does
NOT pick a winner — the user picks.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from pydantic import ValidationError

from .compatibility import (
    CompatibilityDecision,
    CompatibilityRequest,
    decisions_as_prompt_context,
    evaluate_catalog,
)
from .config import load_settings
from .llm import AgentStage, create_chat_model
from .hardware_probe import probe_live
from .schemas import ResearchReport
from .tools import hf_model_info
from .tools.aws_instance import UnknownInstanceType, lookup as instance_lookup
from .tools.model_alias import resolve as resolve_model
from .tools.recommender import load_catalog

log = logging.getLogger(__name__)

_INSTANCE_RE = re.compile(
    r"\b([a-z]\d+[a-z\d\-]*\.(?:\d*x?large|metal))\b", re.IGNORECASE
)


@dataclass
class Parsed:
    model_phrase: str
    instance_phrase: str | None
    target_bits: int | None = None
    backend: str | None = None
    priority: str | None = None
    have_calibration_data: bool | None = None
    allow_qat: bool = False
    need_kv_cache_quant: bool = False


def _parse_input(text: str) -> Parsed:
    m = _INSTANCE_RE.search(text)
    instance = m.group(1).lower() if m else None
    model_phrase = text
    if instance:
        model_phrase = _INSTANCE_RE.sub("", text).strip()
    bit_match = re.search(r"\b([1-8])\s*[- ]?bit\b", model_phrase, re.I)
    target_bits = int(bit_match.group(1)) if bit_match else None
    if bit_match:
        model_phrase = model_phrase[: bit_match.start()] + model_phrase[bit_match.end() :]
    backend_match = re.search(
        r"\b(vllm|transformers|tgi|llama[_ -]?cpp|tensorrt[_ -]?llm)\b",
        model_phrase,
        re.I,
    )
    backend = backend_match.group(1).lower().replace("-", "_").replace(" ", "_") if backend_match else None
    if backend_match:
        model_phrase = model_phrase[: backend_match.start()] + model_phrase[backend_match.end() :]
    priority_match = re.search(r"\b(quality|speed|balanced)\s*(?:first|priority)?\b", model_phrase, re.I)
    priority = priority_match.group(1).lower() if priority_match else None
    if priority_match:
        model_phrase = model_phrase[: priority_match.start()] + model_phrase[priority_match.end() :]
    no_calibration = bool(re.search(r"\b(?:no calibration|calibration[- ]free)\b", model_phrase, re.I))
    model_phrase = re.sub(r"\b(?:no calibration|calibration[- ]free)\b", " ", model_phrase, flags=re.I)
    allow_qat = bool(re.search(r"\bqat\b", model_phrase, re.I))
    need_kv = bool(re.search(r"\bkv[- ]?cache\b", model_phrase, re.I))
    model_phrase = re.sub(r"\b(?:qat|kv[- ]?cache)\b", " ", model_phrase, flags=re.I)
    model_phrase = re.sub(r"\b(port|quantize|to|on|for|using)\b", " ", model_phrase, flags=re.I)
    model_phrase = re.sub(r"\s+", " ", model_phrase).strip()
    return Parsed(
        model_phrase=model_phrase,
        instance_phrase=instance,
        target_bits=target_bits,
        backend=backend,
        priority=priority,
        have_calibration_data=False if no_calibration else None,
        allow_qat=allow_qat,
        need_kv_cache_quant=need_kv,
    )


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

User constraints (authoritative when non-null):
- target_bits: {target_bits}
- backend: {backend}
- priority: {priority}
- have_calibration_data: {have_calibration_data}
- allow_qat: {allow_qat}
- need_kv_cache_quant: {need_kv_cache_quant}

Hardware profile (live nvidia-smi merged with static specs; probe_ok=false means no GPU
was visible at planning time and the live fields are null — treat with appropriate caveat):
{hw_profile}

HuggingFace model info (pay attention to `architectures`):
{hf_info}

Catalog (packaged methods.yaml, authoritative — ids, repo_urls, scores MUST come from here.
The `hyperparameters_default` field, when present, lists tunable knobs and their valid
values; use it to set the candidate's `hyperparameters` to a sensible starting config):
{catalog}

Deterministic constraint results (authoritative for status="blocked"):
{deterministic_constraints}

The deterministic engine has already evaluated catalog facts, requested backend/bit
width, calibration/QAT policy, weight-memory estimates, documented compute capability,
GPU architecture, and model-family constraints. You MUST reject every status="blocked"
method using its reason. A status="port_required" method is NOT incompatible: it may be
a finalist, but its summary must explain that Adapt will attempt a reviewable overlay
port because upstream does not document the target family. You may investigate and
decide status="unknown" methods. An "eligible" status means only that hard constraints
passed; it is not a ranking.

Task — do these steps in order.

1. Per-method walk. For EVERY catalog id above, emit one `considered` entry with:
   - verdict: "include" if the method either supports this model natively OR its
     architecture-specific assumptions can plausibly be ported with an overlay, AND it
     runs on this GPU (compute capability / kernel availability / bit-width support) AND
     fits VRAM at a supported bit width. Otherwise "reject". Lack of documented model
     support by itself is never a rejection reason; status="port_required" is explicitly
     an include path.
   - reason: one line. Cite the specific axis that drove the decision: architecture
     compatibility (hf_info `architectures` vs the method's catalog notes),
     GPU/compute-capability fit (e.g. FP8 needs Hopper sm_90, Marlin kernels need Ampere
     sm_80+), VRAM math (params_b * bits / 8 * 1.4 <= vram_gb), bandwidth/TOPS bottleneck
     reasoning where relevant, or calibration/QAT fit. Ground in the catalog fields +
     hf_info + your own knowledge of these methods — do not assert compatibility the
     catalog and hf_info don't support.
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
   - summary: 2-3 sentences on when this method is the right fit and what it costs,
     grounded in the catalog notes + the model/GPU facts.

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
            gpu_arch = spec.gpu_arch
            memory_bandwidth_gb_s = spec.memory_bandwidth_gb_s
            peak_fp16_tflops = spec.peak_fp16_tflops
            int8_tops = spec.int8_tops
        except UnknownInstanceType:
            instance_type = parsed.instance_phrase

    hw_profile = probe_live(spec)
    if spec is None and hw_profile.probe_ok:
        vram_gb = hw_profile.vram_gb_total
        compute_capability = hw_profile.compute_capability
        gpu_arch = hw_profile.gpu_arch
        memory_bandwidth_gb_s = hw_profile.memory_bandwidth_gb_s
        peak_fp16_tflops = hw_profile.peak_fp16_tflops
        int8_tops = hw_profile.int8_tops

    info = _hf_info(model_id)
    params_b = info.get("params_b")

    compatibility_request = CompatibilityRequest(
        params_b=params_b,
        vram_gb=vram_gb,
        compute_capability=compute_capability,
        gpu_arch=gpu_arch,
        architectures=info.get("architectures") or [],
        target_bits=parsed.target_bits,
        backend=parsed.backend,
        have_calibration_data=parsed.have_calibration_data,
        allow_qat=parsed.allow_qat,
        need_kv_cache_quant=parsed.need_kv_cache_quant,
    )
    deterministic = evaluate_catalog(compatibility_request)

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
        target_bits=parsed.target_bits,
        backend=parsed.backend,
        priority=parsed.priority,
        have_calibration_data=parsed.have_calibration_data,
        allow_qat=parsed.allow_qat,
        need_kv_cache_quant=parsed.need_kv_cache_quant,
        hw_profile=json.dumps(hw_profile.to_dict(), indent=2),
        hf_info=json.dumps(info, indent=2),
        catalog=_catalog_context(),
        deterministic_constraints=decisions_as_prompt_context(deterministic),
    )

    s = load_settings()
    llm = create_chat_model(AgentStage.RESEARCH, s)
    structured = llm.with_structured_output(ResearchReport)

    try:
        report: ResearchReport = structured.invoke(prompt)
        _require_no_blocked_finalists(report, deterministic)
    except (ValidationError, ValueError) as first_err:
        retry_prompt = (
            prompt
            + "\n\nYour previous response failed schema validation:\n"
            + str(first_err)
            + "\n\nRe-emit the ResearchReport. Every catalog id above must appear "
            "exactly once in `considered` with a verdict; every id in `methods` "
            "must have a matching 'include' verdict. Deterministically blocked ids "
            "must be rejected and cannot appear in `methods`."
        )
        report = structured.invoke(retry_prompt)

    _require_no_blocked_finalists(report, deterministic)
    report = _canonicalize_blocked_verdicts(report, deterministic)
    report = _canonicalize_port_verdicts(report, deterministic)
    report = _annotate_port_candidates(report, deterministic)

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


def _require_no_blocked_finalists(
    report: ResearchReport, decisions: list[CompatibilityDecision]
) -> None:
    blocked = {d.method_id for d in decisions if d.hard_blocked}
    invalid = [method.id for method in report.methods if method.id in blocked]
    if invalid:
        raise ValueError(f"deterministically blocked finalists: {invalid}")


def _canonicalize_blocked_verdicts(
    report: ResearchReport, decisions: list[CompatibilityDecision]
) -> ResearchReport:
    """Make hard rejections non-overridable while retaining the full catalog walk."""
    blocked = {d.method_id: d for d in decisions if d.hard_blocked}
    considered = []
    for row in report.considered:
        decision = blocked.get(row.id)
        if decision is None:
            considered.append(row)
            continue
        reason = "; ".join(r.message for r in decision.reasons)
        considered.append(row.model_copy(update={
            "verdict": "reject",
            "reason": f"deterministic constraint: {reason}",
        }))
    return report.model_copy(update={"considered": considered})


def _annotate_port_candidates(
    report: ResearchReport, decisions: list[CompatibilityDecision]
) -> ResearchReport:
    """Attach deterministic port requirements to finalists without trusting LLM output."""
    porting = {d.method_id: d for d in decisions if d.requires_port}
    methods = []
    for method in report.methods:
        decision = porting.get(method.id)
        if decision is None:
            methods.append(method.model_copy(update={"requires_port": False, "port_reason": None}))
            continue
        reason = "; ".join(r.message for r in decision.reasons)
        methods.append(method.model_copy(update={
            "requires_port": True,
            "port_reason": reason,
        }))
    return report.model_copy(update={"methods": methods})


def _canonicalize_port_verdicts(
    report: ResearchReport, decisions: list[CompatibilityDecision]
) -> ResearchReport:
    """Do not let missing upstream model support become a rejection by itself."""
    porting = {d.method_id: d for d in decisions if d.requires_port}
    considered = []
    for row in report.considered:
        decision = porting.get(row.id)
        if decision is None:
            considered.append(row)
            continue
        reason = "; ".join(r.message for r in decision.reasons)
        considered.append(row.model_copy(update={
            "verdict": "include",
            "reason": f"overlay port path: {reason}",
        }))
    return report.model_copy(update={"considered": considered})
