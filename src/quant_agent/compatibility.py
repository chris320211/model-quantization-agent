"""Deterministic method compatibility evaluation.

The LLM-facing Research agent is useful for interpreting incomplete evidence, but
hard constraints should not depend on a model remembering kernel or catalog facts.
This module evaluates the facts that can be decided mechanically and emits an
auditable decision for every catalog method:

``blocked``
    At least one hard constraint is violated. Research may explain the result but
    cannot promote the method to a finalist.
``eligible``
    All requested hard constraints pass and the capability dataset contains model
    family evidence (or the request does not expose a family).
``unknown``
    No hard constraint failed, but capability evidence is incomplete. Research may
    investigate the paper/repository and decide whether to include the method.

The packaged ``method_capabilities.yaml`` is deliberately separate from
``methods.yaml``: the latter is the product catalog, while the former records
hardware/model compatibility facts and their provenance.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from .config import DATA_ROOT
from .tools.recommender import load_catalog

_VRAM_OVERHEAD_FACTOR = 1.4


class CompatibilityRequest(BaseModel):
    """Normalized hard constraints used for the complete catalog walk."""

    params_b: float | None = Field(None, ge=0)
    vram_gb: float | None = Field(None, ge=0)
    compute_capability: float | None = Field(None, ge=0)
    gpu_arch: str | None = None
    architectures: list[str] = Field(default_factory=list)
    target_bits: int | None = Field(None, gt=0)
    backend: str | None = None
    have_calibration_data: bool | None = None
    allow_qat: bool = False
    need_activation_quant: bool = False
    need_kv_cache_quant: bool = False


class ConstraintReason(BaseModel):
    code: str
    message: str
    source: Literal["catalog", "capability", "request", "estimate"]


class CompatibilityDecision(BaseModel):
    method_id: str
    status: Literal["eligible", "blocked", "unknown"]
    chosen_bits: int | None = None
    model_family: str | None = None
    reasons: list[ConstraintReason] = Field(default_factory=list)

    @property
    def hard_blocked(self) -> bool:
        return self.status == "blocked"


_FAMILY_ALIASES: tuple[tuple[str, str], ...] = (
    ("deepseek", "deepseek"),
    ("mixtral", "mixtral"),
    ("mistral", "mistral"),
    ("qwen2", "qwen2"),
    ("qwen", "qwen"),
    ("gemma", "gemma"),
    ("llama", "llama"),
    ("baichuan", "baichuan"),
    ("gptneox", "gpt_neox"),
    ("gpt_neox", "gpt_neox"),
    ("gptj", "gptj"),
    ("gpt_j", "gptj"),
    ("falcon", "falcon"),
    ("bloom", "bloom"),
    ("phi", "phi"),
    ("opt", "opt"),
    ("mpt", "mpt"),
    ("t5", "t5"),
)


def infer_model_family(architectures: list[str] | None) -> str | None:
    joined = " ".join(architectures or []).lower().replace("-", "_")
    for needle, family in _FAMILY_ALIASES:
        if needle in joined:
            return family
    return None


@lru_cache(maxsize=1)
def load_capabilities() -> dict[str, dict[str, Any]]:
    path = DATA_ROOT / "method_capabilities.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        payload = yaml.safe_load(f) or {}
    methods = payload.get("methods", payload)
    if not isinstance(methods, dict):
        raise ValueError("method_capabilities.yaml must contain a 'methods' mapping")
    _validate_capabilities(methods)
    return methods


def _validate_capabilities(methods: dict[str, dict[str, Any]]) -> None:
    catalog = {row["id"]: row for row in load_catalog()}
    missing = set(catalog) - set(methods)
    extra = set(methods) - set(catalog)
    if missing or extra:
        raise ValueError(
            f"capability/catalog id mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for method_id, capability in methods.items():
        if capability.get("family_policy") not in {"evidence", "allowlist", "unrestricted"}:
            raise ValueError(f"invalid family_policy for {method_id}")
        if capability.get("gpu_arch_policy") not in {"evidence", "allowlist"}:
            raise ValueError(f"invalid gpu_arch_policy for {method_id}")
        repo_url = capability.get("repository_url")
        if repo_url not in (catalog[method_id].get("repos") or []):
            raise ValueError(f"capability repository_url is not catalog-pinned for {method_id}")
        arxiv_id = catalog[method_id].get("arxiv_id")
        paper_url = capability.get("paper_url")
        if arxiv_id and paper_url != f"https://arxiv.org/abs/{arxiv_id}":
            raise ValueError(f"capability paper_url does not match catalog for {method_id}")
        for fact in capability.get("facts") or []:
            if fact.get("confidence") not in {"documented", "inferred"}:
                raise ValueError(f"invalid fact confidence for {method_id}")
            if not fact.get("field") or not fact.get("source"):
                raise ValueError(f"capability fact lacks field/source for {method_id}")


def _reason(code: str, message: str, source: str) -> ConstraintReason:
    return ConstraintReason(code=code, message=message, source=source)


def _candidate_bits(method: dict, request: CompatibilityRequest) -> list[int]:
    supported = [int(b) for b in (method.get("bits") or [])]
    if request.target_bits is not None:
        return [request.target_bits] if request.target_bits in supported else []
    preference = (4, 8, 6, 5, 3, 2, 1)
    return [b for b in preference if b in supported]


def _fits(params_b: float, vram_gb: float, bits: int) -> bool:
    return params_b * bits / 8 * _VRAM_OVERHEAD_FACTOR <= vram_gb


def evaluate_method(
    method: dict,
    request: CompatibilityRequest,
    capability: dict[str, Any] | None = None,
) -> CompatibilityDecision:
    capability = capability or {}
    method_id = str(method["id"])
    reasons: list[ConstraintReason] = []
    blocked = False

    backends = method.get("inference_backends") or []
    if request.backend and request.backend != "any" and request.backend not in backends:
        blocked = True
        reasons.append(_reason(
            "backend_mismatch",
            f"requested backend {request.backend!r} is not in {backends}",
            "catalog",
        ))

    if method.get("qat") and not request.allow_qat:
        blocked = True
        reasons.append(_reason("qat_not_allowed", "method requires QAT", "catalog"))

    if request.have_calibration_data is False and method.get("needs_calibration"):
        blocked = True
        reasons.append(_reason(
            "calibration_unavailable", "method requires calibration data", "catalog"
        ))

    quantizes = set(method.get("quantizes") or [])
    if request.need_activation_quant and "activations" not in quantizes:
        blocked = True
        reasons.append(_reason(
            "activation_quant_required", "method does not quantize activations", "catalog"
        ))
    if request.need_kv_cache_quant and "kv_cache" not in quantizes:
        blocked = True
        reasons.append(_reason(
            "kv_cache_quant_required", "method does not quantize the KV cache", "catalog"
        ))
    if not request.need_kv_cache_quant and quantizes == {"kv_cache"}:
        blocked = True
        reasons.append(_reason(
            "kv_cache_only",
            "KV-cache-only method does not reduce model weight memory",
            "catalog",
        ))

    bits_candidates = _candidate_bits(method, request)
    if request.target_bits is not None and not bits_candidates:
        blocked = True
        reasons.append(_reason(
            "bit_width_mismatch",
            f"requested {request.target_bits}-bit is not supported; catalog has {method.get('bits') or []}",
            "catalog",
        ))

    min_cc = capability.get("min_compute_capability")
    if min_cc is not None and request.compute_capability is not None:
        if float(request.compute_capability) < float(min_cc):
            blocked = True
            reasons.append(_reason(
                "compute_capability_too_low",
                f"requires compute capability >= {float(min_cc):g}; target is {request.compute_capability:g}",
                "capability",
            ))

    supported_gpu_arches = {
        str(v).strip().lower() for v in (capability.get("supported_gpu_arches") or [])
    }
    if supported_gpu_arches and request.gpu_arch:
        target_arch = request.gpu_arch.strip().lower()
        if (
            target_arch not in supported_gpu_arches
            and capability.get("gpu_arch_policy") == "allowlist"
        ):
            blocked = True
            reasons.append(_reason(
                "gpu_arch_mismatch",
                f"documented GPU architectures are {sorted(supported_gpu_arches)}; target is {target_arch}",
                "capability",
            ))

    family = infer_model_family(request.architectures)
    supported_families = {
        str(v).strip().lower() for v in (capability.get("supported_families") or [])
    }
    unsupported_families = {
        str(v).strip().lower() for v in (capability.get("unsupported_families") or [])
    }
    if family and family in unsupported_families:
        blocked = True
        reasons.append(_reason(
            "model_family_denied", f"{family} is explicitly unsupported", "capability"
        ))
    elif (
        family
        and supported_families
        and family not in supported_families
        and capability.get("family_policy") == "allowlist"
    ):
        blocked = True
        reasons.append(_reason(
            "model_family_not_supported",
            f"documented families are {sorted(supported_families)}; target is {family}",
            "capability",
        ))

    chosen_bits: int | None = None
    if bits_candidates:
        if request.params_b is not None and request.vram_gb is not None:
            fitting = [
                b for b in bits_candidates if _fits(request.params_b, request.vram_gb, b)
            ]
            if fitting:
                chosen_bits = fitting[0]
            else:
                blocked = True
                estimates = [
                    round(request.params_b * b / 8 * _VRAM_OVERHEAD_FACTOR, 2)
                    for b in bits_candidates
                ]
                reasons.append(_reason(
                    "estimated_vram_exceeded",
                    f"estimated footprints {estimates} GB exceed {request.vram_gb:g} GB",
                    "estimate",
                ))
        else:
            chosen_bits = bits_candidates[0]

    family_evidence_known = bool(
        family is None
        or family in supported_families
        or capability.get("family_policy") == "unrestricted"
    )
    if blocked:
        status: Literal["eligible", "blocked", "unknown"] = "blocked"
    elif family_evidence_known:
        status = "eligible"
        reasons.append(_reason(
            "hard_constraints_pass", "all known deterministic constraints pass", "request"
        ))
    else:
        status = "unknown"
        reasons.append(_reason(
            "missing_family_evidence",
            f"no documented model-family evidence for {family}",
            "capability",
        ))

    return CompatibilityDecision(
        method_id=method_id,
        status=status,
        chosen_bits=chosen_bits,
        model_family=family,
        reasons=reasons,
    )


def evaluate_catalog(request: CompatibilityRequest) -> list[CompatibilityDecision]:
    capabilities = load_capabilities()
    return [
        evaluate_method(method, request, capabilities.get(method["id"]))
        for method in load_catalog()
    ]


def decisions_as_prompt_context(decisions: list[CompatibilityDecision]) -> str:
    rows = []
    for decision in decisions:
        rows.append({
            "id": decision.method_id,
            "status": decision.status,
            "chosen_bits": decision.chosen_bits,
            "model_family": decision.model_family,
            "reasons": [reason.model_dump() for reason in decision.reasons],
        })
    return yaml.safe_dump(rows, sort_keys=False)
