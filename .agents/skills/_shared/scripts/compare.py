#!/usr/bin/env python3
"""Filterable catalog of quantized methods (model × method × GPU instance).

Library users filter `compare/catalog.json`, compare WikiText-2 metrics, then
fetch weights themselves from Hugging Face (`hub_repo_id`). This file is the
table. Hub is only the weight store.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jobs as jobs_mod
import paths
from io_utils import atomic_write_text
from request import load_request
from verify import resolve_workspace_path

SCHEMA_VERSION = 1
QUALITY_PPL_RATIO_MAX = 1.5
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_HUB_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")
_REQUIRED_NUMS = (
    "ppl",
    "ppl_ratio",
    "tokens_per_s",
    "peak_vram_gb",
    "fp16_ppl",
    "fp16_tokens_per_s",
    "fp16_peak_vram_gb",
)


def group_key(model_id: str, gpu_instance: str) -> str:
    return f"{model_id.strip()}|{gpu_instance.strip()}"


def group_filename(model_id: str, gpu_instance: str) -> str:
    model = _SAFE_RE.sub("_", model_id.strip()).strip("._-") or "model"
    gpu = _SAFE_RE.sub("_", gpu_instance.strip()).strip("._-") or "gpu"
    return f"{model}__{gpu}.json"


def contribution_filename(model_id: str, gpu_instance: str, method_name: str) -> str:
    model = _SAFE_RE.sub("_", model_id.strip()).strip("._-") or "model"
    gpu = _SAFE_RE.sub("_", gpu_instance.strip()).strip("._-") or "gpu"
    method = _SAFE_RE.sub("_", method_name.strip()).strip("._-") or "method"
    return f"{model}__{gpu}__{method}.json"


def contributions_dir() -> Path:
    path = paths.COMPARE_ROOT / "contributions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected object JSON: {path}")
    return data


def _packed_artifact(output_dir: Path) -> bool:
    cfg = output_dir / "quantization_config.json"
    if cfg.is_file():
        try:
            payload = json.loads(cfg.read_text())
            fmt = str(payload.get("format") or "").lower()
            if "int4" in fmt or "packed" in fmt:
                return True
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return any(
        path.is_file() and ("packed" in path.name.lower() or "int4" in path.name.lower())
        for path in output_dir.glob("*")
    )


def rank_tuple(row: dict) -> tuple:
    """Higher is better. quality_ok first, then tok/s, then lower VRAM, packed, lower PPL ratio."""
    quality = 1 if row.get("quality_ok") is True else 0
    packed = 1 if row.get("packed") else 0
    tps = row.get("tokens_per_s")
    tps_v = float(tps) if isinstance(tps, (int, float)) else -1.0
    vram = row.get("peak_vram_gb")
    vram_v = -float(vram) if isinstance(vram, (int, float)) else -1e9
    ratio = row.get("ppl_ratio")
    ratio_v = -float(ratio) if isinstance(ratio, (int, float)) else -1e9
    return (quality, tps_v, vram_v, packed, ratio_v)


def rank_methods(methods: list[dict]) -> list[dict]:
    ordered = sorted(methods, key=rank_tuple, reverse=True)
    ranked: list[dict] = []
    for index, row in enumerate(ordered, start=1):
        item = dict(row)
        item["rank"] = index
        ranked.append(item)
    return ranked


def pick_best(methods: list[dict]) -> dict | None:
    usable = [row for row in methods if row.get("quality_ok") is True]
    pool = usable or []
    if not pool:
        return None
    return max(pool, key=rank_tuple)


def empty_group(model_id: str, gpu_instance: str, gpu_name: str | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "gpu_instance": gpu_instance,
        "gpu_name": gpu_name,
        "fp16_baseline": None,
        "methods": [],
        "best": None,
    }


def load_group(model_id: str, gpu_instance: str) -> dict:
    path = paths.COMPARE_ROOT / "groups" / group_filename(model_id, gpu_instance)
    if not path.is_file():
        return empty_group(model_id, gpu_instance)
    data = _read_json(path)
    data.setdefault("methods", [])
    return data


def _write_group(group: dict) -> Path:
    paths.COMPARE_ROOT.mkdir(parents=True, exist_ok=True)
    group_dir = paths.COMPARE_ROOT / "groups"
    group_dir.mkdir(parents=True, exist_ok=True)
    methods = rank_methods(list(group.get("methods") or []))
    group["methods"] = methods
    best = pick_best(methods)
    if best is None:
        group["best"] = None
    else:
        group["best"] = {
            "method_name": best.get("method_name"),
            "slug": best.get("slug"),
            "job_id": best.get("job_id"),
            "rank": best.get("rank"),
            "why": (
                "quality_ok, then higher WikiText-2 tok/s, then lower peak VRAM, "
                "prefer packed, then lower ppl_ratio"
            ),
            "artifact_dir": best.get("artifact_dir"),
            "hub_url": best.get("hub_url"),
        }
    dest = group_dir / group_filename(str(group["model_id"]), str(group["gpu_instance"]))
    atomic_write_text(dest, json.dumps(group, indent=2) + "\n", mode=0o644)
    rebuild_derived()
    return dest


def _index_path() -> Path:
    return paths.COMPARE_ROOT / "index.json"


def _write_index() -> Path:
    entries: list[dict] = []
    for group in load_boards():
        methods = rank_methods(list(group.get("methods") or []))
        best = pick_best(methods)
        best = best if isinstance(best, dict) else {}
        entries.append(
            {
                "model_id": group.get("model_id"),
                "gpu_instance": group.get("gpu_instance"),
                "gpu_name": group.get("gpu_name"),
                "n_methods": len(methods),
                "best_method": best.get("method_name"),
            }
        )
    index = {"schema_version": SCHEMA_VERSION, "groups": entries}
    dest = _index_path()
    atomic_write_text(dest, json.dumps(index, indent=2) + "\n", mode=0o644)
    return dest


def _hub_repo_id(hub_url: str | None) -> str | None:
    if not hub_url:
        return None
    text = str(hub_url).strip().rstrip("/")
    marker = "huggingface.co/"
    if marker in text:
        text = text.split(marker, 1)[1]
    text = text.removeprefix("models/")
    if text.count("/") != 1:
        return None
    return text


def _eq(value, want: str | None) -> bool:
    if want is None or want == "":
        return True
    return str(value or "").strip().lower() == want.strip().lower()


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_contribution(data: dict, *, require_hub: bool = True) -> dict:
    if not isinstance(data, dict):
        raise ValueError("contribution must be a JSON object")
    for key in ("model_id", "method_name", "gpu_instance"):
        if not str(data.get(key) or "").strip():
            raise ValueError(f"contribution missing {key}")
    if data.get("quality_ok") is not True:
        raise ValueError("only quality_ok runs can be contributed")
    for key in _REQUIRED_NUMS:
        if not _finite_number(data.get(key)):
            raise ValueError(f"contribution missing numeric {key}")
    ratio = float(data["ppl_ratio"])
    expected = float(data["ppl"]) / float(data["fp16_ppl"])
    if expected <= 0 or abs(ratio - expected) / expected > 0.02:
        raise ValueError("ppl_ratio must match ppl / fp16_ppl")
    if ratio > QUALITY_PPL_RATIO_MAX:
        raise ValueError(f"ppl_ratio must be <= {QUALITY_PPL_RATIO_MAX}")
    dataset = str(data.get("dataset") or "wikitext-2-raw-v1")
    if "wikitext-2" not in dataset.lower():
        raise ValueError("contributions must use WikiText-2")
    if data.get("max_seq_len") not in (None, 2048):
        raise ValueError("contributions must use 2048-token windows")
    repo = str(data.get("hub_repo_id") or "").strip() or _hub_repo_id(
        data.get("hub_url") if isinstance(data.get("hub_url"), str) else None
    )
    if require_hub:
        if not repo or not _HUB_REPO_RE.fullmatch(repo):
            raise ValueError("contribution needs public hub_repo_id org/name so others can fetch")
        data = dict(data)
        data["hub_repo_id"] = repo
        data["hub_url"] = data.get("hub_url") or f"https://huggingface.co/{repo}"
    return data


def _method_from_contribution(data: dict) -> dict:
    hub_url = data.get("hub_url")
    repo = data.get("hub_repo_id") or _hub_repo_id(hub_url if isinstance(hub_url, str) else None)
    if repo and not hub_url:
        hub_url = f"https://huggingface.co/{repo}"
    return {
        "method_name": data.get("method_name"),
        "slug": data.get("slug"),
        "job_id": data.get("job_id"),
        "quality_ok": data.get("quality_ok") is True,
        "ppl": data.get("ppl"),
        "ppl_ratio": data.get("ppl_ratio"),
        "tokens_per_s": data.get("tokens_per_s"),
        "peak_vram_gb": data.get("peak_vram_gb"),
        "improved_throughput": data.get("improved_throughput"),
        "improved_vram": data.get("improved_vram"),
        "packed": bool(data.get("packed")),
        "artifact_dir": data.get("artifact_dir"),
        "hub_url": hub_url,
        "verification_status": data.get("verification_status"),
    }


def _apply_contribution(boards: dict[str, dict], data: dict) -> None:
    model_id = str(data["model_id"])
    gpu_instance = str(data["gpu_instance"])
    key = group_key(model_id, gpu_instance)
    group = boards.get(key) or empty_group(model_id, gpu_instance, data.get("gpu_name"))
    if data.get("gpu_name"):
        group["gpu_name"] = data.get("gpu_name")
    if not group.get("fp16_baseline"):
        group["fp16_baseline"] = {
            "perplexity": data.get("fp16_ppl"),
            "tokens_per_s": data.get("fp16_tokens_per_s"),
            "peak_vram_gb": data.get("fp16_peak_vram_gb"),
        }
    method_row = _method_from_contribution(data)
    by_method = {str(old.get("method_name")): dict(old) for old in group.get("methods") or []}
    name = str(method_row.get("method_name"))
    prev = by_method.get(name, {})
    merged = {**prev, **{k: v for k, v in method_row.items() if v is not None}}
    if not merged.get("hub_url"):
        merged["hub_url"] = prev.get("hub_url")
    by_method[name] = merged
    group["methods"] = list(by_method.values())
    boards[key] = group


def load_boards() -> list[dict]:
    boards: dict[str, dict] = {}
    groups_dir = paths.COMPARE_ROOT / "groups"
    if groups_dir.is_dir():
        for path in sorted(groups_dir.glob("*.json")):
            group = _read_json(path)
            key = group_key(str(group.get("model_id") or ""), str(group.get("gpu_instance") or ""))
            if key == "|":
                continue
            boards[key] = group
    contrib_dir = paths.COMPARE_ROOT / "contributions"
    if contrib_dir.is_dir():
        for path in sorted(contrib_dir.glob("*.json")):
            data = validate_contribution(_read_json(path), require_hub=True)
            _apply_contribution(boards, data)
    return list(boards.values())


def catalog_rows() -> list[dict]:
    rows: list[dict] = []
    for group in load_boards():
        methods = rank_methods(list(group.get("methods") or []))
        best = pick_best(methods)
        best_name = best.get("method_name") if best else None
        fp16 = group.get("fp16_baseline") if isinstance(group.get("fp16_baseline"), dict) else {}
        for method in methods:
            hub_url = method.get("hub_url")
            repo = _hub_repo_id(hub_url if isinstance(hub_url, str) else None)
            rows.append(
                {
                    "model_id": group.get("model_id"),
                    "method_name": method.get("method_name"),
                    "gpu_instance": group.get("gpu_instance"),
                    "gpu_name": group.get("gpu_name"),
                    "rank": method.get("rank"),
                    "is_best": (
                        method.get("method_name") == best_name
                        and method.get("quality_ok") is True
                    ),
                    "quality_ok": method.get("quality_ok"),
                    "ppl": method.get("ppl"),
                    "ppl_ratio": method.get("ppl_ratio"),
                    "tokens_per_s": method.get("tokens_per_s"),
                    "peak_vram_gb": method.get("peak_vram_gb"),
                    "fp16_ppl": fp16.get("perplexity"),
                    "fp16_tokens_per_s": fp16.get("tokens_per_s"),
                    "fp16_peak_vram_gb": fp16.get("peak_vram_gb"),
                    "packed": method.get("packed"),
                    "hub_url": hub_url,
                    "hub_repo_id": repo,
                    "artifact_dir": method.get("artifact_dir"),
                    "job_id": method.get("job_id"),
                    "slug": method.get("slug"),
                }
            )
    return rows


def query(
    *,
    model_id: str | None = None,
    method_name: str | None = None,
    gpu_instance: str | None = None,
    best_only: bool = False,
) -> list[dict]:
    rows = [
        row
        for row in catalog_rows()
        if _eq(row.get("model_id"), model_id)
        and _eq(row.get("method_name"), method_name)
        and _eq(row.get("gpu_instance"), gpu_instance)
    ]
    if best_only:
        rows = [row for row in rows if row.get("is_best")]
    return rows


def fetch_howto(row: dict) -> dict:
    repo = row.get("hub_repo_id") or _hub_repo_id(
        row.get("hub_url") if isinstance(row.get("hub_url"), str) else None
    )
    if repo:
        local = repo.replace("/", "__")
        return {
            "status": "ok",
            "hub_repo_id": repo,
            "command": f"huggingface-cli download {repo} --local-dir ./{local}",
            "python": f'from huggingface_hub import snapshot_download\nsnapshot_download("{repo}")',
        }
    return {
        "status": "local_only",
        "artifact_dir": row.get("artifact_dir"),
        "message": (
            "No Hub upload for this row yet. Copy artifact_dir, or run "
            "quant-publish --upload, then fetch with huggingface-cli."
        ),
    }


def rebuild_derived() -> dict:
    index = _write_index()
    catalog = _write_catalog()
    readme = _write_markdown()
    return {"index": str(index), "catalog": str(catalog), "readme": str(readme)}


def _write_catalog() -> Path:
    dest = paths.COMPARE_ROOT / "catalog.json"
    payload = {"schema_version": SCHEMA_VERSION, "rows": catalog_rows()}
    atomic_write_text(dest, json.dumps(payload, indent=2) + "\n", mode=0o644)
    return dest


def _write_markdown() -> Path:
    lines = [
        "# Quantized method comparison",
        "",
        "Filter `catalog.json` by `model_id`, `method_name`, `gpu_instance`.",
        "Compare WikiText-2 metrics on the matching rows. Fetch weights yourself",
        "with `huggingface-cli download <hub_repo_id>` (Hub stores files; this table",
        "is the library index).",
        "",
        "```bash",
        "PY=$(command -v python || command -v python3)",
        'S="$PY .agents/skills/_shared/scripts"',
        "$S/compare.py --model-id <org/model> --gpu-instance <instance>",
        "$S/compare.py --model-id <org/model> --method <name> --gpu-instance <instance> --fetch",
        "```",
        "",
        "Rank: `quality_ok`, then tokens/s, then lower VRAM, packed/realquant, then lower PPL ratio.",
        "",
        "## Contribute a run",
        "",
        "Do not edit `catalog.json`. Upload public weights to Hugging Face, then add one",
        "JSON file under `contributions/` and open a PR. Schema: `contributions/README.md`.",
        "",
    ]
    for group in load_boards():
        methods = rank_methods(list(group.get("methods") or []))
        best = pick_best(methods) or {}
        lines.append(f"## {group.get('model_id')} on {group.get('gpu_instance')}")
        if group.get("gpu_name"):
            lines.append(f"GPU: {group['gpu_name']}")
        lines.append("")
        lines.append(
            "| Rank | Method | PPL ratio | tok/s | VRAM GB | packed | quality_ok | pull |"
        )
        lines.append("| ---: | --- | ---: | ---: | ---: | --- | --- | --- |")
        for row in methods:
            pull = row.get("hub_url") or row.get("artifact_dir") or ""
            lines.append(
                "| {rank} | {method} | {ratio} | {tps} | {vram} | {packed} | {qok} | {pull} |".format(
                    rank=row.get("rank"),
                    method=row.get("method_name"),
                    ratio=_num(row.get("ppl_ratio"), 4),
                    tps=_num(row.get("tokens_per_s"), 1),
                    vram=_num(row.get("peak_vram_gb"), 3),
                    packed="yes" if row.get("packed") else "no",
                    qok="yes" if row.get("quality_ok") else "no",
                    pull=pull,
                )
            )
        if best.get("method_name"):
            lines.append("")
            lines.append(f"**Best:** {best['method_name']} (`{best.get('slug') or best.get('method_name')}`).")
        lines.append("")
    dest = paths.COMPARE_ROOT / "README.md"
    atomic_write_text(dest, "\n".join(lines) + "\n", mode=0o644)
    return dest


def _num(value, digits: int) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value):.{digits}f}"


def _public_artifact_dir(output_dir: Path) -> str:
    resolved = output_dir.expanduser().resolve()
    try:
        return str(resolved.relative_to(paths.REPO_ROOT))
    except ValueError:
        return str(resolved)


def record_job(
    *,
    job_id: str,
    request: dict,
    hub_url: str | None = None,
) -> dict:
    meta = jobs_mod.refresh_status(job_id)
    job_directory = jobs_mod.job_dir(job_id)
    benchmark_path = job_directory / "benchmark.json"
    if not benchmark_path.is_file():
        raise RuntimeError("compare record requires jobs/<id>/benchmark.json")
    benchmark = _read_json(benchmark_path)
    comparison = benchmark.get("comparison") if isinstance(benchmark.get("comparison"), dict) else {}
    quantized = benchmark.get("quantized") if isinstance(benchmark.get("quantized"), dict) else {}
    baseline = benchmark.get("fp16_baseline") if isinstance(benchmark.get("fp16_baseline"), dict) else {}
    output_dir = resolve_workspace_path(meta.output_dir)
    model_id = str(request.get("model_id") or "")
    gpu_instance = str(request.get("gpu_instance") or "")
    if not model_id or not gpu_instance:
        raise ValueError("request must include model_id and gpu_instance")
    group = load_group(model_id, gpu_instance)
    group["gpu_name"] = request.get("gpu_name") or request.get("gpu") or group.get("gpu_name")
    if baseline:
        group["fp16_baseline"] = {
            "perplexity": baseline.get("perplexity"),
            "tokens_per_s": baseline.get("tokens_per_s"),
            "peak_vram_gb": baseline.get("peak_vram_gb"),
        }
    method_name = str(request.get("method_name") or "unknown")
    slug = str(request.get("slug") or "")
    row = {
        "method_name": method_name,
        "slug": slug,
        "job_id": job_id,
        "quality_ok": comparison.get("quality_ok") is True,
        "ppl": quantized.get("perplexity"),
        "ppl_ratio": comparison.get("ppl_ratio"),
        "tokens_per_s": quantized.get("tokens_per_s"),
        "peak_vram_gb": quantized.get("peak_vram_gb"),
        "improved_throughput": comparison.get("improved_throughput"),
        "improved_vram": comparison.get("improved_vram"),
        "packed": _packed_artifact(output_dir) if output_dir.is_dir() else False,
        "artifact_dir": _public_artifact_dir(output_dir) if output_dir.exists() else str(output_dir),
        "hub_url": hub_url,
        "verification_status": meta.verification_status,
    }
    by_method: dict[str, dict] = {}
    for old in group.get("methods") or []:
        by_method[str(old.get("method_name"))] = dict(old)
    prev = by_method.get(method_name, {})
    merged = {**prev, **row}
    if not hub_url:
        merged["hub_url"] = prev.get("hub_url")
    by_method[method_name] = merged
    group["methods"] = list(by_method.values())
    dest = _write_group(group)
    group["path"] = str(dest)
    effective_hub = merged.get("hub_url")
    if row["quality_ok"] and effective_hub:
        try:
            group["contribution"] = str(
                export_contribution(job_id=job_id, request=request, hub_url=str(effective_hub))
            )
        except Exception:
            pass
    return group


def export_contribution(*, job_id: str, request: dict, hub_url: str) -> Path:
    meta = jobs_mod.refresh_status(job_id)
    benchmark = _read_json(jobs_mod.job_dir(job_id) / "benchmark.json")
    comparison = benchmark.get("comparison") if isinstance(benchmark.get("comparison"), dict) else {}
    quantized = benchmark.get("quantized") if isinstance(benchmark.get("quantized"), dict) else {}
    baseline = benchmark.get("fp16_baseline") if isinstance(benchmark.get("fp16_baseline"), dict) else {}
    output_dir = resolve_workspace_path(meta.output_dir)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_id": request.get("model_id"),
        "method_name": request.get("method_name"),
        "gpu_instance": request.get("gpu_instance"),
        "gpu_name": request.get("gpu_name") or request.get("gpu"),
        "quality_ok": comparison.get("quality_ok") is True,
        "ppl": quantized.get("perplexity"),
        "ppl_ratio": comparison.get("ppl_ratio"),
        "tokens_per_s": quantized.get("tokens_per_s"),
        "peak_vram_gb": quantized.get("peak_vram_gb"),
        "fp16_ppl": baseline.get("perplexity"),
        "fp16_tokens_per_s": baseline.get("tokens_per_s"),
        "fp16_peak_vram_gb": baseline.get("peak_vram_gb"),
        "packed": _packed_artifact(output_dir) if output_dir.is_dir() else False,
        "hub_url": hub_url,
        "hub_repo_id": _hub_repo_id(hub_url),
        "dataset": benchmark.get("dataset") or "wikitext-2-raw-v1",
        "max_seq_len": benchmark.get("max_seq_len") or 2048,
        "job_id": job_id,
        "slug": request.get("slug"),
        "artifact_dir": _public_artifact_dir(output_dir) if output_dir.exists() else None,
        "verification_status": meta.verification_status,
    }
    payload = validate_contribution(payload, require_hub=True)
    dest = contributions_dir() / contribution_filename(
        str(payload["model_id"]),
        str(payload["gpu_instance"]),
        str(payload["method_name"]),
    )
    atomic_write_text(dest, json.dumps(payload, indent=2) + "\n", mode=0o644)
    rebuild_derived()
    return dest


def accept_contribution(path: Path) -> Path:
    payload = validate_contribution(_read_json(path), require_hub=True)
    dest = contributions_dir() / contribution_filename(
        str(payload["model_id"]),
        str(payload["gpu_instance"]),
        str(payload["method_name"]),
    )
    atomic_write_text(dest, json.dumps(payload, indent=2) + "\n", mode=0o644)
    rebuild_derived()
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter the quantized catalog, or record a WikiText-2 job into it"
    )
    parser.add_argument("--job-id")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--hub-url")
    parser.add_argument("--model-id", help="Filter: Hugging Face model id")
    parser.add_argument("--method", dest="method_name", help="Filter: quantization method")
    parser.add_argument("--gpu-instance", help="Filter: instance type, e.g. g5.2xlarge")
    parser.add_argument("--best", action="store_true", help="Only the best row per matching board")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Print how to download matching Hub repos (does not download)",
    )
    parser.add_argument(
        "--export-contribution",
        action="store_true",
        help="Write compare/contributions/<model>__<gpu>__<method>.json for a PR",
    )
    parser.add_argument(
        "--accept-contribution",
        type=Path,
        help="Validate a contribution JSON and merge it into the catalog",
    )
    parser.add_argument("--rebuild", action="store_true", help="Rewrite catalog.json from groups + contributions")
    args = parser.parse_args()
    try:
        if args.accept_contribution:
            dest = accept_contribution(args.accept_contribution)
            print(json.dumps({"status": "accepted", "path": str(dest)}, indent=2))
            return 0
        if args.rebuild:
            print(json.dumps(rebuild_derived(), indent=2))
            return 0
        if args.job_id:
            if args.request is None:
                raise SystemExit("--request is required with --job-id")
            request = load_request(args.request)
            if args.export_contribution:
                if not args.hub_url:
                    raise SystemExit("--export-contribution needs --hub-url (public Hub repo)")
                dest = export_contribution(
                    job_id=args.job_id, request=request, hub_url=args.hub_url
                )
                print(json.dumps({"status": "exported", "path": str(dest)}, indent=2))
                return 0
            payload = record_job(job_id=args.job_id, request=request, hub_url=args.hub_url)
            if args.best:
                print(json.dumps(payload.get("best"), indent=2))
            else:
                print(json.dumps(payload, indent=2))
            return 0
        rows = query(
            model_id=args.model_id,
            method_name=args.method_name,
            gpu_instance=args.gpu_instance,
            best_only=args.best,
        )
        if args.fetch:
            print(json.dumps({"rows": rows, "fetch": [fetch_howto(row) for row in rows]}, indent=2))
        else:
            print(json.dumps({"rows": rows}, indent=2))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
