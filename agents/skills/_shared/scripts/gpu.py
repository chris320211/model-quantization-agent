#!/usr/bin/env python3
"""Record GPU facts from --gpu-instance plus live nvidia-smi when present."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import shutil
import subprocess

from env import child_env
from paths import REPO_ROOT

_AWS_CANDIDATES = (
    REPO_ROOT / "agents" / "skills" / "_shared" / "reference" / "aws_instances.yaml",
)
_GPU_SPEC_CANDIDATES = (
    REPO_ROOT / "agents" / "skills" / "_shared" / "reference" / "gpu_specs.yaml",
)


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _first_yaml(candidates: tuple[Path, ...]) -> dict:
    for path in candidates:
        data = _load_yaml(path)
        if data:
            return data
    return {}


def _nvidia_smi() -> list[dict] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            env=child_env(include_hf=False),
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, total_mib, free_mib, driver = parts
        try:
            total = int(total_mib)
            free = int(free_mib)
        except ValueError:
            continue
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "vram_gb_total": round(total / 1024, 2),
                "vram_gb_free": round(free / 1024, 2),
                "driver_version": driver,
            }
        )
    return gpus


def gpu_facts(gpu_instance: str) -> dict:
    instance = gpu_instance.strip()
    facts: dict = {
        "gpu_instance": instance,
        "instance_type": instance,
    }

    catalog = _first_yaml(_AWS_CANDIDATES)
    entry = catalog.get(instance.lower()) if catalog else None
    if isinstance(entry, dict):
        gpu = str(entry.get("gpu", ""))
        facts["instance_type"] = instance.lower()
        if "vram_gb" in entry:
            facts["vram_gb"] = float(entry["vram_gb"])
        if "gpu_count" in entry:
            facts["gpu_count"] = int(entry["gpu_count"])
        if gpu:
            facts["gpu"] = gpu
        specs = _first_yaml(_GPU_SPEC_CANDIDATES)
        gpu_spec = specs.get(gpu, {}) if specs else {}
        if isinstance(gpu_spec, dict):
            if gpu_spec.get("gpu_arch"):
                facts["gpu_arch"] = gpu_spec["gpu_arch"]
            if gpu_spec.get("compute_capability") is not None:
                facts["compute_capability"] = float(gpu_spec["compute_capability"])

    gpus = _nvidia_smi()
    facts["nvidia_smi"] = gpus is not None
    if gpus:
        facts["gpus"] = gpus
        facts["gpu_count"] = facts.get("gpu_count") or len(gpus)
        facts["vram_gb_free"] = round(sum(g["vram_gb_free"] for g in gpus), 2)
        facts["vram_gb_total_live"] = round(sum(g["vram_gb_total"] for g in gpus), 2)
        facts["gpu_name"] = gpus[0]["name"]
        if "vram_gb" not in facts:
            facts["vram_gb"] = facts["vram_gb_total_live"]
        if "gpu" not in facts:
            facts["gpu"] = gpus[0]["name"]
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description="Print GPU facts as JSON")
    parser.add_argument("--gpu-instance", required=True)
    args = parser.parse_args()
    print(json.dumps(gpu_facts(args.gpu_instance), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
