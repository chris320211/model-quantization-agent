from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from ..config import REPO_ROOT

_PATH = REPO_ROOT / "seed" / "aws_instances.yaml"


@dataclass(frozen=True)
class InstanceSpec:
    instance_type: str
    vram_gb: float
    gpu_count: int
    gpu: str


class UnknownInstanceType(KeyError):
    pass


@lru_cache(maxsize=1)
def _load() -> dict[str, dict]:
    if not _PATH.exists():
        return {}
    with _PATH.open() as f:
        return yaml.safe_load(f) or {}


def lookup(instance_type: str) -> InstanceSpec:
    """Resolve an AWS instance type (e.g. 'g5.xlarge') to its GPU spec.

    Raises UnknownInstanceType if not in seed/aws_instances.yaml.
    """
    data = _load()
    key = instance_type.strip().lower()
    entry = data.get(key)
    if entry is None:
        raise UnknownInstanceType(f"Unknown AWS instance type: {instance_type!r}")
    return InstanceSpec(
        instance_type=key,
        vram_gb=float(entry["vram_gb"]),
        gpu_count=int(entry["gpu_count"]),
        gpu=str(entry["gpu"]),
    )


def known_types() -> list[str]:
    return sorted(_load().keys())
