"""Detect the right torch build for the local GPU.

Ampere/Ada (sm_80–sm_89, e.g. A10G/A100/L40S/L4) → torch 2.3.1 cu121.
Hopper+ (sm_90+, H100/H200/B200) → torch 2.4.1 cu124 (where FlashAttention 2 and
FP8 kernels expect cu124-era runtimes).
Unknown or no GPU → default to the Ampere pin so laptop `--dry` runs still work.

Override via env var ``QUANT_AGENT_TORCH_SPEC`` (format: ``torch==X.Y.Z|cuZZZ``)
so users can pin exotic combinations without editing the code.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass


_DEFAULT_SPEC: "TorchSpec"  # forward declared below
_HOPPER_SPEC: "TorchSpec"


@dataclass(frozen=True)
class TorchSpec:
    torch_pin: str   # e.g. "torch==2.3.1"
    cuda_tag: str    # e.g. "cu121" — used to build the index-url

    @property
    def index_url(self) -> str:
        return f"https://download.pytorch.org/whl/{self.cuda_tag}"

    def pip_install(self) -> str:
        return f"pip install --index-url {self.index_url} {self.torch_pin}"

    def pip_install_argv(self, python: str) -> list[str]:
        """Safe argv form used by subprocess callers (no shell interpolation)."""
        return [python, "-m", "pip", "install", "--index-url", self.index_url, self.torch_pin]


_DEFAULT_SPEC = TorchSpec(torch_pin="torch==2.3.1", cuda_tag="cu121")
_HOPPER_SPEC = TorchSpec(torch_pin="torch==2.4.1", cuda_tag="cu124")


def _parse_override(raw: str) -> TorchSpec | None:
    raw = raw.strip()
    if "|" not in raw:
        return None
    pin, tag = raw.split("|", 1)
    pin = pin.strip()
    tag = tag.strip()
    if not re.fullmatch(r"torch==\d+\.\d+\.\d+(?:[A-Za-z0-9_.+-]*)?", pin):
        return None
    if not re.fullmatch(r"cu\d{3}", tag):
        return None
    return TorchSpec(torch_pin=pin, cuda_tag=tag)


def _compute_capability() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    first = r.stdout.strip().splitlines()
    if not first:
        return None
    try:
        return float(first[0].strip())
    except ValueError:
        return None


def detect_torch_spec() -> TorchSpec:
    override_raw = os.environ.get("QUANT_AGENT_TORCH_SPEC", "").strip()
    if override_raw:
        parsed = _parse_override(override_raw)
        if parsed is not None:
            return parsed

    cc = _compute_capability()
    if cc is not None and cc >= 9.0:
        return _HOPPER_SPEC
    return _DEFAULT_SPEC
