"""Live GPU probe via nvidia-smi, merged with static InstanceSpec yaml data.

Used by the research agent to feed real-world GPU state (free VRAM, driver
version, ECC mode) into the recommendation prompt. Falls back to static-only
specs when no GPU is present (e.g. CI, planning-only runs).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass

from .tools.aws_instance import InstanceSpec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareProfile:
    """Static spec + observed runtime state. Fields default to None when unknown."""
    instance_type: str | None
    gpu: str | None
    gpu_count: int | None
    gpu_arch: str | None
    compute_capability: float | None
    vram_gb_total: float | None
    vram_gb_free: float | None
    memory_bandwidth_gb_s: float | None
    peak_fp16_tflops: float | None
    int8_tops: float | None
    driver_version: str | None
    ecc_mode: str | None
    probe_ok: bool

    def to_dict(self) -> dict:
        return asdict(self)


_NVIDIA_QUERY = "name,memory.total,memory.free,driver_version,ecc.mode.current"


def _run_nvidia_smi() -> list[dict] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={_NVIDIA_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError) as e:
        log.warning("nvidia-smi probe failed: %s", e)
        return None

    rows: list[dict] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        name, mem_total, mem_free, driver, ecc = parts
        try:
            rows.append(
                {
                    "name": name,
                    "memory_total_mib": float(mem_total),
                    "memory_free_mib": float(mem_free),
                    "driver_version": driver,
                    "ecc_mode": ecc,
                }
            )
        except ValueError:
            continue
    return rows or None


def probe_live(spec: InstanceSpec | None = None) -> HardwareProfile:
    """Merge static InstanceSpec with live nvidia-smi state.

    When spec is None and no GPU is visible, returns a profile with probe_ok=False
    and only the fields that could be filled. Never raises — failure tolerant by
    design so the research agent can still run on planning-only machines.
    """
    rows = _run_nvidia_smi()

    if spec is None and rows is None:
        return HardwareProfile(
            instance_type=None, gpu=None, gpu_count=None, gpu_arch=None,
            compute_capability=None, vram_gb_total=None, vram_gb_free=None,
            memory_bandwidth_gb_s=None, peak_fp16_tflops=None, int8_tops=None,
            driver_version=None, ecc_mode=None, probe_ok=False,
        )

    if rows:
        first = rows[0]
        vram_total_gb = first["memory_total_mib"] / 1024.0
        vram_free_gb = first["memory_free_mib"] / 1024.0
        driver = first["driver_version"]
        ecc = first["ecc_mode"]
    else:
        vram_total_gb = vram_free_gb = None
        driver = ecc = None

    if spec is None:
        # Live-only: we can't infer arch/bandwidth without the yaml, so leave None.
        return HardwareProfile(
            instance_type=None, gpu=rows[0]["name"] if rows else None,
            gpu_count=len(rows) if rows else None, gpu_arch=None,
            compute_capability=None, vram_gb_total=vram_total_gb,
            vram_gb_free=vram_free_gb, memory_bandwidth_gb_s=None,
            peak_fp16_tflops=None, int8_tops=None,
            driver_version=driver, ecc_mode=ecc, probe_ok=rows is not None,
        )

    return HardwareProfile(
        instance_type=spec.instance_type,
        gpu=spec.gpu,
        gpu_count=spec.gpu_count,
        gpu_arch=spec.gpu_arch,
        compute_capability=spec.compute_capability,
        vram_gb_total=vram_total_gb if vram_total_gb is not None else spec.vram_gb,
        vram_gb_free=vram_free_gb,
        memory_bandwidth_gb_s=spec.memory_bandwidth_gb_s,
        peak_fp16_tflops=spec.peak_fp16_tflops,
        int8_tops=spec.int8_tops,
        driver_version=driver,
        ecc_mode=ecc,
        probe_ok=rows is not None,
    )
