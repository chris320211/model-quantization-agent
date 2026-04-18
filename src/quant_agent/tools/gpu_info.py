from __future__ import annotations

import json
import shutil
import subprocess

from langchain_core.tools import tool


def _nvidia_smi() -> list[dict] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    gpus = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, total_mib, free_mib, driver = parts
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "vram_gb_total": round(int(total_mib) / 1024, 2),
                "vram_gb_free": round(int(free_mib) / 1024, 2),
                "driver_version": driver,
            }
        )
    return gpus


@tool
def gpu_info() -> str:
    """Detect local CUDA GPUs and their total/free VRAM via nvidia-smi.

    Use this on the EC2 box before calling recommend_quantization, so you can
    pass the real VRAM budget instead of asking the user. Returns a JSON string
    with a list of GPUs and an aggregated 'total_vram_gb' field (sum across cards).
    """
    gpus = _nvidia_smi()
    if gpus is None:
        return json.dumps(
            {"error": "nvidia-smi not available — not on a CUDA host, or driver missing."}
        )
    if not gpus:
        return json.dumps({"error": "No GPUs detected."})
    total = round(sum(g["vram_gb_total"] for g in gpus), 2)
    return json.dumps({"gpus": gpus, "count": len(gpus), "total_vram_gb": total}, indent=2)
