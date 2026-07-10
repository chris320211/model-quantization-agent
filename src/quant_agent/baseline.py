"""fp16 baseline measurement.

Runs the user's model unquantized through the same measurement script as the
quantized iterations, so the tune loop has a "did we beat fp16?" reference
point. The result is cached at ``~/.cache/quant-agent/fp16_baselines.json``
keyed by ``(model_id, instance_type)`` because measuring fp16 a second time
is wasted GPU minutes — the underlying weights don't change between runs.

Uses a synthetic ``_fp16_reference`` venv (built once, reused across runs)
that contains only ``torch + transformers + accelerate + datasets`` — no
quantization library. The venv path matches the layout the executor expects
(``.venvs/<id>/bin/python``) so the same ``run_measurement`` plumbing applies.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from .config import REPO_ROOT, child_env
from .measurement import run_measurement
from .pareto import Metrics
from .tools.torch_spec import detect_torch_spec

log = logging.getLogger(__name__)

_FP16_VENV_ID = "_fp16_reference"
_VENV_ROOT = REPO_ROOT / ".venvs"
_CACHE_PATH = Path(os.path.expanduser("~/.cache/quant-agent/fp16_baselines.json"))


def _venv_dir() -> Path:
    return _VENV_ROOT / _FP16_VENV_ID


def _venv_python() -> Path:
    return _venv_dir() / "bin" / "python"


def _ensure_venv() -> Path:
    """Build .venvs/_fp16_reference/ if missing. Returns the python path.

    Idempotent: if the venv already has torch + transformers, returns immediately.
    """
    py = _venv_python()
    if py.exists():
        return py

    vd = _venv_dir()
    vd.mkdir(parents=True, exist_ok=True)
    log.info("creating fp16 reference venv at %s", vd)

    create = subprocess.run(
        ["python3", "-m", "venv", str(vd)],
        capture_output=True, text=True, timeout=120,
        env=child_env(include_hf=False),
    )
    if create.returncode != 0:
        raise RuntimeError(
            f"fp16 venv creation failed: {create.stderr or create.stdout}"
        )

    spec = detect_torch_spec()
    steps = [
        "pip install --upgrade pip wheel",
        spec.pip_install(),
        "pip install transformers accelerate safetensors sentencepiece datasets",
    ]
    activate = f"source {vd}/bin/activate"
    for step in steps:
        cmd = f"{activate} && {step}"
        r = subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True, text=True, timeout=900,
            env=child_env(include_hf=False),
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"fp16 venv install step failed ({step!r}): "
                f"{(r.stderr or r.stdout)[-1500:]}"
            )
    return py


def _cache_key(model_id: str, instance_type: str | None) -> str:
    return f"{model_id}::{instance_type or 'unknown'}"


def _load_cache() -> dict[str, dict]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("fp16 baseline cache read failed: %s", e)
        return {}


def _save_cache(cache: dict[str, dict]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))
    except OSError as e:
        log.warning("fp16 baseline cache write failed: %s", e)


def measure_fp16_baseline(
    *,
    model_id: str,
    instance_type: str | None,
    job_dir: Path,
    use_cache: bool = True,
    timeout_s: int = 1800,
) -> Metrics:
    """Measure fp16 latency / VRAM / ppl. Cached per (model_id, instance_type).

    ``job_dir`` receives ``measure.py``, ``measure.log``, and ``metrics.json`` —
    same layout as a quantized iteration so downstream reporters don't special-case it.
    """
    key = _cache_key(model_id, instance_type)
    cache = _load_cache()

    if use_cache and key in cache:
        d = cache[key]
        try:
            return Metrics(
                prefill_ms=float(d["prefill_ms"]),
                decode_ms=float(d["decode_ms"]),
                vram_gb=float(d["vram_gb"]),
                ppl=float(d["ppl"]),
            )
        except (KeyError, ValueError) as e:
            log.warning("fp16 cache entry %s invalid (%s); re-measuring", key, e)

    py = _ensure_venv()
    metrics = run_measurement(
        job_dir=job_dir,
        model_path=model_id,
        venv_python=py,
        timeout_s=timeout_s,
    )

    cache[key] = metrics.to_dict()
    _save_cache(cache)
    return metrics
