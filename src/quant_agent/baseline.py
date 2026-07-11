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

from .config import REPO_ROOT, child_env, require_host_execution
from .measurement import BENCHMARK_VERSION, run_measurement
from .pareto import Metrics
from .tools.torch_spec import detect_torch_spec
from .runtime_deps import RUNTIME_PACKAGES
from .io_utils import atomic_write_text

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
    require_host_execution("fp16 baseline environment setup")
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
        [str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel"],
        spec.pip_install_argv(str(py)),
        [str(py), "-m", "pip", "install", *RUNTIME_PACKAGES],
    ]
    for step in steps:
        r = subprocess.run(
            step,
            capture_output=True, text=True, timeout=900,
            env=child_env(include_hf=False),
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"fp16 venv install step failed ({' '.join(step)!r}): "
                f"{(r.stderr or r.stdout)[-1500:]}"
            )
    return py


def _cache_key(model_id: str, instance_type: str | None) -> str:
    spec = detect_torch_spec()
    overrides = ",".join(
        f"{k}={os.environ[k]}" for k in sorted(os.environ) if k.startswith("MEASURE_")
    )
    return (
        f"{model_id}::{instance_type or 'unknown'}::bench={BENCHMARK_VERSION}::"
        f"{spec.torch_pin}::{spec.cuda_tag}::{overrides}"
    )


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
        atomic_write_text(_CACHE_PATH, json.dumps(cache, indent=2, sort_keys=True))
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
            return Metrics.from_dict(d)
        except (KeyError, TypeError, ValueError) as e:
            log.warning("fp16 cache entry %s invalid (%s); re-measuring", key, e)

    py = _ensure_venv()
    metrics = run_measurement(
        job_dir=job_dir,
        model_path=model_id,
        venv_python=py,
        timeout_s=timeout_s,
        dtype="float16",
    )

    cache[key] = metrics.to_dict()
    _save_cache(cache)
    return metrics
