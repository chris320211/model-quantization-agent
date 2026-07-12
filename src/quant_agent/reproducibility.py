"""Best-effort, secret-free reproducibility records for quantization jobs."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = "1.0"
MANIFEST_FILENAME = "reproducibility.json"
_RUNTIME_DISTRIBUTIONS = (
    "quant-agent",
    "torch",
    "transformers",
    "accelerate",
    "huggingface-hub",
    "auto-gptq",
    "autoawq",
    "bitsandbytes",
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ReproducibilityManifest:
    schema_version: str
    created_at: str
    model_id: str
    method_id: str
    script_sha256: str
    output_dir: str
    execution_mode: str
    python: dict[str, str]
    platform: dict[str, str]
    runtime_versions: dict[str, str]
    gpu_cuda: dict[str, Any]
    method_repo_commit: str | None
    execution_command: tuple[str, ...] = ()
    overlay_sha256: str | None = None
    overlay_snapshot_path: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _safe_command(args: list[str], *, cwd: Path | None = None) -> str | None:
    """Run a bounded metadata-only command with a credential-free environment."""
    env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH")
        if key in os.environ
    }
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in _RUNTIME_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _gpu_cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    rows = _safe_command(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if rows:
        gpus = []
        for row in rows.splitlines():
            fields = [field.strip() for field in row.split(",")]
            if len(fields) == 4:
                gpus.append(
                    {
                        "name": fields[0],
                        "compute_capability": fields[1],
                        "memory_mib": fields[2],
                        "driver_version": fields[3],
                    }
                )
        if gpus:
            info["gpus"] = gpus
    nvcc = _safe_command(["nvcc", "--version"])
    if nvcc:
        release = re.search(r"release\s+([^,\s]+)", nvcc)
        info["cuda_toolkit"] = release.group(1) if release else nvcc.splitlines()[-1]
    return info


def _method_repo_commit(repo_dir: Path) -> str | None:
    if not repo_dir.exists():
        return None
    commit = _safe_command(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    if commit and _COMMIT_RE.fullmatch(commit.lower()):
        return commit.lower()
    return None


def build_manifest(
    *,
    method_id: str,
    model_id: str,
    script_code: str,
    output_dir: str,
    execution_mode: str,
    method_repo_dir: Path,
    created_at: str | None = None,
    execution_command: list[str] | tuple[str, ...] = (),
    overlay_sha256: str | None = None,
    overlay_snapshot_path: str | None = None,
) -> ReproducibilityManifest:
    return ReproducibilityManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        model_id=model_id,
        method_id=method_id,
        script_sha256=hashlib.sha256(script_code.encode("utf-8")).hexdigest(),
        output_dir=output_dir,
        execution_mode=execution_mode,
        python={
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        platform={
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        runtime_versions=_runtime_versions(),
        gpu_cuda=_gpu_cuda_info(),
        method_repo_commit=_method_repo_commit(method_repo_dir),
        execution_command=tuple(execution_command),
        overlay_sha256=overlay_sha256,
        overlay_snapshot_path=overlay_snapshot_path,
    )


def write_manifest_atomic(manifest: ReproducibilityManifest, job_dir: Path) -> Path:
    """Persist a manifest without exposing partial JSON to concurrent readers."""
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / MANIFEST_FILENAME
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(manifest.to_json())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path
