from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    model: str
    seed_path: Path
    output_dir: Path
    github_token: str | None
    hf_token: str | None


# Environment variables safe to expose to child processes. Deliberately excludes
# every cloud credential (ANTHROPIC/GITHUB) — those are only needed by the parent
# agent, never by a launched quantization script, a cloned repo's setup.py, or a
# measurement subprocess. See child_env().
_CHILD_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TERM",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "TORCH_CUDA_ARCH_LIST",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
)


def child_env(extra: dict[str, str] | None = None, *, include_hf: bool = True) -> dict[str, str]:
    """Build a minimal environment for a child process.

    Starts from an allowlist (``_CHILD_ENV_ALLOWLIST``) rather than the parent's full
    ``os.environ`` so cloud secrets loaded by ``load_dotenv()`` never reach untrusted
    code (LLM-authored scripts, cloned-repo ``setup.py``, measurement subprocesses).

    Args:
        extra:      Additional key/values to set (merged last; overrides allowlist).
        include_hf: When True, forward the HuggingFace token under both
                    ``HUGGINGFACE_HUB_TOKEN`` and ``HF_TOKEN`` (needed to load gated
                    models). Pass False for children that never touch the Hub.
    """
    env: dict[str, str] = {
        k: os.environ[k] for k in _CHILD_ENV_ALLOWLIST if k in os.environ
    }
    if include_hf:
        tok = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if tok:
            env["HUGGINGFACE_HUB_TOKEN"] = tok
            env["HF_TOKEN"] = tok
    if extra:
        env.update(extra)
    return env


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or key.startswith("sk-ant-REPLACE"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    return Settings(
        anthropic_api_key=key,
        model=os.environ.get("QUANT_AGENT_MODEL", "claude-sonnet-4-6"),
        seed_path=REPO_ROOT / "seed" / "methods.yaml",
        output_dir=REPO_ROOT / "out",
        github_token=os.environ.get("GITHUB_TOKEN") or None,
        hf_token=os.environ.get("HUGGINGFACE_HUB_TOKEN") or None,
    )
