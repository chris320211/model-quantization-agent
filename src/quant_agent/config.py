from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    model: str
    chroma_dir: Path
    embed_model: str
    seed_path: Path
    output_dir: Path
    github_token: str | None
    hf_token: str | None


def load_settings() -> Settings:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or key.startswith("sk-ant-REPLACE"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    chroma_dir = Path(os.environ.get("QUANT_AGENT_CHROMA_DIR", "data/chroma"))
    if not chroma_dir.is_absolute():
        chroma_dir = REPO_ROOT / chroma_dir

    output_dir = REPO_ROOT / "out"

    return Settings(
        anthropic_api_key=key,
        model=os.environ.get("QUANT_AGENT_MODEL", "claude-sonnet-4-6"),
        chroma_dir=chroma_dir,
        embed_model=os.environ.get(
            "QUANT_AGENT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        seed_path=REPO_ROOT / "seed" / "methods.yaml",
        output_dir=output_dir,
        github_token=os.environ.get("GITHUB_TOKEN") or None,
        hf_token=os.environ.get("HUGGINGFACE_HUB_TOKEN") or None,
    )
