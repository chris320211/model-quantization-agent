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
    voyage_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket_name: str
    seed_path: Path
    output_dir: Path
    github_token: str | None
    hf_token: str | None


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or key.startswith("sk-ant-REPLACE"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    def _require(name: str) -> str:
        val = os.environ.get(name, "").strip()
        if not val:
            raise RuntimeError(f"{name} is not set in .env")
        return val

    return Settings(
        anthropic_api_key=key,
        model=os.environ.get("QUANT_AGENT_MODEL", "claude-sonnet-4-6"),
        voyage_api_key=_require("VOYAGE_API_KEY"),
        qdrant_url=_require("QDRANT_URL"),
        qdrant_api_key=_require("QDRANT_API_KEY"),
        r2_account_id=_require("R2_ACCOUNT_ID"),
        r2_access_key_id=_require("R2_ACCESS_KEY_ID"),
        r2_secret_access_key=_require("R2_SECRET_ACCESS_KEY"),
        r2_bucket_name=_require("R2_BUCKET_NAME"),
        seed_path=REPO_ROOT / "seed" / "methods.yaml",
        output_dir=REPO_ROOT / "out",
        github_token=os.environ.get("GITHUB_TOKEN") or None,
        hf_token=os.environ.get("HUGGINGFACE_HUB_TOKEN") or None,
    )
