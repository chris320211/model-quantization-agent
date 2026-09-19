#!/usr/bin/env python3
"""Download a Hugging Face model snapshot into .cache/hf-snapshots/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import re

from huggingface_hub import snapshot_download

from env import child_env
from paths import HF_SNAP_ROOT

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")


def safe_snapshot_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.strip())


def _redact(message: str, token: str | None) -> str:
    if token:
        return message.replace(token, "***")
    return message


def snapshot_model(model_id: str) -> Path:
    trimmed = model_id.strip()
    if not _MODEL_ID_RE.fullmatch(trimmed):
        raise ValueError(f"model_id is not org/model: {model_id!r}")

    hf_env = child_env(include_hf=True)
    token = hf_env.get("HUGGINGFACE_HUB_TOKEN") or hf_env.get("HF_TOKEN")

    dest = HF_SNAP_ROOT / safe_snapshot_id(trimmed)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id=trimmed, local_dir=str(dest), token=token)
    except Exception as exc:
        raise RuntimeError(_redact(str(exc), token)) from None
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot a Hugging Face model onto this machine")
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    try:
        path = snapshot_model(args.model_id)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
