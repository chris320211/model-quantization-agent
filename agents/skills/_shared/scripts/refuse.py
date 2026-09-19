#!/usr/bin/env python3
"""Refuse papers that are not weight-PTQ methods for an existing HF checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import re

_QAT = re.compile(r"\bqat\b|quantization[ -]aware training", re.I)
_QAT_TRAINING_ONLY = re.compile(
    r"trained from scratch|requires (?:re)?training|not a (?:ptq|post-training)|"
    r"cannot quantize an existing|fine-tuning compute|data-free qat",
    re.I,
)
_KV = re.compile(r"kv[ -]cache|key-value cache", re.I)
_KV_ONLY = re.compile(
    r"does not (?:reduce|quantize|save) (?:the )?(?:model )?weights|"
    r"no weight (?:save|quantization)|"
    r"kv-cache-only|only (?:the )?kv[ -]cache|"
    r"orthogonal to weight|"
    r"does not produce (?:a )?(?:saved )?(?:weight|checkpoint)|"
    r"runtime kv[ -]cache",
    re.I,
)
_BITNET = re.compile(r"\bbitnet\b", re.I)
_BITNET_158 = re.compile(r"1\.58|b1\.58|1\.58-bit", re.I)


def classify_paper(text: str) -> dict | None:
    """Return a refusal payload, or None if the paper looks usable as weight PTQ."""
    if _BITNET.search(text) and _BITNET_158.search(text):
        return {
            "status": "refused",
            "reason": "bitnet_1_58",
            "detail": "BitNet 1.58 is trained from scratch, not a Llama/HF PTQ method.",
        }
    if _QAT.search(text) and _QAT_TRAINING_ONLY.search(text):
        return {
            "status": "refused",
            "reason": "qat_training_only",
            "detail": "Paper looks like QAT-training-only; it does not quantize an existing checkpoint.",
        }
    if _KV.search(text) and _KV_ONLY.search(text):
        return {
            "status": "refused",
            "reason": "kv_cache_only",
            "detail": "Paper looks like KV-cache-only quantization with no saved weight artifact.",
        }
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Refuse QAT-only, KV-cache-only, or BitNet 1.58 papers")
    parser.add_argument("paper_path", type=Path)
    args = parser.parse_args()
    path = args.paper_path
    if not path.is_file():
        print(json.dumps({"status": "error", "error": f"paper text not found: {path}"}))
        return 1
    text = path.read_text(errors="replace")
    refusal = classify_paper(text)
    if refusal is None:
        print(json.dumps({"status": "ok"}))
        return 0
    print(json.dumps(refusal))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
