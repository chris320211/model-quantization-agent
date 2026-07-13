#!/usr/bin/env python3
"""Write a validated, content-addressed port overlay bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_agent.config import REPO_ROOT
from quant_agent.port_overlay import PortOverlaySession


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-commit")
    parser.add_argument("--patch-file", type=Path, required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--evidence-file", action="append", default=[])
    parser.add_argument("--target-module", action="append", default=[])
    args = parser.parse_args()

    safe_model = "".join(c if c.isalnum() or c in "._-" else "_" for c in args.model_id).strip("_")
    session = PortOverlaySession(
        root=REPO_ROOT / "out" / "overlays" / args.method_id / safe_model,
        method_id=args.method_id,
        model_id=args.model_id,
        base_commit=args.base_commit,
    )
    result = session.write(
        patch=args.patch_file.read_text(),
        rationale=args.rationale,
        evidence_files=args.evidence_file,
        target_modules=args.target_module,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
