#!/usr/bin/env python3
"""Run the package's staged script and optional overlay validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_agent.config import host_execution_policy
from quant_agent.port_overlay import validate_overlay_script
from quant_agent.tools.script_io import validate_staged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hyperparameters-json", default="{}")
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--allow-unsafe-host-execution", action="store_true")
    args = parser.parse_args()

    code = args.script.read_text()
    hyperparameters = json.loads(args.hyperparameters_json)
    with host_execution_policy(args.allow_unsafe_host_execution):
        result = validate_staged(
            code,
            args.method_id,
            expected_model_id=args.model_id,
            expected_output_dir=args.output_dir,
            locked_hyperparameters=hyperparameters,
        )
    payload = result.as_dict()
    if result.ok and args.overlay_dir is not None:
        try:
            validate_overlay_script(code, args.overlay_dir)
        except Exception as exc:
            payload = {**payload, "ok": False, "stage": "overlay", "message": str(exc)}
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
