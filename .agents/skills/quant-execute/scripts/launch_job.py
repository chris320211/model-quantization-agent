#!/usr/bin/env python3
"""Launch through quant_agent.executor so storage and overlay contracts stay canonical."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_agent.config import host_execution_policy
from quant_agent.executor import default_output_dir, launch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--parent-job-id")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--fix-note")
    parser.add_argument("--tune-iter", type=int, default=0)
    parser.add_argument("--hyperparameters-json", default="{}")
    parser.add_argument("--overlay-source", type=Path)
    parser.add_argument("--allow-unsafe-host-execution", action="store_true")
    args = parser.parse_args()

    with host_execution_policy(args.allow_unsafe_host_execution):
        meta = launch(
            method_id=args.method_id,
            model_id=args.model_id,
            script_code=args.script.read_text(),
            output_dir=args.output_dir or default_output_dir(args.method_id, args.model_id),
            parent_job_id=args.parent_job_id,
            attempt=args.attempt,
            fix_note=args.fix_note,
            tune_iter=args.tune_iter,
            hyperparameters=json.loads(args.hyperparameters_json) or None,
            overlay_source=args.overlay_source,
        )
    print(meta.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
