#!/usr/bin/env python3
"""Invoke the package's canonical deterministic compatibility engine."""
from __future__ import annotations

import argparse
import json

from quant_agent.compatibility import CompatibilityRequest, evaluate_catalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-b", type=float)
    parser.add_argument("--vram-gb", type=float)
    parser.add_argument("--compute-capability", type=float)
    parser.add_argument("--gpu-arch")
    parser.add_argument("--architecture", action="append", default=[])
    parser.add_argument("--target-bits", type=int)
    parser.add_argument("--backend")
    parser.add_argument(
        "--calibration", choices=("available", "unavailable", "unknown"), default="unknown"
    )
    parser.add_argument("--allow-qat", action="store_true")
    parser.add_argument("--need-activation-quant", action="store_true")
    parser.add_argument("--need-kv-cache-quant", action="store_true")
    args = parser.parse_args()

    calibration = {
        "available": True,
        "unavailable": False,
        "unknown": None,
    }[args.calibration]
    request = CompatibilityRequest(
        params_b=args.params_b,
        vram_gb=args.vram_gb,
        compute_capability=args.compute_capability,
        gpu_arch=args.gpu_arch,
        architectures=args.architecture,
        target_bits=args.target_bits,
        backend=args.backend,
        have_calibration_data=calibration,
        allow_qat=args.allow_qat,
        need_activation_quant=args.need_activation_quant,
        need_kv_cache_quant=args.need_kv_cache_quant,
    )
    print(json.dumps([decision.model_dump() for decision in evaluate_catalog(request)], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
