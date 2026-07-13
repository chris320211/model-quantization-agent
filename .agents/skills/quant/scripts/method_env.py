#!/usr/bin/env python3
"""Acquire a catalog-pinned method repo and build its canonical venv."""
from __future__ import annotations

import argparse

from quant_agent.config import host_execution_policy
from quant_agent.tools.repo_tool import clone_method_repo, install_method_venv


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    clone = sub.add_parser("clone")
    clone.add_argument("--method-id", required=True)
    clone.add_argument("--repo-url", required=True)

    install = sub.add_parser("install")
    install.add_argument("--method-id", required=True)
    install.add_argument("--step", action="append", default=[])

    parser.add_argument("--allow-unsafe-host-execution", action="store_true")
    args = parser.parse_args()

    with host_execution_policy(args.allow_unsafe_host_execution):
        if args.command == "clone":
            result = clone_method_repo.invoke(
                {"method_id": args.method_id, "repo_url": args.repo_url}
            )
        else:
            result = install_method_venv.invoke(
                {"method_id": args.method_id, "install_steps": args.step}
            )
    print(result)
    return 0 if '"status": "ok"' in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
