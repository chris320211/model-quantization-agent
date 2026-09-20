#!/usr/bin/env python3
"""Back-compat alias. Prefer library.py."""
from library import *  # noqa: F401,F403
from library import main

if __name__ == "__main__":
    raise SystemExit(main())
