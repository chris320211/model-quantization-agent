"""Back-compat re-export. The real entry point is now ``orchestrator.run``."""
from __future__ import annotations

from .orchestrator import run

__all__ = ["run"]
