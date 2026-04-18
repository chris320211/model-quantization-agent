"""Test-wide setup. Runs before any test module imports quant_agent."""
from __future__ import annotations

import os

# config.load_settings() rejects the .env.example placeholder ("sk-ant-REPLACE..."),
# so overwrite it for test runs before anything imports quant_agent.
if os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-REPLACE") or not os.environ.get(
    "ANTHROPIC_API_KEY"
):
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
