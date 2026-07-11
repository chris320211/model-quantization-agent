#!/usr/bin/env python
"""Run the canonical packaged measurement program.

Kept as a tiny skill artifact so measurement behavior cannot drift from the Python
pipeline. ``quant-agent`` must be installed in the active environment.
"""
from quant_agent.measurement import MEASURE_SCRIPT

exec(compile(MEASURE_SCRIPT, "<quant-agent-measurement>", "exec"))
