from __future__ import annotations

import pytest

from quant_agent.tools import aws_instance


def test_g5_xlarge_is_24gb():
    spec = aws_instance.lookup("g5.xlarge")
    assert spec.instance_type == "g5.xlarge"
    assert spec.vram_gb == 24.0
    assert spec.gpu_count == 1
    assert spec.gpu == "A10G"


def test_case_and_whitespace_insensitive():
    spec = aws_instance.lookup("  G5.XLarge  ")
    assert spec.instance_type == "g5.xlarge"


def test_multi_gpu_instance_totals():
    spec = aws_instance.lookup("p4d.24xlarge")
    assert spec.gpu_count == 8
    assert spec.vram_gb == 320.0  # already summed in yaml


def test_unknown_instance_raises():
    with pytest.raises(aws_instance.UnknownInstanceType):
        aws_instance.lookup("not.a.real.instance")


def test_known_types_nonempty_and_sorted():
    ts = aws_instance.known_types()
    assert "g5.xlarge" in ts
    assert ts == sorted(ts)
