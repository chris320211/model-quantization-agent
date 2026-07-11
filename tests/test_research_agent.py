from quant_agent.research_agent import _parse_input


def test_parse_input_extracts_constraints_without_polluting_model_phrase():
    parsed = _parse_input(
        "quantize llama2 7b to g5.xlarge 4-bit for vllm speed priority no calibration"
    )
    assert parsed.instance_phrase == "g5.xlarge"
    assert parsed.target_bits == 4
    assert parsed.backend == "vllm"
    assert parsed.priority == "speed"
    assert parsed.have_calibration_data is False
    assert parsed.model_phrase == "llama2 7b"


def test_parse_input_extracts_qat_and_kv_cache_flags():
    parsed = _parse_input("quantize org/model for p5.48xlarge using QAT KV-cache")
    assert parsed.allow_qat is True
    assert parsed.need_kv_cache_quant is True
    assert parsed.model_phrase == "org/model"
