from quant_agent import hardware_probe
from quant_agent.tools.aws_instance import InstanceSpec


def test_probe_live_aggregates_all_gpu_memory(monkeypatch):
    rows = [
        {
            "name": "A10G", "memory_total_mib": 24 * 1024,
            "memory_free_mib": 20 * 1024, "driver_version": "1", "ecc_mode": "Off",
        },
        {
            "name": "A10G", "memory_total_mib": 24 * 1024,
            "memory_free_mib": 18 * 1024, "driver_version": "1", "ecc_mode": "Off",
        },
    ]
    monkeypatch.setattr(hardware_probe, "_run_nvidia_smi", lambda: rows)
    spec = InstanceSpec(instance_type="x", vram_gb=48, gpu_count=2, gpu="A10G")
    profile = hardware_probe.probe_live(spec)
    assert profile.vram_gb_total == 48
    assert profile.vram_gb_free == 38
