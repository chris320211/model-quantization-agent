# EC2 requirements

## Recommended AMI
**AWS Deep Learning AMI GPU PyTorch 2.x — Ubuntu 22.04.** Ships with NVIDIA driver, CUDA 12.1 toolkit, and Python 3.10 preinstalled, so `bootstrap_ec2.sh` just layers the quant libs on top.

## Is it AMI-specific?
No — any AMI that satisfies these works:
- NVIDIA driver (matching CUDA 12.x)
- CUDA 12.1 toolkit (for torch wheels from `cu121` index)
- Python ≥ 3.10
- `bash`, `git`

On a bare Ubuntu 22.04 AMI you'd install the NVIDIA driver yourself. For a different CUDA version, override the torch wheel index:
```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash scripts/bootstrap_ec2.sh
```
Amazon Linux 2023 DLAMI also works.

## Is it instance-specific?
**Must be an NVIDIA-GPU instance.** The executor requires `nvidia-smi`, and every quant library compiles CUDA kernels.

Suggested types:
| Model size | Instance | VRAM |
|---|---|---|
| ≤ 13B @ 4-bit | `g5.xlarge`, `g6.xlarge` | 24 GB |
| ≤ 34B @ 4-bit | `g6e.xlarge` | 48 GB |
| ≤ 70B @ 4-bit | `g5.12xlarge` (4× A10G) | 96 GB |
| ≥ 70B, W4A4 research | `p4d.24xlarge`, `p5.48xlarge` | 320+ GB |

Will **not** run on: CPU-only (`t3`, `m7i`, ...), Graviton/ARM (`g5g`, `c7g`), Trainium/Inferentia (`trn1`, `inf2`). Size the EBS volume ≥ 2× the fp16 model weights — quantized output sits next to the source.
