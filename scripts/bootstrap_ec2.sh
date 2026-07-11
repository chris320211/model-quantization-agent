#!/usr/bin/env bash
# Bootstrap per-method venvs on a CUDA EC2 instance.
#
# Prereqs: Ubuntu 22.04 with NVIDIA driver + CUDA 12.1 toolkit installed
# (AWS Deep Learning AMI GPU PyTorch 2.x works out of the box).
#
# Idempotent: skips any .venvs/<name>/ that already has a torch install.
# Venv names MUST match catalog ids in seed/methods.yaml (executor.venv_python
# resolves .venvs/<method_id>/).
# Usage:  bash scripts/bootstrap_ec2.sh [methods...]
#   no args -> install all default methods (awq gptq bnb_nf4 bnb_llm_int8)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ROOT="${REPO_ROOT}/.venvs"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

DEFAULT_METHODS=(awq gptq bnb_nf4 bnb_llm_int8)
METHODS=("${@:-${DEFAULT_METHODS[@]}}")

command -v nvidia-smi >/dev/null || {
  echo "ERROR: nvidia-smi not found. This script must run on a CUDA EC2 instance." >&2
  exit 1
}

command -v "${PYTHON_BIN}" >/dev/null || {
  echo "ERROR: ${PYTHON_BIN} not found. Install Python 3.10+ or set PYTHON_BIN." >&2
  exit 1
}

mkdir -p "${VENV_ROOT}"

make_venv() {
  local name="$1"; shift
  local venv="${VENV_ROOT}/${name}"
  local marker="${venv}/.installed"

  if [[ -f "${marker}" ]]; then
    echo "[skip] ${name} already provisioned (${marker} exists)"
    return 0
  fi

  echo "[create] ${venv}"
  "${PYTHON_BIN}" -m venv "${venv}"
  # shellcheck disable=SC1091
  source "${venv}/bin/activate"

  pip install --upgrade pip wheel
  # torch goes first so method packages resolve against it
  pip install --index-url "${TORCH_INDEX}" "torch==2.3.1"
  pip install \
    "transformers==4.46.3" "accelerate==1.1.1" "safetensors==0.4.5" \
    "sentencepiece==0.2.0" "datasets==3.1.0"

  # method-specific deps
  "$@"

  deactivate
  touch "${marker}"
  echo "[done] ${name}"
}

# AWQ
install_awq() { pip install autoawq; }

# GPTQ (GPTQModel is the current maintained fork)
install_gptq() { pip install gptqmodel datasets; }

# bitsandbytes NF4 / LLM.int8 (both catalog ids install the same package)
install_bnb() { pip install bitsandbytes; }

for m in "${METHODS[@]}"; do
  case "$m" in
    awq)          make_venv awq          install_awq ;;
    gptq)         make_venv gptq         install_gptq ;;
    bnb_nf4)      make_venv bnb_nf4      install_bnb ;;
    bnb_llm_int8) make_venv bnb_llm_int8 install_bnb ;;
    *)            echo "[warn] unknown method '$m' — skipping" ;;
  esac
done

echo
echo "Venvs installed under ${VENV_ROOT}:"
ls -1 "${VENV_ROOT}"
