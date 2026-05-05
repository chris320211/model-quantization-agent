#!/usr/bin/env bash
# Securely load model-quantization-agent .env into the current shell.
#
# Usage:
#   source /home/ubuntu/model-quantization-agent/.claude/skills/_shared/load_env.sh
#
# Or with an explicit path:
#   source .../_shared/load_env.sh /path/to/.env
#
# Behavior:
#   - Refuses to source a world-readable .env (mode != 600); auto-tightens to 600.
#   - Exports every KEY=value line via `set -a` (matches python-dotenv semantics
#     for the format `quant-agent setup` writes).
#   - Aliases HUGGINGFACE_HUB_TOKEN -> HF_TOKEN so libraries reading either name
#     see the same value.
#   - Never echoes secret values; only echoes key NAMES that were loaded.
#   - Returns non-zero on missing file so callers can `|| { ...; exit 1; }`.

_quant_load_env() {
  local env_file="${1:-${QUANT_AGENT_ENV:-/home/ubuntu/model-quantization-agent/.env}}"

  if [[ ! -f "$env_file" ]]; then
    echo "[env] $env_file not found." >&2
    echo "[env] Run \`quant-agent setup\` (hidden input via getpass, writes mode 600)" >&2
    echo "[env] or invoke the /quant-setup skill for guided setup." >&2
    return 1
  fi

  # Permission hardening — never source a file other users can read.
  local mode
  mode=$(stat -c %a "$env_file" 2>/dev/null || stat -f %A "$env_file" 2>/dev/null)
  if [[ -z "$mode" ]]; then
    echo "[env] could not stat $env_file" >&2
    return 1
  fi
  if [[ "$mode" != "600" ]]; then
    echo "[env] tightening $env_file from mode $mode to 600" >&2
    chmod 600 "$env_file" || {
      echo "[env] chmod 600 failed; refusing to source" >&2
      return 1
    }
  fi

  # Capture the keys we are about to set so we can announce them by name only.
  local loaded_keys
  loaded_keys=$(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$env_file" | cut -d= -f1 | tr '\n' ' ')

  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  local rc=$?
  set +a

  if [[ $rc -ne 0 ]]; then
    echo "[env] failed to source $env_file (rc=$rc)" >&2
    return $rc
  fi

  # HuggingFace libraries are split between two env-var names. Normalize.
  if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
  elif [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  fi

  echo "[env] loaded keys from $env_file: $loaded_keys" >&2
  return 0
}

_quant_load_env "$@"
