#!/usr/bin/env bash
# Strictly parse model-quantization-agent credentials into the current shell.
#
# Usage:
#   source /home/ubuntu/model-quantization-agent/.claude/skills/_shared/load_env.sh
#
# Or with an explicit path:
#   source .../_shared/load_env.sh /path/to/.env
#
# Behavior:
#   - Refuses a world-readable credential file (mode != 600); auto-tightens to 600.
#   - Parses an explicit key allowlist without evaluating shell syntax.
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

  # Parse data; never `source` the file. Only the documented keys are accepted,
  # so shell syntax and command substitutions remain inert text.
  local loaded_keys="" line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ ! "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      echo "[env] malformed assignment; refusing to load" >&2
      return 1
    fi
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    case "$key" in
      OPENAI_API_KEY|GITHUB_TOKEN|HUGGINGFACE_HUB_TOKEN|HF_TOKEN|QUANT_AGENT_MODEL|QUANT_AGENT_REASONING_EFFORT|QUANT_AGENT_*_MODEL|QUANT_AGENT_*_REASONING_EFFORT|QUANT_AGENT_TORCH_SPEC)
        export "$key=$value"
        loaded_keys+="$key "
        ;;
      *)
        echo "[env] unsupported key $key; refusing to load" >&2
        return 1
        ;;
    esac
  done < "$env_file"

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
