#!/bin/sh
# Git credential helper for GITHUB_TOKEN. Never logs. Do not call from chat.
# GIT_ASKPASS points here; git provides the prompt as $1.
case "$1" in
  *[Uu]sername*) printf '%s\n' "x-access-token" ;;
  *) printf '%s\n' "${GITHUB_TOKEN-}" ;;
esac
