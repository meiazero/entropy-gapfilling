#!/bin/bash
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')

if echo "$COMMAND" | grep -qE "(^|&&\s*|;\s*)python "; then
  echo "Blocked: use 'uv run' instead of raw 'python'" >&2
  exit 2
fi

if echo "$COMMAND" | grep -qE "(^|&&\s*|;\s*)pip (install|uninstall)"; then
  echo "Blocked: use 'uv pip' instead of raw 'pip'" >&2
  exit 2
fi

if echo "$COMMAND" | grep -qE "(^|&&\s*|;\s*)(python -m venv|virtualenv) "; then
  echo "Blocked: use 'uv venv' instead of 'python -m venv' or 'virtualenv'" >&2
  exit 2
fi

if echo "$COMMAND" | grep -qE "(^|&&\s*|;\s*)conda (install|create|activate)"; then
  echo "Blocked: use 'uv' instead of 'conda'" >&2
  exit 2
fi

exit 0
