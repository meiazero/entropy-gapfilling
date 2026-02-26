#!/bin/bash
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')

if echo "$COMMAND" | grep -qE "(^|&&\s*|;\s*)(jupyter|jupyter-notebook|jupyter-lab|ipython notebook)"; then
  echo "Blocked: .py scripts only - no Jupyter notebooks" >&2
  exit 2
fi

exit 0
