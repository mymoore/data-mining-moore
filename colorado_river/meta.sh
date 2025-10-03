#!/usr/bin/env bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1

# Allow an optional leading "--" for consistency with other wrappers
ARGS=( )
SKIP_FIRST=false
for arg in "$@"; do
  if ! $SKIP_FIRST && [[ "$arg" == "--" ]]; then
    SKIP_FIRST=true
    continue
  fi
  ARGS+=("$arg")
done

./python.sh meta.py "${ARGS[@]}"
