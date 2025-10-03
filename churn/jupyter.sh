#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")"/context.sh
if ! jupyter_exe "$@"
then
  echo "ERROR: jupyter_exe $@ failed" 1>&2
  exit 1
fi
