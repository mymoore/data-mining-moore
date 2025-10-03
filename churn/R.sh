#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")"/context.sh
if ! R_exe "$@"
then
  echo "ERROR: R_exe $@ failed" 1>&2
  exit 1
fi
