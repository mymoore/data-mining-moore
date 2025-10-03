#!/usr/bin/env bash
# px.sh — ensure env is ready, then run px.py in it (passes args through)
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
./python.sh px.py "$@"
