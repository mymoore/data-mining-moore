#!/usr/bin/env bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
./python.sh -m streamlit run app_coffee.py "$@"
