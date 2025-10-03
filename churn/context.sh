#!/usr/bin/env bash

function context_dir() {
    echo "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
}

function conda_exe() {
    local cmd bin dist
    for cmd in conda; do
        for dist in miniforge3 miniconda3; do
            if command -v $cmd >/dev/null 2>&1 || [ -x "$HOME/$dist/bin/$cmd" ]; then
                if [ $# -eq 0 ]; then
                    echo $cmd
                else
                    "$cmd" "$@"
                fi
                return
            fi
        done
    done
    echo "ERROR: neither 'mamba' nor 'conda' found in PATH." 1>&2
    exit 1
}

function conda_venv() {
    echo "$(context_dir)/.venv"
}

function python_exe() {
    conda_exe run -p "$(conda_venv)" python "$@"
}

function R_exe() {
    conda_exe run -p "$(conda_venv)" R "$@"
}

function jupyter_exe() {
    conda_exe run -p "$(conda_venv)" jupyter "$@"
}

function config_json() {
    local file="$(context_dir)/config.json"
    if [ $# -eq 0 ]; then
        echo "$file"
    else
        jq "$@" "$file"
    fi
}

export -f conda_exe
export -f conda_venv
export -f python_exe
export -f R_exe
export -f jupyter_exe
export -f context_dir
export -f config_json

if [[ "${BASH_SOURCE[0]:-}" == "${0:-}" ]]; then
    "$@"
fi
