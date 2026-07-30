#!/usr/bin/env bash
set -euo pipefail

if command -v qneuro-condor >/dev/null 2>&1; then
  exec qneuro-condor run "$@"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec python3 "$script_dir/run_qneuro_condor.py" "$@"
