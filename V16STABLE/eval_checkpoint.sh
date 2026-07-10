#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/checkpoint.pt" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWANLAB_TRACKING_MODE=disabled
exec "${SCRIPT_DIR}/run_train.sh" --eval-only-checkpoint "$1"
