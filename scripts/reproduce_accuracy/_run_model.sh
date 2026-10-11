#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODEL [reproduce_v19_e2e options...]" >&2
  exit 2
fi

model=$1
shift

case "$model" in
  qwen3-4b|qwen3-8b|qwen3-14b|qwen3-32b|llama2-7b|llama3.1-8b-instruct) ;;
  *)
    echo "unsupported model: $model" >&2
    exit 2
    ;;
esac

package_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$package_root"

python_bin=${PYTHON_BIN:-python}
exec "$python_bin" -m scripts.reproduce_v19_e2e --model "$model" "$@"
