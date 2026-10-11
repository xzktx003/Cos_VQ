#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/_run_model.sh" llama3.1-8b-instruct "$@"
