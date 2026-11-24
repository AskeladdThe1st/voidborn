#!/bin/bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python || ! -f frontend/dist/index.html ]]; then
    echo 'Complete the macOS setup in README.md first.' >&2
    exit 1
fi

echo 'Open http://127.0.0.1:8000 in your browser. Press Ctrl+C to stop.'
exec .venv/bin/python main.py
