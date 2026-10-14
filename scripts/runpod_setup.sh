#!/usr/bin/env bash
# Prepare a fresh RunPod pod: system tools, Python environment, Kaggle credentials check.
set -euo pipefail

cd "$(dirname "$0")/.."

apt-get update -qq
apt-get install -y -qq p7zip-full make

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --locked

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]] && [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    echo "ERROR: Kaggle credentials not found." >&2
    echo "Set KAGGLE_USERNAME and KAGGLE_KEY as pod environment variables," >&2
    echo "or copy kaggle.json to ~/.kaggle/kaggle.json and run: chmod 600 ~/.kaggle/kaggle.json" >&2
    exit 1
fi

make check
df -h .
echo "Setup complete. Next: uv run retina-dr download --raw-dir data/raw"
