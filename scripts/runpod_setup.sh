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

has_token=${KAGGLE_API_TOKEN:+yes}
has_legacy=$([[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]] && echo yes || true)
if [[ -z "$has_token" && -z "$has_legacy" && ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    echo "ERROR: Kaggle credentials not found." >&2
    echo "Add KAGGLE_API_TOKEN as a pod environment variable (use a RunPod secret)." >&2
    exit 1
fi

make check
df -h .
echo "Setup complete. Next: uv run retina-dr download --raw-dir data/raw"
