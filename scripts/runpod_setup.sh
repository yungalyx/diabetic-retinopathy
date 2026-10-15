#!/usr/bin/env bash
# Prepare a RunPod pod: system tools, Python environment on local disk, credential and GPU checks.
# Lint, typecheck, and tests run before each push (make check); the pod only checks the GPU.
set -euo pipefail

cd "$(dirname "$0")/.."

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv-cache}"
export UV_LINK_MODE=copy

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

uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
df -h /root /workspace
echo "Setup complete."
