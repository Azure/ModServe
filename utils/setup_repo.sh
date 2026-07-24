#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v pixi >/dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi

pixi run setup

# HF_HOME (model cache) lives on the root-owned resource disk; create it once.
HF_HOME=$(pixi run python -c 'from lmm_serving import config as c; print(c.HF_HOME)')
if [ ! -w "$HF_HOME" ]; then
    sudo mkdir -p "$HF_HOME"
    sudo chown "$(id -un):$(id -gn)" "$HF_HOME"
fi

echo "Setup complete. Run experiments with:  pixi run python -m lmm_serving.launcher run monolith <scale>"
