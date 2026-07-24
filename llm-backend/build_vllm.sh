#!/usr/bin/env bash
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_COMMIT="c53dc466b1b802d45d0b61cce36908334ea7a23e"
PATCH_FILE="$HERE/vllm-lmm-serving.patch"
VLLM_SRC_DIR="${VLLM_SRC_DIR:-$HERE/vllm}"
WHEEL_URL="https://wheels.vllm.ai/${BASE_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

echo ">> vLLM base commit : $BASE_COMMIT"
echo ">> patch            : $PATCH_FILE"
echo ">> source dir       : $VLLM_SRC_DIR"

if [ ! -d "$VLLM_SRC_DIR/.git" ]; then
    mkdir -p "$VLLM_SRC_DIR"
    git -C "$VLLM_SRC_DIR" init -q
    git -C "$VLLM_SRC_DIR" remote add origin https://github.com/vllm-project/vllm.git
fi
if ! git -C "$VLLM_SRC_DIR" cat-file -e "${BASE_COMMIT}^{commit}" 2>/dev/null; then
    echo ">> fetching base commit (shallow) ..."
    git -C "$VLLM_SRC_DIR" fetch -q --depth 1 origin "$BASE_COMMIT"
fi

git -C "$VLLM_SRC_DIR" checkout -q -f "$BASE_COMMIT"
git -C "$VLLM_SRC_DIR" clean -fdq
git -C "$VLLM_SRC_DIR" apply --whitespace=nowarn "$PATCH_FILE"
echo ">> applied patch onto pristine base."

if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    echo ">> SKIP_INSTALL=1 -> fetch + patch only; skipping pip install."
    exit 0
fi

export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL_URL"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.7.2}"

echo ">> installing vLLM editable from $VLLM_SRC_DIR (precompiled, no build) ..."
pip install --editable "$VLLM_SRC_DIR"
echo ">> done: vLLM backend installed"
