#!/bin/bash
# Entrypoint: download models if needed, then start llama-server

set -e

MODEL_DIR="${VLM_MODEL_DIR:-/app/models}"
MODEL_FILE="${VLM_MODEL_FILE:-Qwen3-VL-4B-Instruct-Q8_0.gguf}"
MMPROJ_FILE="${VLM_MMPROJ_FILE:-mmproj-F16.gguf}"

# Download models if missing
/app/download_models.sh

# Detect llama-server version and use compatible arguments
# Newer versions use --n-parallel, older use -np or don't support it
PARALLEL_ARG=""
if /app/llama-server --help 2>&1 | grep -q '\-\-n-parallel'; then
    PARALLEL_ARG="--n-parallel 1"
elif /app/llama-server --help 2>&1 | grep -q '\-np'; then
    PARALLEL_ARG="-np 1"
fi
# If neither found, omit the argument (defaults to 1 anyway)

# Start llama-server with VLM configuration
# Context of 8192 is enough for most images (4000+ tokens)
exec /app/llama-server \
    --model "$MODEL_DIR/$MODEL_FILE" \
    --mmproj "$MODEL_DIR/$MMPROJ_FILE" \
    --alias "Qwen3-VL-4B" \
    --n-gpu-layers 999 \
    --ctx-size 8192 \
    $PARALLEL_ARG \
    --batch-size 1024 \
    --ubatch-size 256 \
    --port 8080 \
    --flash-attn \
    --jinja \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    "$@"
