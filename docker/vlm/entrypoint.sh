#!/bin/bash
# Entrypoint: download models if needed, then start llama-server

set -e

MODEL_DIR="${VLM_MODEL_DIR:-/app/models}"
MODEL_FILE="${VLM_MODEL_FILE:-Qwen3-VL-4B-Instruct-Q8_0.gguf}"
MMPROJ_FILE="${VLM_MMPROJ_FILE:-mmproj-F16.gguf}"

# Download models if missing
/app/download_models.sh

# Get llama-server help text once for version detection
HELP_TEXT=$(/app/llama-server --help 2>&1 || true)

# Detect llama-server version and use compatible arguments
# Newer versions use --n-parallel, older use -np or don't support it
PARALLEL_ARG=""
if echo "$HELP_TEXT" | grep -q '\-\-n-parallel'; then
    PARALLEL_ARG="--n-parallel 1"
elif echo "$HELP_TEXT" | grep -q '\-np'; then
    PARALLEL_ARG="-np 1"
fi

# --flash-attn requires value in some versions (on|off|auto), not in others
FLASH_ARG=""
if echo "$HELP_TEXT" | grep -q '\-\-flash-attn \[on|off|auto\]'; then
    FLASH_ARG="--flash-attn on"
elif echo "$HELP_TEXT" | grep -q '\-\-flash-attn'; then
    FLASH_ARG="--flash-attn"
fi

# Start llama-server with VLM configuration
# Context of 32768 supports high-res images (4000+ tokens) + 2048 output tokens
exec /app/llama-server \
    --model "$MODEL_DIR/$MODEL_FILE" \
    --mmproj "$MODEL_DIR/$MMPROJ_FILE" \
    --alias "Qwen3-VL-4B" \
    --n-gpu-layers 999 \
    --ctx-size 32768 \
    $PARALLEL_ARG \
    --batch-size 2048 \
    --ubatch-size 512 \
    --port 8080 \
    $FLASH_ARG \
    --jinja \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    "$@"
