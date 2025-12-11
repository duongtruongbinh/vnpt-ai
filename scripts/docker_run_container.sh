#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="vnpt-ai:latest"
TEST_DATA_DIR="$PROJECT_ROOT/test_data"
OUTPUT_DIR="$PROJECT_ROOT/test_output"
LLM_MODEL_SMALL="Qwen/Qwen3-0.6B"
LLM_MODEL_LARGE="Qwen/Qwen3-0.6B"
EMBEDDING_MODEL="bkai-foundation-models/vietnamese-bi-encoder"
USE_VNPT_API=False

mkdir -p "$TEST_DATA_DIR" "$OUTPUT_DIR"

docker run \
    -v "$TEST_DATA_DIR:/data:ro" \
    -v "$OUTPUT_DIR:/output" \
    -e USE_VNPT_API="$USE_VNPT_API" \
    -e LLM_MODEL_SMALL="$LLM_MODEL_SMALL" \
    -e LLM_MODEL_LARGE="$LLM_MODEL_LARGE" \
    -e EMBEDDING_MODEL="$EMBEDDING_MODEL" \
    "$IMAGE_NAME"