#!/bin/bash
set -e

echo "=== VNPT AI SUBMISSION START ==="
echo "Input Dir: $DATA_INPUT_DIR"
echo "Output Dir: $DATA_OUTPUT_DIR"

if [ -z "$VNPT_LARGE_TOKEN_ID" ]; then
    echo "[WARNING] No External API Key provided. Using default/hardcoded keys inside config."
else
    echo "[INFO] External API Keys detected via Environment Variables."
fi

# Chạy file entry-point chính
echo "Starting prediction pipeline..."
python predict.py