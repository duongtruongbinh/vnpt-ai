#!/bin/bash
set -e

# Ensure directories exist
mkdir -p /data /output

# Log environment info
echo "[Entrypoint] DATA_INPUT_DIR: $DATA_INPUT_DIR (test input files)"
echo "[Entrypoint] DATA_OUTPUT_DIR: $DATA_OUTPUT_DIR (output files)"
echo "[Entrypoint] DATA_DIR: ${DATA_DIR:-/code/data} (knowledge base data)"
echo "[Entrypoint] Python path: $(which python)"
echo "[Entrypoint] Starting application..."

# Execute the command
exec "$@"