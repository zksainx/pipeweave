#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PIPEWEAVE_IR_PYTHON:-/home/chen/anaconda3/envs/drkernel-vllm027-cu132/bin/python}
RUN_DIR=${1:-"$PROJECT_ROOT/runs/rtx5090-smoke"}

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m pipeweave_ir env \
  --require-cuda-major 13 \
  --cuda-bin /usr/local/cuda-13.2/bin \
  --out "$RUN_DIR/environment.json"

"$PYTHON_BIN" -m pipeweave_ir capture \
  --suite "$PROJECT_ROOT/configs/rtx5090_smoke.json" \
  --out "$RUN_DIR" \
  --require-cuda-major 13 \
  --cuda-bin /usr/local/cuda-13.2/bin

"$PYTHON_BIN" -m pipeweave_ir features \
  --artifacts "$RUN_DIR/artifacts" \
  --out "$RUN_DIR/features.jsonl"

"$PYTHON_BIN" -m pipeweave_ir dataset --run "$RUN_DIR" --out "$RUN_DIR/dataset.jsonl"
