#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${MODEL_DIR:?Set MODEL_DIR to the local model directory}"
PYTHON="${PYTHON:-python}"
RESULTS_DIR="${RESULTS_DIR:-results/quality}"
mkdir -p "$RESULTS_DIR"
# Ten-question smoke test; separate from the two-step latency configuration.
"$PYTHON" -m analysis.llada_torch_npu.ceval_diffusion_subset \
  --model "$MODEL_DIR" --device npu:0 --per_subject 2 --groups 1 --group_start 1 \
  --steps 24 --gen_length 32 --block_length 8 --temperature 0.7 --seed 0 \
  --threshold 0.95 | tee "$RESULTS_DIR/ceval_subset.log"
