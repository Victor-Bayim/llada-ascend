#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${MODEL_DIR:?Set MODEL_DIR to the local LLaDA-MoE-7B-A1B-Instruct directory}"
PYTHON="${PYTHON:-python}"
RESULTS_DIR="${RESULTS_DIR:-results/perf}"
mkdir -p "$RESULTS_DIR"
# Same shape/flag schedule as the report, but record multiple runs by default.
# Set ITERS=1 for the historical count; this does not reproduce historical timings.
for batch in 1 16; do
  for length in 32 64 128 256 512 1024 2048; do
    flags=()
    if [ "$batch" -eq 1 ]; then
      flags+=(--vectorized_topk --use_where_update)
    elif [ "$length" -ge 2048 ]; then
      flags+=(--logits_chunk 64 --masked_lm_head)
    elif [ "$length" -ge 512 ]; then
      flags+=(--logits_chunk 256)
    fi
    "$PYTHON" -m analysis.llada_torch_npu.bench_diffusion \
      --model "$MODEL_DIR" --device npu:0 --batch_size "$batch" \
      --prompt_len "$length" --gen_length "$length" --block_length "$((length / 2))" \
      --steps 2 --temperature 0 --iters "${ITERS:-5}" --warmup 1 \
      --threshold 0.95 --mm_bmm_nd --inference_mode "${flags[@]}" \
      --json_out "$RESULTS_DIR/len${length}_b${batch}.json" \
      | tee "$RESULTS_DIR/len${length}_b${batch}.log"
  done
done
