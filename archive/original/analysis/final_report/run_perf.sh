#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_CMD="${DOCKER_CMD:-docker}"
IMAGE="${IMAGE:-anoosxj/huaweiict-llada-xllm:0.7.1-slim}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/datasets}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$ROOT_DIR/analysis}"
THRESHOLD="${THRESHOLD:-0.95}"
PREFIX_LOOK="${PREFIX_LOOK:-0}"
AFTER_LOOK="${AFTER_LOOK:-0}"

mkdir -p "$LOG_DIR"

if ! $DOCKER_CMD image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[ERROR] Docker image not found: $IMAGE" >&2
  echo "[ERROR] Please run: docker pull $IMAGE" >&2
  exit 1
fi

if [ ! -d "$MODEL_DIR/LLaDA-MoE-7B-A1B-Instruct" ]; then
  echo "[ERROR] Model path not found: $MODEL_DIR/LLaDA-MoE-7B-A1B-Instruct" >&2
  exit 1
fi

DEV_ARGS=()
for dev in /dev/davinci* /dev/davinci_manager /dev/hisi_hdc; do
  if [ -e "$dev" ]; then
    DEV_ARGS+=("--device" "$dev")
  fi
done

COMMON_RUN=(
  $DOCKER_CMD run --rm --privileged --security-opt label=disable
  "${DEV_ARGS[@]}"
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
  -v "$MODEL_DIR:/workspace/models"
  -v "$DATASET_DIR:/workspace/datasets"
  -v "$LOG_DIR:/workspace/logs"
  -v "$ANALYSIS_DIR:/workspace/analysis"
  "$IMAGE"
)

BASE_ENV=$'source /usr/local/Ascend/ascend-toolkit/set_env.sh\n'
BASE_ENV+=$'export NPU_MEMORY_FRACTION=0.8\n'
BASE_ENV+=$'export PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:64\n'
BASE_ENV+=$'export ASCEND_VISIBLE_DEVICES=0\n'
BASE_ENV+=$'export PYTHONPATH=/workspace:/workspace/xllm-src/build/lib.linux-aarch64-cpython-311:$PYTHONPATH\n'

LENGTHS=(32 64 128 256 512 1024 2048)
CSV_OUT="$LOG_DIR/perf_latency.csv"

run_case() {
  local batch="$1"
  local length="$2"
  local block_length="$3"
  local steps="$4"
  local logits_chunk="$5"
  local extra_flags="$6"
  local log_file="$LOG_DIR/perf_len${length}_b${batch}.log"
  local cmd="$BASE_ENV python /workspace/analysis/llada_torch_npu/bench_diffusion.py \
  --model /workspace/models/LLaDA-MoE-7B-A1B-Instruct \
  --device npu:0 --batch_size ${batch} --prompt_len ${length} --gen_length ${length} \
  --block_length ${block_length} --steps ${steps} --temperature 0.0 --iters 1 --warmup 1 \
  --threshold ${THRESHOLD} --prefix_look ${PREFIX_LOOK} --after_look ${AFTER_LOOK} \
  --mm_bmm_nd --inference_mode ${extra_flags}"
  if [ "${logits_chunk}" -gt 0 ]; then
    cmd="${cmd} --logits_chunk ${logits_chunk}"
  fi
  "${COMMON_RUN[@]}" bash -lc "$cmd" | tee "$log_file"
}

for length in "${LENGTHS[@]}"; do
  block_length=$((length / 2))
  steps=2
  extra_flags="--vectorized_topk --use_where_update"
  run_case 1 "$length" "$block_length" "$steps" 0 "$extra_flags"
done

for length in "${LENGTHS[@]}"; do
  block_length=$((length / 2))
  steps=2
  logits_chunk=0
  extra_flags=""
  if [ "$length" -ge 2048 ]; then
    logits_chunk=64
    extra_flags="--masked_lm_head"
  elif [ "$length" -ge 512 ]; then
    logits_chunk=256
  fi
  run_case 16 "$length" "$block_length" "$steps" "$logits_chunk" "$extra_flags"
done

echo "batch,prompt_len,gen_length,block_length,steps,threshold,prefix_look,after_look,latency_avg_s" > "$CSV_OUT"
for batch in 1 16; do
  for length in "${LENGTHS[@]}"; do
    log_file="$LOG_DIR/perf_len${length}_b${batch}.log"
    latency="$(sed -n 's/.*latency_avg_s=\\([0-9.]*\\).*/\\1/p' "$log_file" | tail -n1)"
    block_length=$((length / 2))
    echo "${batch},${length},${length},${block_length},2,${THRESHOLD},${PREFIX_LOOK},${AFTER_LOOK},${latency}" >> "$CSV_OUT"
  done
done
