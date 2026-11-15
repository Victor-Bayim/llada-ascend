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

if [ ! -d "$DATASET_DIR/ceval" ]; then
  echo "[ERROR] CEval dataset cache not found: $DATASET_DIR/ceval" >&2
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
BASE_ENV+=$'export HF_DATASETS_OFFLINE=1\n'
BASE_ENV+=$'export TRANSFORMERS_OFFLINE=1\n'
BASE_ENV+=$'export HF_HUB_OFFLINE=1\n'

"${COMMON_RUN[@]}" bash -lc "$BASE_ENV python /workspace/analysis/llada_torch_npu/ceval_diffusion_subset.py \
  --model /workspace/models/LLaDA-MoE-7B-A1B-Instruct \
  --device npu:0 --per_subject 2 --groups 1 --group_start 1 \
  --steps 24 --gen_length 32 --block_length 8 --temperature 0.7 \
  --threshold ${THRESHOLD} --prefix_look ${PREFIX_LOOK} --after_look ${AFTER_LOOK}" \
  | tee "$LOG_DIR/ceval_subset.log"
