#!/usr/bin/env bash
# Persistent multi-GPU zero-shot SAM3 replay of saved plans.
set -Eeuo pipefail

RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SUBSET="${SUBSET:-${RELEASE_ROOT}/config/test.tsv}"
PLANS="${PLAN_ROOT:-${RELEASE_ROOT}/plans/test}"
OUTPUT="${DENSE_OUTPUT_ROOT:-${RELEASE_ROOT}/outputs/dense}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RETRY_SECONDS="${RETRY_SECONDS:-15}"

: "${VQA_PROJECT_ROOT:?Set VQA_PROJECT_ROOT to the directory containing dataset/}"
: "${SAM3_ROOT:?Set SAM3_ROOT to the official SAM3 source checkout}"
: "${SAM3_CHECKPOINT:?Set SAM3_CHECKPOINT to the official sam3.pt checkpoint}"

IFS=',' read -r -a GPUS <<<"${GPU_LIST}"
NUM_GPUS="${#GPUS[@]}"
NUM_SHARDS=$((NUM_GPUS * WORKERS_PER_GPU))
mkdir -p "${OUTPUT}"

export VQA_DATA_SPLIT="${VQA_DATA_SPLIT:-test}"
export VQA_VALID_ROOT="${VQA_VALID_ROOT:-${VQA_PROJECT_ROOT}/dataset/${VQA_DATA_SPLIT}}"
export VQA_ANNOTATION_PATH="${VQA_ANNOTATION_PATH:-${VQA_VALID_ROOT}/annotations/grounded_question_${VQA_DATA_SPLIT}.json}"
export VQA_PROJECT_ROOT SAM3_ROOT SAM3_CHECKPOINT

pids=()
for ((rank = 0; rank < NUM_SHARDS; rank++)); do
  gpu_slot=$((rank % NUM_GPUS))
  worker_slot=$((rank / NUM_GPUS))
  CUDA_VISIBLE_DEVICES="${GPUS[gpu_slot]}" "${PYTHON_BIN}" \
    "${RELEASE_ROOT}/src/pipeline/persistent_test_worker.py" \
    --rank "${rank}" --num-shards "${NUM_SHARDS}" \
    --subset "${SUBSET}" --plan-root "${PLANS}" --output-root "${OUTPUT}" \
    --max-attempts "${MAX_ATTEMPTS}" --retry-seconds "${RETRY_SECONDS}" \
    >"${OUTPUT}/run.gpu${gpu_slot}.worker${worker_slot}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [[ "${status}" -ne 0 ]]; then
  "${PYTHON_BIN}" "${RELEASE_ROOT}/src/pipeline/merge_compact_predictions.py" \
    --output-root "${OUTPUT}" --subset "${SUBSET}" \
    --output "${OUTPUT}/predictions.partial.json" --allow-missing || true
  exit 1
fi
"${PYTHON_BIN}" "${RELEASE_ROOT}/src/pipeline/merge_compact_predictions.py" \
  --output-root "${OUTPUT}" --subset "${SUBSET}" \
  --output "${OUTPUT}/predictions.json"
