#!/usr/bin/env bash
# Shardable Gemini 3.7 Flash plan-only launcher.
set -Eeuo pipefail

RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SUBSET="${SUBSET:-${RELEASE_ROOT}/config/test.tsv}"
OUTPUT_ROOT="${PLAN_OUTPUT_ROOT:-${RELEASE_ROOT}/outputs/plans}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.7-flash}"

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT for Vertex AI}"
: "${VQA_PROJECT_ROOT:?Set VQA_PROJECT_ROOT to the directory containing dataset/}"

index=0
while IFS=$'\t' read -r video_id question_id || [[ -n "${video_id:-}" ]]; do
  [[ -z "${video_id:-}" || -z "${question_id:-}" ]] && continue
  if (( index % NUM_SHARDS == SHARD_INDEX )); then
    "${PYTHON_BIN}" "${RELEASE_ROOT}/src/pipeline/multi_anchor_box_baseline.py" \
      --video-id "${video_id}" --question-id "${question_id}" --plan-only \
      --gemini-model "${GEMINI_MODEL}" --max-anchor-frames 3 \
      --sam-stride 10 --boundary-radius 15 --segment-gate-margin 15 \
      --google-cloud-project "${GOOGLE_CLOUD_PROJECT}" --output-root "${OUTPUT_ROOT}"
  fi
  index=$((index + 1))
done < "${SUBSET}"
