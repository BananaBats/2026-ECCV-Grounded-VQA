#!/usr/bin/env bash
# Shardable Gemini 30-frame sparse amodal-box launcher.
set -Eeuo pipefail

RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
OUTPUT_ROOT="${SPARSE_OUTPUT_ROOT:-${RELEASE_ROOT}/outputs/sparse}"

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT for Vertex AI}"
export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

exec "${PYTHON_BIN}" "${RELEASE_ROOT}/src/sparse_fusion/generate_sparse.py" \
  --subset "${SUBSET:-${RELEASE_ROOT}/config/test.tsv}" \
  --plan-root "${PLAN_ROOT:-${RELEASE_ROOT}/plans/test}" \
  --video-root "${VIDEO_ROOT:-${VQA_PROJECT_ROOT:-${RELEASE_ROOT}}/dataset/test/videos}" \
  --output-root "${OUTPUT_ROOT}" \
  --num-shards "${NUM_SHARDS}" --shard-index "${SHARD_INDEX}" "$@"
