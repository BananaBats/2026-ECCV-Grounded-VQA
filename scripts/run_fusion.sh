#!/usr/bin/env bash
# Run v13-style fusion, then use zero-shot dense fallback for sparse failures.
set -Eeuo pipefail

RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SUBSET="${SUBSET:-${RELEASE_ROOT}/config/test.tsv}"
PLAN_ROOT="${PLAN_ROOT:-${RELEASE_ROOT}/plans/test}"
DENSE_ROOT="${DENSE_ROOT:-${RELEASE_ROOT}/outputs/dense}"
SPARSE_ROOT="${SPARSE_ROOT:-${RELEASE_ROOT}/outputs/sparse}"
FUSED_ROOT="${FUSED_ROOT:-${RELEASE_ROOT}/outputs/fused}"
NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "${FUSED_ROOT}"
pids=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
  "${PYTHON_BIN}" "${RELEASE_ROOT}/src/sparse_fusion/fuse_dense_sparse.py" \
    --subset "${SUBSET}" --plan-root "${PLAN_ROOT}" \
    --dense-root "${DENSE_ROOT}" --sparse-root "${SPARSE_ROOT}" \
    --output-root "${FUSED_ROOT}" --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard}" >"${FUSED_ROOT}/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}" || true
done

"${PYTHON_BIN}" "${RELEASE_ROOT}/src/sparse_fusion/finalize_fusion.py" \
  --subset "${SUBSET}" --fusion-root "${FUSED_ROOT}" \
  --dense-root "${DENSE_ROOT}" --output "${FUSED_ROOT}/predictions.json" \
  --summary "${FUSED_ROOT}/fusion_final_summary.json"
