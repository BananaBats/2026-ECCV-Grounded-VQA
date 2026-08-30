#!/usr/bin/env bash
# Apply the fixed validation-derived router. The alternate branch is external.
set -Eeuo pipefail

RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ALTERNATE_FUSED_PREDICTIONS:?Set this to the excluded alternate-branch predictions or shard directory}"

"${PYTHON_BIN:-python3}" "${RELEASE_ROOT}/src/routing/route_predictions.py" \
  --base "${BASE_FUSED_PREDICTIONS:-${RELEASE_ROOT}/outputs/fused/predictions.json}" \
  --alternate "${ALTERNATE_FUSED_PREDICTIONS}" \
  --plans "${PLAN_ROOT:-${RELEASE_ROOT}/plans/test}" \
  --subset "${SUBSET:-${RELEASE_ROOT}/config/test.tsv}" \
  --output "${FINAL_OUTPUT:-${RELEASE_ROOT}/outputs/final/predictions.json}" \
  --manifest "${ROUTE_MANIFEST:-${RELEASE_ROOT}/outputs/final/route_manifest.json}"
