#!/usr/bin/env bash
set -Eeuo pipefail

RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
gzip -cd "${RELEASE_ROOT}/artifacts/final_predictions.json.gz" \
  > "${1:-${RELEASE_ROOT}/artifacts/final_predictions.json}"
