#!/usr/bin/env bash
set -Eeuo pipefail

RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${RELEASE_ROOT}/plans/test" "${RELEASE_ROOT}/sparse_predictions/test"
tar -xzf "${RELEASE_ROOT}/inputs/plans_test.tar.gz" \
  -C "${RELEASE_ROOT}/plans/test"
tar -xzf "${RELEASE_ROOT}/inputs/sparse_predictions_test.tar.gz" \
  -C "${RELEASE_ROOT}/sparse_predictions/test"
