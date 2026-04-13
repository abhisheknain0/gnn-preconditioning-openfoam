#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${WM_PROJECT_USER_DIR:-}" ]]; then
  echo "WM_PROJECT_USER_DIR is not set. Load OpenFOAM modules first." >&2
  exit 1
fi

SRC_LIB="${STACK_ROOT}/libIterReduceCoupling"
SRC_BASE="${STACK_ROOT}/custom/incompressibleFluidIterReduce"
SRC_MOD="${STACK_ROOT}/custom/iterReduceIncompressibleFluid"
SRC_APP="${STACK_ROOT}/custom/iterReduceIncompressibleFluidApp"

DST_LIB="${WM_PROJECT_USER_DIR}/modules/iterReduceCoupling"
DST_BASE="${WM_PROJECT_USER_DIR}/modules/incompressibleFluidIterReduce"
DST_MOD="${WM_PROJECT_USER_DIR}/modules/iterReduceIncompressibleFluid"
DST_APP="${WM_PROJECT_USER_DIR}/applications/solvers/iterReduceIncompressibleFluid"

mkdir -p "${WM_PROJECT_USER_DIR}/modules" "${WM_PROJECT_USER_DIR}/applications/solvers"
mkdir -p "${DST_LIB}" "${DST_BASE}" "${DST_MOD}" "${DST_APP}"

rsync -a --delete "${SRC_LIB}/" "${DST_LIB}/"
rsync -a --delete "${SRC_BASE}/" "${DST_BASE}/"
rsync -a --delete "${SRC_MOD}/" "${DST_MOD}/"
rsync -a --delete "${SRC_APP}/" "${DST_APP}/"

echo "Synced sources to:"
echo "  ${DST_LIB}"
echo "  ${DST_BASE}"
echo "  ${DST_MOD}"
echo "  ${DST_APP}"
