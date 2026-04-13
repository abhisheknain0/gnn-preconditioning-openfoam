#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/sync_to_user_dir.sh"

if [[ -z "${WM_PROJECT_USER_DIR:-}" ]]; then
  echo "WM_PROJECT_USER_DIR is not set. Load OpenFOAM modules first." >&2
  exit 1
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for cand in python3.12 python3 python; do
    if command -v "${cand}" >/dev/null 2>&1; then
      if "${cand}" -c 'import torch' >/dev/null 2>&1; then
        PYTHON_BIN="${cand}"
        break
      fi
    fi
  done
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "Could not find a Python interpreter with torch installed." >&2
  exit 1
fi

if [[ -z "${TORCH_DIR:-}" ]]; then
  for cand in \
    "${HOME}/.local/libtorch-2.5.1-cu121-cxx11/libtorch" \
    "${HOME}/.local"/libtorch*-cxx11/libtorch
  do
    if [[ -d "${cand}" && -d "${cand}/include" && -d "${cand}/lib" ]]; then
      TORCH_DIR="${cand}"
      export TORCH_DIR
      break
    fi
  done
fi

if [[ -z "${TORCH_DIR:-}" ]]; then
  TORCH_DIR="$("${PYTHON_BIN}" -c 'import os, torch; print(os.path.dirname(torch.__file__))')"
  export TORCH_DIR
fi

if [[ -z "${TORCH_ABI:-}" ]]; then
  if [[ "${TORCH_DIR}" == *libtorch*-cxx11/libtorch ]]; then
    TORCH_ABI=1
  else
    TORCH_ABI="$("${PYTHON_BIN}" -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')"
  fi
  export TORCH_ABI
fi

if [[ "${TORCH_ABI}" != "1" ]]; then
  echo "Refusing to build with TORCH_ABI=${TORCH_ABI}. OpenFOAM build must use ABI=1 cxx11 libtorch." >&2
  exit 1
fi

echo "Using PYTHON_BIN=${PYTHON_BIN}"
echo "Using TORCH_DIR=${TORCH_DIR}"
echo "Using TORCH_ABI=${TORCH_ABI}"

LIB_DIR="${WM_PROJECT_USER_DIR}/modules/iterReduceCoupling"
BASE_DIR="${WM_PROJECT_USER_DIR}/modules/incompressibleFluidIterReduce"
MOD_DIR="${WM_PROJECT_USER_DIR}/modules/iterReduceIncompressibleFluid"
APP_DIR="${WM_PROJECT_USER_DIR}/applications/solvers/iterReduceIncompressibleFluid"

echo "Building libIterReduceCoupling..."
wmake libso "${LIB_DIR}"

echo "Building libincompressibleFluid_iterreduce..."
wmake libso "${BASE_DIR}"

echo "Building libiterReduceIncompressibleFluid..."
wmake libso "${MOD_DIR}"

echo "Building iterReduceIncompressibleFluid app..."
wmake "${APP_DIR}"

echo "Build complete."
echo "User app binary should be in: ${FOAM_USER_APPBIN}"
