#!/bin/bash
# GNN Simulator Environment Setup
# Portable setup script for HPC cluster

RUNTIME_ONLY=false
if [[ "${1:-}" == "--runtime-only" ]]; then
    RUNTIME_ONLY=true
fi

# Detect whether this script is being sourced into the current shell.
_SETUP_ENV_SOURCED=0
(return 0 2>/dev/null) && _SETUP_ENV_SOURCED=1

echo "=========================================="
echo "GNN Simulator Environment Setup"
echo "=========================================="
if [[ "${RUNTIME_ONLY}" == "true" ]]; then
    echo "Mode: runtime-only (no pip installs)"
fi

# Ensure environment-modules is available in this shell.
if ! command -v module >/dev/null 2>&1; then
    if [[ -f /usr/share/Modules/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/Modules/init/bash
    elif [[ -f /opt/lmod/lmod/init/bash ]]; then
        # shellcheck disable=SC1091
        source /opt/lmod/lmod/init/bash
    elif [[ -f /opt/lmod/lmod/init/profile ]]; then
        # shellcheck disable=SC1091
        source /opt/lmod/lmod/init/profile
    fi
fi
if ! command -v module >/dev/null 2>&1; then
    echo "Error: 'module' command not found. Load environment-modules first." >&2
    if [[ "${_SETUP_ENV_SOURCED}" -eq 1 ]]; then
        return 1
    fi
    exit 1
fi

# Load HPC modules
# Note: Loading order matters due to dependencies
module purge

# Base compiler (required by many modules)
module load GCC/13.3.0

# MPI (required by gmsh and OpenFOAM)
module load OpenMPI/5.0.3

# Gmsh (requires GCC/13.3.0 and OpenMPI/5.0.3)
module load gmsh/4.13.0

# Python and scientific computing
module load Python/3.12.3
module load CUDA/12.6.3
module load SciPy-bundle/2024.05        # Provides: numpy, scipy, matplotlib
module load Python-bundle-PyPI/2024.06  # Provides: pip, setuptools

# HDF5 and related
module load HDF5/1.14.5
module load h5py/3.12.1

# OpenFOAM (may depend on GCC and MPI)
module load OpenFOAM/12

# Build tools
module load CMake/3.29.3
module load Boost/1.85.0

# Set up Python paths
export PATH="$PATH:$HOME/.local/bin"
USER_SITE=$(python3 -m site --user-site)
export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:$PYTHONPATH}"

# Keep Matplotlib cache/config on node-local scratch when possible.
if [[ -z "${MPLCONFIGDIR:-}" ]]; then
    export MPLCONFIGDIR="${TMPDIR:-/tmp}/${USER:-user}/matplotlib"
    mkdir -p "${MPLCONFIGDIR}" 2>/dev/null || true
fi

if [[ "${RUNTIME_ONLY}" == "true" ]]; then
    echo ""
    echo "Environment loaded."
    echo "WM_PROJECT_DIR=${WM_PROJECT_DIR:-UNSET}"
    echo "WM_PROJECT_USER_DIR=${WM_PROJECT_USER_DIR:-UNSET}"
    echo "python3=$(command -v python3)"
    echo "foamRun=$(command -v foamRun || true)"
    echo "=========================================="
    if [[ "${_SETUP_ENV_SOURCED}" -eq 1 ]]; then
        return 0
    fi
    exit 0
fi

echo "Installing Python runtime packages..."
# Keep this list intentionally small. The project uses a custom MeshGraphNet
# implementation plus PyG Data/DataLoader containers; it does not import the
# optional compiled PyG extension packages or stale experiment dependencies.
python3 -m pip install --user --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install --user --no-cache-dir torch-geometric==2.7.0
python3 -m pip install --user --no-cache-dir packaging pillow

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Available packages:"
python3 -c "
import torch, torch_geometric, h5py, matplotlib
print(f'  PyTorch: {torch.__version__}')
print(f'  PyTorch Geometric: {torch_geometric.__version__}')
print(f'  h5py: {h5py.__version__}')
print(f'  matplotlib: {matplotlib.__version__} (from SciPy-bundle)')
"

echo ""
echo "Verifying required tools:"
if command -v gmsh &> /dev/null; then
    GMSH_VERSION=$(gmsh --version 2>/dev/null | head -1 || echo "unknown")
    echo "  ✓ gmsh: $GMSH_VERSION"
else
    echo "  ✗ gmsh: NOT FOUND (required for mesh generation)"
fi

if command -v gmshToFoam &> /dev/null; then
    echo "  ✓ gmshToFoam: Available (from OpenFOAM)"
else
    echo "  ✗ gmshToFoam: NOT FOUND (required for mesh conversion)"
fi

if command -v pimpleFoam &> /dev/null; then
    echo "  ✓ pimpleFoam: Available (from OpenFOAM)"
else
    echo "  ✗ pimpleFoam: NOT FOUND (required for CFD simulation)"
fi

echo ""
echo "Environment is ready for command-line training/inference jobs."
echo "=========================================="
