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
    echo "Mode: runtime-only (no pip installs, no kernel reinstall)"
fi

# Ensure environment-modules is available in this shell.
if ! command -v module >/dev/null 2>&1; then
    if [[ -f /usr/share/Modules/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/Modules/init/bash
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
export PATH="$HOME/.local/bin:$PATH"
USER_SITE=$(python3 -m site --user-site)
export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:$PYTHONPATH}"

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

echo "Installing PyTorch packages..."
python3 -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing PyTorch Geometric..."
python3 -m pip install --user torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install --user imageio jaxtyping onnx nvtx timm treelib termcolor GitPython hydra-core omegaconf einops ninja packaging

echo "Installing Jupyter kernel..."
python3 -m pip install --user ipykernel ipywidgets

# Create kernel wrapper dynamically (uses current script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_WRAPPER="${SCRIPT_DIR}/kernel_wrapper.sh"

echo "Creating kernel wrapper..."
cat > "$KERNEL_WRAPPER" << 'EOF'
#!/bin/bash
# Kernel wrapper - loads HPC modules before starting Python
# Note: Loading order matters due to dependencies
source /usr/share/Modules/init/bash
module purge

# Base compiler (required by many modules)
module load GCC/13.3.0

# MPI (required by gmsh and OpenFOAM)
module load OpenMPI/5.0.3

# Gmsh (requires GCC/13.3.0 and OpenMPI/5.0.3)
module load gmsh/4.13.0

# Python and scientific computing
module load Python/3.12.3 CUDA/12.6.3
module load SciPy-bundle/2024.05 Python-bundle-PyPI/2024.06

# HDF5 and related
module load HDF5/1.14.5 h5py/3.12.1

# OpenFOAM (may depend on GCC and MPI)
module load OpenFOAM/12

# Build tools
module load CMake/3.29.3 Boost/1.85.0

export PATH="$HOME/.local/bin:$PATH"
USER_SITE=$(python3 -m site --user-site)
export PYTHONPATH="$USER_SITE:$PYTHONPATH"
exec python3 "$@"
EOF
chmod +x "$KERNEL_WRAPPER"

# Remove old kernels
echo "Cleaning up old kernels..."
jupyter kernelspec uninstall -f gnn-preconditioner 2>/dev/null || true
jupyter kernelspec uninstall -f gnn_sim 2>/dev/null || true

# Install kernel
echo "Installing Jupyter kernel..."
python3 -m ipykernel install --user --name gnn-simulator --display-name "GNN-Simulator"

# Update kernel to use wrapper (using direct JSON write to avoid escaping issues)
KERNEL_JSON="$HOME/.local/share/jupyter/kernels/gnn-simulator/kernel.json"
cat > "$KERNEL_JSON" << JSONEOF
{
 "argv": [
  "${KERNEL_WRAPPER}",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "GNN-Simulator",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
JSONEOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Kernel: GNN-Simulator"
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
echo "Next: Open JupyterLab and select 'GNN-Simulator' kernel"
echo "=========================================="
