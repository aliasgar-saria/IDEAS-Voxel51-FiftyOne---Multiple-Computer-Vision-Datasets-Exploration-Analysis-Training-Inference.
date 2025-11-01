#!/bin/bash
# Galaxy10 Pipeline - Quick Installation Script (Miniconda)

set -e  # Exit on error

echo "=========================================="
echo "Galaxy10 Pipeline Installation"
echo "=========================================="
echo ""

# Check conda
if ! command -v conda &> /dev/null; then
    echo " Conda not found. Please install Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found"

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    USE_CUDA=true
else
    echo "⚠️  No NVIDIA GPU detected - will install CPU-only version"
    USE_CUDA=false
fi

echo ""
echo "Creating conda environment..."
conda create -n galaxy10 python=3.11 -y

echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate galaxy10

echo ""
echo "Installing PyTorch..."
if [ "$USE_CUDA" = true ]; then
    echo "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate galaxy10"
echo ""
echo "To verify installation:"
echo "  jupyter lab notebooks/00_setup_verification.ipynb"
echo ""
echo "To start the pipeline:"
echo "  jupyter lab notebooks/COMPLETE_MASTER.ipynb"
echo ""
echo "Don't forget to download Galaxy10_DECals.h5 dataset!"
echo "=========================================="
