# Installation Guide - Galaxy10 Pipeline

## Quick Install (Miniconda + Pip - Recommended)

This method uses Miniconda for environment management and pip for fast package installation.

### Step 1: Create Conda Environment

```bash
# Create conda environment with Python 3.10
conda create -n galaxy10 python=3.10 -y

# Activate environment
conda activate galaxy10
```

### Step 2: Install PyTorch with CUDA 12.1

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (if you have older GPU)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install All Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
jupyter lab notebooks/00_setup_verification.ipynb
```

---

## Alternative: Conda Environment (Slower)

If you prefer conda, use the environment.yml file:

```bash
conda env create -f environment.yml
conda activate galaxy10
```

**Note**: Conda environment solving can take 10-30 minutes. Pure pip is much faster (2-5 minutes).

---

## System Requirements

### Minimum
- Python 3.10+
- 16GB RAM
- 50GB disk space
- Ubuntu 22.04 LTS or similar

### Recommended
- Python 3.11+
- 32GB RAM
- NVIDIA GPU with 4GB+ VRAM (RTX 500 Ada or better)
- CUDA 12.1 drivers installed
- 200GB disk space

---

## Verify CUDA Installation

Before installing PyTorch, verify CUDA is installed:

```bash
nvidia-smi
```

You should see your GPU and CUDA version. If not:

```bash
# Install NVIDIA drivers (Ubuntu)
sudo apt update
sudo apt install nvidia-driver-535  # or latest

# Reboot
sudo reboot
```

---

## Troubleshooting

### CUDA Not Available After Install

```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# Reinstall with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### FiftyOne Installation Issues

```bash
# Install system dependencies (Ubuntu)
sudo apt-get install ffmpeg libsm6 libxext6

# Reinstall FiftyOne
pip install --upgrade fiftyone fiftyone-brain
```

### CLIP Installation Issues

```bash
# Install from source
pip install git+https://github.com/openai/CLIP.git

# Or use alternative
pip install open_clip_torch
```

### Out of Memory During Install

```bash
# Install packages one at a time
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm scikit-learn
pip install fiftyone fiftyone-brain
# ... continue with remaining packages
```

---

## Quick Start After Installation

```bash
# Activate environment
conda activate galaxy10

# Download Galaxy10 dataset (place in project root)
# From: https://www.kaggle.com/datasets/jaimetrickz/galaxy10-decals

# Verify setup
jupyter lab notebooks/00_setup_verification.ipynb

# Run pipeline
jupyter lab notebooks/COMPLETE_MASTER.ipynb
```

---

## Uninstall

```bash
# Deactivate and remove conda environment
conda deactivate
conda env remove -n galaxy10
```

---

**Installation time**: 2-5 minutes with pip, 10-30 minutes with conda depending upon your internet speed

