# Galaxy10 Deep Learning Analysis Pipeline
<img width="1000" height="662" alt="image" src="https://github.com/user-attachments/assets/fb2c64e1-3f2d-4942-b578-c7df9201d1f1" />


[![FiftyOne](https://img.shields.io/badge/FiftyOne-1.9.0+-orange.svg)](https://voxel51.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/Environment-conda-brightgreen.svg)](https://docs.conda.io/)
[![Torch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

Production-ready machine learning pipeline for automated galaxy morphology classification using the Galaxy10 DECals dataset (17,736 images across 10 morphological classes). Optimized for consumer hardware with full CPU parallelization and 4GB GPU support.

## 🌟 Key Features

- **4 Pre-trained Models**: ViT-S/16, EfficientNetV2-S, Galaxy Zoo CNN, DeepGalaxnet, CLIP
- **Full CPU Parallelization**: (n-1) cores across all stages for maximum throughput
- **4GB VRAM Optimized**: RTX 500 Ada compatible with automatic OOM recovery
- **Interactive Exploration**: FiftyOne app with Voxel GPT and CLIP text search
- **Model Interpretability**: GradCAM attention maps and confusion analysis
- **Anomaly Detection**: LOF and Isolation Forest for quality assurance

## 🏗️ Pipeline Architecture

```
Data (H5) → Embeddings (4 models) → Training → Visualization → Anomaly Detection
                                                      ↓
                                              FiftyOne App Integration
                                                       ↓
         Interactive Visulization  +   Clip Search    +  Embedding    +   Model Evaluation   
                   ↓                        ↓                 ↓                  ↓
               Filtering       Short by Similarity        UMAP/ PCA              ↓
                                                           Summary | Class Performence | Confusion Matrics                                                     
```

## 📊 Model Architecture

| Model | Output Dim | Specialization |
|-------|------------|----------------|
| ViT-S/16 | 384 | Global attention patterns |
| EfficientNetV2-S | 1,280 | Balanced accuracy/speed |
| Galaxy Zoo CNN | 2,048 | Astronomical features |
| DeepGalaxnet | 1,024 | Morphology-specific |
| CLIP | 512 | Text-based search |

## 🚀 Quick Start

### 1. Environment Setup

**Option A: Automated Install (Easiest)**
```bash
./install.sh
```

**Option B: Manual Install (Miniconda + Pip)**
```bash
# Create conda environment
conda create -n galaxy10 python=3.11 -y
conda activate galaxy10

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

**Option C: Pure Conda** (slower): See `INSTALL.md`

### 2. Download Dataset

Download the Galaxy10 DECals dataset (Galaxy10_DECals.h5) and place it in the project root:
- Kaggle: https://www.kaggle.com/datasets/jaimetrickz/galaxy10-decals

### 3. Verify Setup

Run the setup verification notebook:
```bash
jupyter lab notebooks/00_setup_verification.ipynb
```

### 4. Run Pipeline

Execute notebooks in order:
1. `01_data_exploration.ipynb` - Load and validate dataset
2. `02_model_embeddings.ipynb` - Extract embeddings from 4 models
3. `03_ensemble_training.ipynb` - Train and select best classifier
4. `04_visualization.ipynb` - FiftyOne app 
5. `05_anomaly_detection.ipynb` - Detect outliers and artifacts n `not updated X`



## 📁 Project Structure

```
galaxy10/
├── notebooks/
│   ├── 00_setup_verification.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_embeddings.ipynb
│   ├── 03_ensemble_training.ipynb
│   ├── 04_visualization.ipynb 
│   ├── 05_anomaly_detection.ipynb    X
│
x  too large so not updated  
├── src/
│   ├── config.py                    # Global configuration
│   └── models/
│       ├── extractor.py             # EmbeddingExtractor class
│       └── ensemble.py              # EnsembleTrainer class
├── artifacts/
│   ├── embeddings/                  # Cached .npy files
│   ├── models/                      # Trained classifiers
│   └── visualizations/              # Plots
├── environment.yml                  # Conda environment
└── Galaxy10_DECals.h5              # Dataset (download separately)
```

## ⚙️ CPU Optimization

The pipeline uses (n-1) CPU cores for parallel processing:

```python
n_jobs = cpu_count() - 1  # Reserve 1 core for system
```

Applied to:
- `DataLoader(num_workers=n_jobs)` - Data loading
- `RandomForest(n_jobs=n_jobs)` - Training
- `UMAP(n_jobs=n_jobs)` - Dimensionality reduction
- `LocalOutlierFactor(n_jobs=n_jobs)` - Anomaly detection

**Performance**: 90-95% CPU utilization during compute-heavy stages

## 🎯 Capabilities

### 1. Multi-Model Embeddings
- Parallel extraction from 4 architectures
- Cached to disk (no re-computation)
- GPU batch processing with OOM recovery

### 2. Ensemble Learning
-- Logistic Regression  
- Random Forest  
- Extra Trees  
- XGBoost  
- K-Nearest Neighbors  
- Automatic best-model selection
- 5-fold stratified cross-validation

### 3. Interactive Visualization
- UMAP//PCA projections
- FiftyOne app: similarity search, filtering classification
- CLIP text search: "spiral galaxy with bright core"
- FiftyOne aoo: Model Evaluation

### 4. Anomaly Detection
- Local Outlier Factor (LOF)
- Isolation Forest
- Identifies: rare morphologies, mislabels, artifacts


## 💾 System Requirements

- **CPU**: Multi-core (16+ cores recommended)
- **GPU**: 4GB VRAM (RTX 500 Ada tested)
- **RAM**: 32GB recommended
- **Disk**: 100GB for dataset + artifacts
- **OS**: Ubuntu 22.04 LTS (kernel 6.8) or similar

## 📚 Dataset Information

**Galaxy10 DECals** (Dieleman et al., 2019)
- Total Samples: 17,736 images
- Image Size: 256×256×3 (RGB)
- Source: DESI Legacy Survey
- Label Quality: Citizen science (Galaxy Zoo)

### Morphological Classes

| ID | Label | Count | Description |
|----|-------|-------|-------------|
| 0 | Disturbed Galaxies | 1,081 | Mergers, interactions |
| 1 | Merging Galaxies | 1,853 | Active mergers |
| 2 | Round Smooth | 2,645 | Elliptical galaxies |
| 3 | In-between Round Smooth | 2,027 | Intermediate morphology |
| 4 | Cigar-Shaped Smooth | 334 | Edge-on ellipticals |
| 5 | Barred Spiral | 2,043 | Spiral + bar feature |
| 6 | Unbarred Tight Spiral | 1,829 | Tightly wound spirals |
| 7 | Unbarred Loose Spiral | 1,423 | Loosely wound spirals |
| 8 | Edge-on with Bulge | 1,873 | Edge-on + central bulge |
| 9 | Edge-on without Bulge | 2,628 | Pure edge-on disks |

## 🔧 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch_size in notebook cells
batch_size = 8 # or 4/2

# Clear GPU cache
torch.cuda.empty_cache()
```

### FiftyOne App Not Opening
```bash
# Check FiftyOne installation
fiftyone --version

# Reinstall if needed
conda install -c conda-forge fiftyone
```

### Slow Performance
- Ensure you're using (n-1) CPU cores
- Check GPU is being utilized: `nvidia-smi`
- Verify embeddings are cached (check artifacts/embeddings/)

## 📖 Documentation

- `notebooks/README.md` - Detailed notebook guide
- `COMPLETE_PROJECT_SUMMARY.md` - Full pipeline documentation
- Inline docstrings in all Python modules

## 📄 License

MIT License - See LICENSE file for details

##  Acknowledgments

- Galaxy10 DECals dataset: Dieleman et al. (2019)
- FiftyOne: Voxel51
- Pre-trained models: TIMM, OpenAI CLIP
- Galaxy Zoo: Citizen science contributors

---

**Ready to explore 17,736 galaxies! 🌌**
