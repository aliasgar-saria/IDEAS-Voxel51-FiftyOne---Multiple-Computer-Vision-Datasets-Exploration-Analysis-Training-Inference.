#  Image Deduplication with FiftyOne

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Libraries](https://img.shields.io/badge/Libraries-FiftyOne%2C%20Pillow%2C%20ImageHash%2C%20Seaborn-green.svg)](#)

> A Jupyter notebook demonstrating automated **data deduplication** using image similarity metrics, perceptual hashing, and visual analytics.  
> This project combines powerful libraries like **FiftyOne**, **ImageHash**, and **Matplotlib** to detect and visualize duplicate entries within large datasets.

---

## 🎯 Project Overview

Duplicate records are a major issue in modern datasets, especially those involving multimedia or large-scale image repositories.  
This notebook explores multiple methods for identifying and removing duplicates using **visual hashing, correlation matrices, and similarity metrics**.


 ## Features

-  **Perceptual Hashing**: Identifies exact and near-duplicate images using imagehash algorithms
-  **Deep Learning Embeddings**: Uses MobileNet-v2 for semantic similarity analysis
-  **Interactive Visualization**: Explore and analyze results with FiftyOne's web interface
-  **Dataset Management**: Load, clean, and organize image collections efficiently
-  **Batch Processing**: Handles datasets of various sizes with optimized processing
-  **Customizable Thresholds**: Adjustable sensitivity for different use cases

---

##  Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Core** | Python 3.13, NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Image Processing** | Pillow (PIL), ImageHash |
| **Data Management** | FiftyOne |
| **System Utilities** | OS |

---
## Results

### Dataset Processing
- **Total Images Processed**: 51 images successfully loaded and analyzed
- **Dataset Format**: ImageDirectory format with proper metadata integration
- **Embedding Computation**: All 51 images processed with MobileNet-v2 embeddings

### Duplicate Detection Results
The system identified **8 potential duplicate pairs** .

### Performance Metrics
- **Embedding Computation**: 55.4 seconds for 51 images (0.9 samples/second)
- **Hash Computation**: 62.0 milliseconds for 51 images (822.9 samples/second)
- **Memory Usage**: Efficient processing with no memory issues reported
-  The system demonstrated robust performance in identifying both exact duplicates (distance=0) and near-duplicates with varying similarity levels, making it suitable for dataset cleaning and quality control workflows.



