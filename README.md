# Attention U-Net for Sentinel-2 Agricultural Land Cover Segmentation

## Author
**Hannah Fathi**  
MSc Artificial Intelligence  

## Research Timeline
Spring 2025  

---

## Overview

This repository presents an **end-to-end deep learning framework for semantic segmentation of agricultural land cover** using Sentinel-2 multispectral imagery.

The system is designed for **operational-scale geospatial analysis** and integrates:
- Multi-spectral remote sensing data (Sentinel-2 L2A)
- Spectral index engineering (NDVI, NDWI, MNDWI, EVI, AWEI_sh)
- Weakly-supervised auto-labeling (rule-based + clustering fusion)
- Attention-enhanced U-Net architecture
- Patch-based training pipeline
- GIS-ready inference and GeoTIFF export

The objective is to enable **accurate, scalable, and reproducible agricultural land cover mapping**.

---

## Contributions

This work contributes a reproducible and scalable framework for multispectral agricultural segmentation by integrating weak supervision, attention mechanisms, and spectral index engineering into a unified deep learning pipeline.

### 1. Weakly-Supervised Labeling Pipeline
A hybrid annotation framework is introduced that combines:
- Spectral index-based rule generation (NDVI, NDWI, MNDWI, AWEI_sh)
- Unsupervised clustering (KMeans in spectral feature space)
- Weighted fusion strategy with noise suppression

This enables scalable dataset generation without manual labeling.

---

### 2. Attention-Enhanced Multispectral U-Net
A custom U-Net architecture is proposed with:
- Attention-gated skip connections
- Squeeze-and-Excitation (SE) channel recalibration
- Weight Standardized convolutions
- Group Normalization for small-batch stability

This improves boundary delineation in heterogeneous agricultural landscapes.

---

### 3. Spectral Index Fusion Strategy
A novel input representation strategy is used:
- Multi-band Sentinel-2 data (6 bands)
- Five vegetation/water indices
- Controlled redundancy injection for index emphasis

This enhances separability between spectrally similar classes (soil vs vegetation).

---

### 4. End-to-End GIS-Ready Pipeline
The system supports:
- Sliding window inference
- Test-Time Augmentation (TTA)
- DenseCRF refinement
- GeoTIFF export for GIS integration

---

### 5. Reproducible Patch-Based Training Framework
A fully deterministic pipeline with:
- Stratified patch sampling
- Water-aware balancing strategy
- Metadata tracking per patch

 ---
 
## Study Area

- **Region:** Veys–Sheyban agricultural zone, Khuzestan, Iran  
- **Satellite:** Sentinel-2 Level-2A (Google Earth Engine)  
- **Projection:** UTM Zone 39N (EPSG:32639)  
- **Resolution:** 10 meters  
- **Time Window:** Feb 20 – Apr 19, 2025 (spring season)

---

## Dataset

### Input Composition (16 Channels)

#### Spectral Bands
- B2 (Blue)
- B3 (Green)
- B4 (Red)
- B8 (NIR)
- B11 (SWIR1)
- B12 (SWIR2)

#### Spectral Indices
- NDVI
- NDWI
- MNDWI
- EVI
- AWEI_sh

#### Final Tensor
```

H × W × 16

````

### Key Preprocessing Steps
- NoData removal and masking
- Percentile clipping (2–98%)
- Radiometric normalization
- Index computation (NaN-safe)
- Channel stacking

---

## Labeling Strategy (Weak Supervision)

A hybrid labeling approach was implemented:

### Rule-Based Component
- NDVI thresholding (Otsu ≈ 0.436)
- MNDWI thresholding (Otsu ≈ 0.098)
- AWEI_sh refinement for water detection

### Unsupervised Component
- KMeans clustering in spectral feature space (NDVI, NDWI, EVI)

### Fusion Strategy
Weighted voting:
- Rule-based: 2.5
- Clustering: 1.0

### Post-processing
- Morphological filtering
- Noise removal
- Small object suppression
- Boundary smoothing

---

## Model Architecture

A customized **Attention U-Net** was designed with the following components:

- Weight Standardized Convolutions
- Group Normalization (GN)
- Squeeze-and-Excitation (SE) blocks
- Attention-gated skip connections
- Bilinear decoder upsampling
- Progressive dropout regularization

### Model Size
```

7.44M trainable parameters

````

---

## Ablation Study

To evaluate the contribution of each architectural and data-driven component, we conduct systematic ablation experiments.

### 1. Effect of Attention Mechanism

| Configuration            | mIoU   | F1-score |
|------------------------|--------|----------|
| U-Net (baseline)       | 0.8910 | 0.9275   |
| + SE blocks            | 0.9054 | 0.9391   |
| + Attention gates      | 0.9186 | 0.9513   |
| Full Attention U-Net   | **0.9263** | **0.9617** |

---

### 2. Effect of Spectral Indices

| Input Configuration        | mIoU   |
|--------------------------|--------|
| RGB only                 | 0.8421 |
| RGB + spectral bands    | 0.8893 |
| + NDVI, NDWI            | 0.9102 |
| + Full index set        | **0.9263** |

---

### 3. Effect of Labeling Strategy

| Labeling Method         | mIoU   |
|------------------------|--------|
| Rule-based only        | 0.8834 |
| KMeans only            | 0.8012 |
| Fusion (proposed)      | **0.9263** |

---

### 4. Effect of Model Capacity

| Base Channels | Params | mIoU   |
|--------------|--------|--------|
| 32           | 7.4M   | 0.9230 |
| 48           | 16.7M  | 0.9227 |

### Insight
Increasing model capacity does not improve performance, indicating that:
- The problem is feature-limited, not capacity-limited
- Feature engineering (indices) is more important than model scaling

 ---
 
## Training Configuration

* Optimizer: AdamW
* Scheduler: OneCycleLR
* Loss Function:

  * Weighted Cross-Entropy Loss
  * Dice Loss
* Mixed Precision (AMP)
* Batch Size: 16
* Early stopping patience: 8 epochs

### Class Weights

```

[1.0, 1.0, 10.0]

```

---

## Evaluation Protocol

### Metrics

* Pixel Accuracy
* Mean IoU (mIoU)
* Frequency Weighted IoU
* Macro Precision / Recall / F1-score
* Cohen’s Kappa

### Validation Strategy

* Patch-based evaluation
* Stratified split (60/20/20)
* Water-aware sampling balance

---

## Results

### Overall Performance

| Metric         | Value  |
| -------------- | ------ |
| Pixel Accuracy | 0.9621 |
| Mean IoU       | 0.9263 |
| Macro F1       | 0.9617 |
| Cohen’s Kappa  | 0.9300 |

---

### Per-Class IoU

| Class      | IoU    |
| ---------- | ------ |
| Soil       | 0.9203 |
| Vegetation | 0.9332 |
| Water      | 0.9254 |

---

## Baseline Comparison

To validate the effectiveness of the proposed Attention U-Net, we compare it against standard segmentation architectures commonly used in remote sensing.

| Model                | Input Type        | mIoU   | F1-score | Params |
|---------------------|------------------|--------|----------|--------|
| Standard U-Net      | RGB only         | 0.8712 | 0.9120   | 31M    |
| ResNet-UNet         | RGB + bands      | 0.8945 | 0.9283   | 42M    |
| DeepLabV3+          | Multispectral    | 0.9071 | 0.9394   | 58M    |
| **Proposed (Attn-UNet)** | Sentinel-2 + indices | **0.9263** | **0.9617** | 7.44M |

### Key Observation
- The proposed model achieves **higher accuracy with significantly fewer parameters**
- Attention + spectral indices provide better class separation than deeper CNNs

 ---
 ## Figures

The following visualizations are available in the repository:

- Training/validation loss curves
- Confusion matrix (normalized & raw)
- Per-class IoU bar plot
- Qualitative segmentation results (RGB / GT / Prediction)
- Inference GeoTIFF output visualization

> All figures are stored in the `assets/figures/` directory.

 ---
 
## Key Findings

* Attention mechanisms significantly improve boundary delineation
* Spectral indices strongly enhance separability of water vs soil
* Model converges within ~10 epochs (stable optimization)
* Water class exhibits highest robustness across all metrics
* Primary error source: soil–vegetation spectral overlap

---

## Inference Pipeline

* Sliding window inference (256×256, stride 128)
* Cosine window blending
* Test-Time Augmentation (TTA)
* Spectral index-guided probability fusion
* DenseCRF refinement
* Morphological post-processing
* GeoTIFF export for GIS integration

---

## Reproducibility

To ensure reproducibility:

* Fixed random seeds
* Deterministic patch extraction
* Metadata logging per patch
* Train/val/test stratified sampling

---

## Installation

```bash
git clone https://github.com/hannah-fathi/attention-unet-sentinel2-landcover-segmentation.git
cd attention-unet-sentinel2-landcover-segmentation
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python src/train.py
```

### Evaluation

```bash
python src/evaluate.py
```

### Inference

```bash
python src/inference.py --input data/sample.tif
```

---

## Scientific Contribution

This work demonstrates that **combining spectral index engineering with attention-based deep learning** significantly improves segmentation performance in agricultural remote sensing applications.

Key contributions:

* Weakly-supervised labeling pipeline for Sentinel-2
* Attention-enhanced U-Net for multispectral segmentation
* Spectral index fusion strategy for class refinement
* End-to-end GIS-ready inference framework

---

## Limitations

* Limited number of land-cover classes (3-class setup)
* Single-region training (domain generalization not tested)
* Limited ablation studies on architectural components

---

## Future Work

* Multi-temporal crop monitoring
* Transformer-based segmentation (Swin-UNet, SegFormer)
* Domain adaptation across different climatic zones
* Expansion to crop-type classification
* Publication targets:

  * IEEE TGRS
  * ISPRS Journal of Photogrammetry and Remote Sensing
  * Remote Sensing (MDPI)

---

## Data Availability

The Sentinel-2 dataset used in this study is derived from Google Earth Engine (GEE) and consists of Level-2A surface reflectance products. The processed composite used in this work is available upon reasonable request.

Due to size constraints, preprocessed patches and training metadata are not included in this repository but can be regenerated using the provided preprocessing pipeline.

---

## References

[1] Ronneberger et al., 2015, U-Net, MICCAI

[2] Vaswani et al., 2017, Attention Is All You Need, NeurIPS

[3] Gao et al., 2023, Sensors

[4] Qi et al., 2024, Remote Sensing

[5] Zhao et al., 2023, arXiv

[6] Lei et al., 2024, Artificial Intelligence Review

[7] Luo et al., 2024, Information Processing in Agriculture

---

## License

MIT License
