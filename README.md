# VERA-Multiwavelet-CNN

**MATLAB implementation of the proposed Multiwavelet-CNN biometric recognition framework evaluated on VERA Palm Vein and VERA Finger Vein datasets.**

---

**Paper:** Lightweight Deep Identification Model based on Multi-Wavelet Transform with Multimodal Fusion of Palm and Finger Veins  
**Journal:** International Journal of Intelligent Engineering and Systems (IJIES), 2025  
**Authors:** Athmar Haider Fadhil, Ahmed Aldhahab, Hanaa Mohsin Ali Al Abboodi  
**Affiliation:** Department of Electrical Engineering, University of Babylon, Iraq  
**Contact:** hanaa.ali@uobabylon.edu.iq

---

## ⚠️ DATASET ACCESS WARNING

> **The VERA Finger Vein and VERA Palm Vein datasets are protected research datasets provided by the Idiap Research Institute.**
>
> **The datasets are NOT publicly distributed through this repository.**
>
> **Access to the datasets requires authorization from the dataset owners.**
>
> Researchers must obtain official permission and comply with all usage conditions before using the datasets. Access may require the submission of an application, approval by the dataset owners, and the signing of a data usage agreement. Users are solely responsible for obtaining all required permissions before conducting experiments.

| Database | Official Link |
|---|---|
| **Dataset (Palm Vein)** | https://www.idiap.ch/en/scientific-research/data/vera-palmvein |
| **Dataset (Finger Vein)** | https://www.idiap.ch/en/scientific-research/data/vera-fingervein |

---

## Reproducibility Statement

This repository contains the **complete MATLAB implementation** used for preprocessing, feature extraction, CNN training, and comparative experiments reported in the associated publication. Due to licensing restrictions, the VERA datasets are not redistributed. Researchers can reproduce all experiments after obtaining official dataset access from the Idiap Research Institute.

---

## MATLAB Requirements

| Requirement | Version |
|---|---|
| MATLAB | R2021a or newer |
| Image Processing Toolbox | required |
| Deep Learning Toolbox | required |
| Statistics and Machine Learning Toolbox | required |

---

## Repository Structure

```
VERA-Multiwavelet-CNN/
│
│  ── Proposed Method ──────────────────────────────
├── Multiwavelet_pre_feature.m   Feature extraction (2D-DMWT)
├── multiwalide_multiwavelet.m   2D-DMWT core function
├── generate.m                   Wavelet matrix (GHM)
├── cspreproc1.m                 Critical-sampling preprocessing
├── permutation.m                Row permutation
├── pppermutesub.m               Sub-band permutation
├── pppermuterc.m                Row/column permutation
├── CNN_Multiwavelet.m           CNN training (palm/finger/fusion)
│
│  ── Comparative Baseline ─────────────────────────
├── pre_1.m                      Image resize preprocessing
├── CLAHE.m                      CLAHE enhancement
├── DWT.m                        2D-DWT baseline extraction
├── CNN_Aug_LL2.m                CNN training (DWT baseline)
│
│  ── Evaluation & Reproducibility ─────────────────
├── evaluate.m                   Full metrics (Acc/Prec/Rec/F1/Spec)
├── generate_split_lists.m       Saves train/val split lists
│
│  ── Documentation ────────────────────────────────
├── sample_results/              Example outputs and figures
├── LICENSE
└── README.md
```

---

## Feature Extraction Pipeline

| Step | Operation |
|---|---|
| 1 | Image acquisition |
| 2 | RGB-to-grayscale conversion |
| 3 | Resize to 256×256 |
| 4 | CLAHE contrast enhancement |
| 5 | 2D-DMWT decomposition (GHM filter) |
| 6 | Extract main LL sub-band (128×128) |
| 7 | Partition LL into four sub-regions |
| 8 | Average sub-regions → 64×64 feature map (LL2) |
| 9 | Flatten to 4096-dimensional feature vector |
| 10 | CNN-based classification |

---

## Dataset Paths

```matlab
% Palm Vein
dataRoot = 'C:\Users\hp\Desktop\VERA-Palmvein\VERA-Palmvein\raw';

% Finger Vein
dataRoot = 'C:\Users\hp\Desktop\VERA-Fingervein\VERA-Fingervein\raw';
```

---

## How to Reproduce the Experiments

### Proposed Method (2D-DMWT)

```
1. pre_1.m                       (resize raw images)
2. CLAHE.m                       (CLAHE enhancement)
3. Multiwavelet_pre_feature.m    (extract DMWT features)
4. CNN_Multiwavelet.m            (set SCENARIO = 'palm' | 'finger' | 'fusion')
5. evaluate.m                    (compute full metrics)
```

### Comparative Baseline (2D-DWT)

```
1. pre_1.m       (resize raw images)
2. CLAHE.m       (CLAHE enhancement)
3. DWT.m         (extract DWT LL2 features)
4. CNN_Aug_LL2.m (train and evaluate)
5. evaluate.m    (compute full metrics)
```

### Generate Train/Validation Split Lists

```matlab
run generate_split_lists.m
% Saves to split_lists/ folder
% Contains: person IDs, session IDs, filenames, class labels
% for all six evaluation scenarios (palm/finger/fusion × train/val)
```

---

## Training Configuration (Table 3)

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Learning rate drop | ×0.1 at epoch 70 |
| Max epochs | 80 |
| Mini-batch size | 16 |
| L2 regularization | 0.0001 |
| Augmentation | RandXReflection, Translation ±2px, Scale [0.98–1.02] |
| Applied to | Training subset only |
| Split | 80/20 stratified per identity |
| **Random seed** | **42** |

---

## Reported Results (VERA Database)

| Scenario | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Palm Identification | 99.77% | 99.82% | 99.77% | 99.77% |
| Finger Identification | 85.23% | 65.83 | 68.18 | 66.38 |
| Palm-Finger Fusion | 90.34% | 81.23 | 76.52 | 78.80 |
| Palm Gender | 100% | 100% | 100% | 100% |
| Finger Gender | 93.18% | 92.73 | 91.07 | 91.89 |
| Fusion Gender | 98.86% | 93.92 | 96.47 | 95.18 |

---

## Citation

```
Athmar H. Fadhil, Ahmed Aldhahab, Hanaa M. Al Abboodi,
"Lightweight Deep Identification Model based on Multi-Wavelet Transform
with Multimodal Fusion of Palm and Finger Veins",
International Journal of Intelligent Engineering and Systems (IJIES), 2025.
```

*Citation details will be updated after publication.*
