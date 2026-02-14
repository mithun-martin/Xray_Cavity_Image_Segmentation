# Automated Dental Caries Segmentation using Deep Learning

## Overview
This project implements a deep learning pipeline for the automated semantic segmentation of dental caries (cavities) in panoramic X-ray images. In medical image analysis, segmenting early-stage lesions is critical for computer-aided diagnosis. 

The primary engineering challenge addressed in this project is **severe class imbalance**—where healthy teeth and background pixels vastly outnumber the tiny pixel footprint of cavities. 

## Technical Architecture


The core model utilizes a **U-Net++** architecture to preserve fine-grained, low-level details of small lesions that are typically lost in standard downsampling.
* **Encoder:** ResNet34 (Initialized with ImageNet weights for robust feature extraction).
* **Attention Mechanism:** Integrated Spatial and Channel Squeeze and Excitation (scSE) modules in the decoder to dynamically suppress background noise (gums/jawbone) and amplify cavity feature maps.

## Engineering Highlights
* **Loss Function Engineering:** Replaced standard Binary Cross-Entropy with a custom combination of **Focal Tversky Loss** (to heavily penalize False Negatives) and **Weighted BCE** (`pos_weight=200.0`). This forces the model to prioritize faint, difficult-to-detect decay.
* **Operating Point Calibration:** Optimized the decision threshold to `0.10` to maximize clinical safety, prioritizing Sensitivity (Recall) without destroying Specificity.
* **Robust Augmentation:** Developed a dynamic Albumentations pipeline including `CLAHE` (Contrast Limited Adaptive Histogram Equalization) to standardize washed-out X-ray exposures, alongside Affine and Elastic transforms to ensure Out-Of-Distribution (OOD) generalization.
* **Advanced Optimization:** Utilized the `AdamW` optimizer paired with a `OneCycleLR` learning rate scheduler to rapidly escape local minimums and achieve fast convergence.

## Results
Tested on a pediatric dental panoramic radiograph dataset (512x512 resolution). The model achieved highly clinically viable metrics:
* **Intersection over Union (IoU):** 0.7091
* **Recall (Sensitivity):** 99.79%
* **Specificity:** 91.89%

## Tech Stack
* **Deep Learning:** PyTorch, Torch Automatic Mixed Precision (AMP)
* **Model Architectures:** Segmentation Models PyTorch (SMP)
* **Computer Vision:** OpenCV, Albumentations
* **Data Manipulation:** NumPy, Pandas

## Project Structure


```text
├── data/                   # (Not uploaded) Contains Train/Test image and mask splits
├── notebooks/              # Jupyter/Kaggle notebooks containing the training pipeline
├── checkpoints/            # Saved model weights (.pth)
└── README.md
