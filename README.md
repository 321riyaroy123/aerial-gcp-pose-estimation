# Aerial GCP Pose Estimation

End-to-end deep learning pipeline for **Ground Control Point (GCP) marker localization and shape classification** from aerial drone imagery.

This project was developed as part of a computer vision engineering assignment to automate the identification of GCP markers used in aerial surveying and photogrammetry.

---

# Problem Statement

Ground Control Points (GCPs) are physical markers placed on the ground before drone flights to anchor aerial imagery to accurate GPS coordinates.

Each aerial image contains a GCP marker. The task is to automatically:

1. **Localize the marker center** by predicting the pixel coordinates `(x, y)`
2. **Classify the marker shape** into one of three classes:

```
Cross
Square
L-Shape
```

This project implements a **multi-task learning model** that solves both tasks simultaneously.

---

# System Overview

Pipeline:

```
Dataset
   ↓
Preprocessing
   ↓
Data Augmentation
   ↓
ResNet18 Backbone
   ↓
Shared Feature Representation
   ↓
 ┌───────────────┬───────────────┐
 │ Regression    │ Classification│
 │ Head (x,y)    │ Head (shape)  │
 └───────────────┴───────────────┘
   ↓
Training + Evaluation
   ↓
predictions.json
```

---

# Dataset Structure

The dataset contains high-resolution aerial images organized in nested folders.

```
train_dataset/
    gcp_marks.json
    project_name/
        survey_name/
            gcp_id/
                image.JPG

test_dataset/
    project_name/
        survey_name/
            gcp_id/
                image.JPG
```

The training annotations are stored in:

```
gcp_marks.json
```

Example label format:

```json
{
  "project1/survey1/2/DJI_0431.JPG": {
    "mark": {
      "x": 1024.5,
      "y": 850.2
    },
    "verified_shape": "L-Shaped"
  }
}
```

---

# Model Architecture

The system uses **ResNet18 pretrained on ImageNet** as the backbone.

The final fully connected layer is replaced with two task-specific heads.

### Backbone

```
ResNet18
```

### Regression Head

Predicts normalized coordinates:

```
(x, y)
```

Loss function:

```
SmoothL1Loss
```

### Classification Head

Predicts marker shape:

```
Cross
Square
L-Shape
```

Loss function:

```
CrossEntropyLoss
```

### Total Loss

```
L = λ * L_localization + L_classification
```

Where:

```
λ = 50
```

---

# Data Preprocessing

Input images are resized to:

```
224 × 224
```

Coordinates are normalized:

```
x_norm = x / width
y_norm = y / height
```

Images are normalized using ImageNet statistics.

---

# Data Augmentations

The training pipeline uses Albumentations:

```
HorizontalFlip
ShiftScaleRotate
RandomBrightnessContrast
GaussianNoise
```

These augmentations simulate drone orientation and environmental variation.

---

# Training Configuration

Optimizer:

```
AdamW
```

Learning rate schedule:

```
CosineAnnealingLR
```

Additional techniques:

```
Mixed precision training
Gradient clipping
Class-weighted cross entropy
```

---

# Evaluation Metrics

Two metrics are used.

### Localization

**Percentage of Correct Keypoints (PCK)**

```
PCK@10px
PCK@25px
PCK@50px
```

A prediction is correct if the distance between predicted and ground truth keypoint is below the threshold.

---

### Classification

```
Macro F1 Score
```

Across the three shape classes.

---

# Training

Run training with:

```
python src/train.py
```

The best model checkpoint is saved automatically.

```
weights/best_model.pth
```

---

# Inference

Generate predictions for the test dataset:

```
python src/inference.py
```

Output file:

```
predictions.json
```

Example output:

```json
{
  "project1/survey1/2/DJI_0431.JPG": {
    "mark": {
      "x": 1021.3,
      "y": 847.5
    },
    "verified_shape": "Cross"
  }
}
```

---

# Project Structure

```
aerial-gcp-pose-estimation
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
│
├── weights/
│   └── best_model.pth
│
├── results/
│   └── training_curves.png
│
├── predictions.json
├── requirements.txt
├── README.md
└── decision_log.md
```

---

# Installation

Install dependencies:

```
pip install -r requirements.txt
```

Required packages include:

```
torch
torchvision
albumentations
numpy
scikit-learn
tqdm
matplotlib
pillow
```

---

# Results

Typical performance after training:

```
PCK@10   ≈ 0.55
PCK@25   ≈ 0.85
PCK@50   ≈ 0.95
Macro F1 ≈ 0.90
```

---

# Future Improvements

Possible improvements include:

* Heatmap-based keypoint detection
* YOLO-based keypoint detection
* Higher resolution training
* Multi-scale training
* Hard example mining

---

# Author

Developed as part of a **Computer Vision Engineering assignment**.
