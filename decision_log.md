# Decision Log – Aerial GCP Pose Estimation

## 1. Problem Understanding

The objective of this assignment is to build a computer vision pipeline capable of processing aerial imagery and performing two tasks simultaneously:

1. **Keypoint Localization** – Predict the pixel coordinates `(x, y)` of the center of a Ground Control Point (GCP) marker.
2. **Shape Classification** – Identify the physical shape of the marker from three possible classes:

* Cross
* Square
* L-Shaped

This is a **multi-task learning problem**, combining a **regression task (localization)** and a **classification task (shape recognition)**.

The primary engineering challenge lies in accurately predicting the marker center from high-resolution aerial images while maintaining good classification performance.

---

# 2. Dataset Analysis

The dataset consists of high-resolution aerial images (2048×1365) organized in a nested folder structure simulating a real production environment.

Key observations during exploratory analysis:

* Images contain **exactly one GCP marker**.
* Marker shapes vary between Cross, Square, and L-Shape.
* Some annotation entries were missing `verified_shape`, which required filtering.
* Image resolution is significantly larger than what most CNN backbones expect.

### Dataset Challenges

1. Large image size → high memory usage.
2. Nested directory structure → requires custom loader.
3. Missing shape labels → must skip invalid samples.

To address these issues, a preprocessing pipeline was implemented.

---

# 3. Preprocessing Strategy

Images were resized to:

224 × 224

This decision was made because:

* It matches the expected input resolution for **ResNet architectures**.
* It significantly reduces GPU memory usage.
* It allows faster experimentation and training.

Marker coordinates were normalized to the range `[0, 1]`:

x_norm = x / width
y_norm = y / height

This makes the regression task **resolution independent** and stabilizes training.

Image normalization used **ImageNet statistics** since the backbone network is pretrained.

---

# 4. Data Augmentation

Drone imagery can vary in orientation and lighting conditions. To improve generalization, the following augmentations were applied:

* Horizontal flipping
* Shift/scale/rotate transformations
* Random brightness and contrast adjustments
* Gaussian noise

These augmentations simulate variations such as:

* drone orientation
* lighting conditions
* slight positional changes

Keypoints were transformed alongside the images using **Albumentations keypoint augmentation support**.

---

# 5. Model Architecture

A **multi-task convolutional neural network** was implemented.

### Backbone

ResNet18 (pretrained on ImageNet)

Reasons for choosing ResNet18:

* Lightweight and fast to train
* Strong feature extraction capabilities
* Well-suited for transfer learning
* Easily adaptable to multi-task outputs

---

### Multi-Task Heads

The backbone produces a shared feature representation which is passed to two separate heads.

#### Regression Head

Predicts normalized coordinates `(x, y)` representing the center of the marker.

Loss function:

SmoothL1Loss

This loss was selected instead of MSE because it is less sensitive to large outliers during early training.

---

#### Classification Head

Predicts the marker shape.

Classes:

Cross
Square
L-Shape

Loss function:

CrossEntropyLoss

To handle class imbalance, class weights were applied during training.

---

# 6. Training Strategy

The following training configuration was used:

Optimizer:

AdamW

Learning rate schedule:

CosineAnnealingLR

Additional techniques used:

* Mixed precision training for faster computation
* Gradient clipping to stabilize training
* Weighted classification loss to address class imbalance

Total loss function:

L_total = λ * L_localization + L_classification

where

λ = 50

This weighting ensures the localization task receives sufficient training signal.

---

# 7. Evaluation Metrics

Two separate evaluation metrics were used.

### Localization Accuracy

Percentage of Correct Keypoints (PCK)

A prediction is considered correct if the predicted center lies within a threshold distance from the ground truth.

Thresholds evaluated:

* PCK@10px
* PCK@25px
* PCK@50px

---

### Classification Performance

Macro F1 Score was used to evaluate classification performance across the three marker classes.

Macro F1 is preferred because it treats each class equally regardless of dataset imbalance.

---

# 8. Challenges Encountered

Several challenges were encountered during development.

### 1. Coordinate Scaling Errors

Early training runs produced zero PCK scores due to incorrect coordinate normalization after image resizing.

Solution:

Coordinates were normalized relative to the resized image dimensions to ensure consistency between labels and predictions.

---

### 2. Regression Instability

Using Mean Squared Error resulted in unstable localization early in training.

Solution:

SmoothL1Loss was used instead.

---

### 3. Class Imbalance

Some marker shapes appeared more frequently than others.

Solution:

Weighted CrossEntropyLoss was implemented to balance training.

---

# 9. Future Improvements

Several improvements could further enhance performance:

* Heatmap-based keypoint detection architectures
* Higher resolution training (e.g., 384×384 inputs)
* YOLO-based keypoint detection models
* Multi-scale training strategies
* Hard example mining

---

# 10. Summary

This project demonstrates a practical computer vision pipeline for **automated Ground Control Point localization and classification**.

The solution uses a **multi-task ResNet18 architecture**, combining regression and classification within a single network.

Key strengths of the system include:

* Efficient training using transfer learning
* Robust data augmentation
* Balanced multi-task learning
* Reproducible training and inference pipeline
