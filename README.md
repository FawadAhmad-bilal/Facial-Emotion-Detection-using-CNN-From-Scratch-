# Facial Emotion Detection using CNN (From Scratch)

A deep learning project that classifies human facial emotions from images using a Convolutional Neural Network built and trained from scratch on the FER2013 dataset, with webcam-based inference for real-world testing. This is part of a broader face analysis system in progress.

## Overview

This project trains a CNN to classify facial expressions into 7 emotion categories: **Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise**.

The model is built entirely from scratch (no pretrained/transfer learning) using TensorFlow/Keras, trained on the FER2013 dataset, and paired with OpenCV-based face detection for testing predictions on live webcam captures. It's designed as the foundation for a larger real-time face analysis pipeline (emotion, age, gender, and face recognition).

## Dataset

- **FER2013** — 35,887 grayscale facial images (48x48, resized to 128x128 for training)
- 7 emotion classes
- Source: [Kaggle - FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

## Model Architecture

- 3 Convolutional blocks (64 → 128 → 128 filters) with Batch Normalization and Max Pooling
- Data augmentation via Random Horizontal Flip
- Fully connected layers (128 → 64) with Dropout for regularization
- Softmax output layer for 7-class classification

## Training Details

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Callbacks:** EarlyStopping (patience=8, monitors val_accuracy) + ReduceLROnPlateau
- **Epochs:** Up to 40 (with early stopping)

## Results

| Metric | Score |
|---|---|
| Validation Accuracy | ~59% |
| Validation Loss | 1.30 |

> Note: FER2013 is a notoriously challenging dataset due to low-resolution, ambiguous, and mislabeled samples. A ~55-65% accuracy range is consistent with published benchmarks for CNNs trained from scratch without transfer learning.

## Webcam Inference

The trained model is tested using live webcam capture:

1. Captures a photo via webcam
2. Detects faces using OpenCV's Haar Cascade classifier
3. Crops, resizes, and normalizes each detected face
4. Feeds the face into the trained CNN for emotion prediction
5. Draws a bounding box and predicted emotion (with confidence score) on the image

> Currently implemented as click-to-capture inference (built for Google Colab, which cannot access a local camera continuously). A fully continuous, real-time video loop version — running locally with OpenCV's `VideoCapture` — is the next step.

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV (face detection + webcam capture)
- NumPy, Pandas
- Google Colab (training + inference environment)

## Project Structure

```
├── emotion_detection_model_CNN.ipynb   # Model training + webcam inference notebook
├── emotion_model.h5                    # Saved trained model
└── README.md
```

## Roadmap

- [x] Data preprocessing pipeline (loading, normalization, augmentation)
- [x] CNN architecture design and training
- [x] Model evaluation
- [x] Webcam-based inference (click-to-capture, Colab)
- [ ] Fully real-time webcam detection (continuous video loop, local OpenCV)
- [ ] Integration into a full face analysis system (age, gender, face recognition)

## Author

**Fawad Ahmad Bilal**
BS Artificial Intelligence, University of Haripur
[GitHub](https://github.com/FawadAhmad-bilal)
