# Helipad-detection

 # 🚁 Satellite Helipad Detector

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://helipad-detection-juhb9q9k7c8zzspwhekbcn.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-98.9%25-green)

A Computer Vision application that detects the presence of helipads in satellite imagery using Deep Learning. 

**🔴 [Live Demo: Click Here to Try the App](https://helipad-detection-juhb9q9k7c8zzspwhekbcn.streamlit.app/)**
**🔴 [Kaggle: Click Here to run the notebook](https://www.kaggle.com/code/moezzouarip2m/notebook70ace3d56c/notebook)**


---

## 📝 Project Overview

This project solves a binary classification problem: identifying whether a satellite image contains a **Helipad** or **No Helipad**. 

It was built as part of a Kaggle competition, achieving a top-tier score of **0.98942** on the leaderboard (The "Kylian Mbappé" of models! ⚽).

### ✨ Key Features
*   **High Accuracy:** ~98.9% accuracy on the test set.
*   **Architecture:** Uses **ResNet50** with Transfer Learning (Pre-trained on ImageNet).
*   **Robustness:** Trained with Test Time Augmentation (TTA) and geometric transforms (Rotation/Flipping) to handle satellite orientation.
*   **Real-time Interface:** Deployed using **Streamlit** for instant drag-and-drop inference.

---

## 🧠 The Model

The core of the application is a fine-tuned **ResNet50** Deep Residual Network.

*   **Input:** 224x224 RGB Satellite Images.
*   **Backbone:** ResNet50 (Frozen early layers, fine-tuned deep layers).
*   **Head:** Custom fully connected layers with Dropout (0.4) to prevent overfitting.
*   **Loss Function:** BCEWithLogitsLoss for numerical stability.


