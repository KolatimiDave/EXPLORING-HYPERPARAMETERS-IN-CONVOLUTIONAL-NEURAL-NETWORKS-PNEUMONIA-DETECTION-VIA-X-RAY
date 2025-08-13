# Exploring Hyperparameters in Convolutional Neural Networks: Pneumonia Detection via X-ray

This repository contains the complete source code, configurations, and experimental scripts for the research project **"Exploring Hyperparameters in Convolutional Neural Networks: Pneumonia Detection via X-ray"**.  
The project systematically investigates the impact of various hyperparameters on CNN performance for pneumonia detection using chest X-ray images.

---

## 📄 Project Overview
This work explores the effect of learning rate, batch size, dropout rate, number of epochs, and optimizer choice on CNN performance. Multiple architectures were evaluated, including **VGG16, ResNet50, InceptionV3, and MobileNetV2**, using transfer learning.


All experiments were executed on **Kaggle Notebooks** with a **NVIDIA Tesla P100 GPU** runtime for accelerated training.  
The full Kaggle notebooks, along with this GitHub repository, are publicly available for full reproducibility.


Experiments were conducted with:
- **Grid Search**
- **Random Search**
- **Bayesian Optimization** (Optuna)

---

## 🛠 Features
- **DICOM image preprocessing** (resizing, RGB conversion, normalization)
- **Stratified data splits** (70% train / 15% validation / 15% test)
- **GPU-accelerated training** with TensorFlow
- **Custom F1-score metric** in Keras
- **Automated hyperparameter optimization** with Optuna
- **Performance tracking** with confusion matrices and saved trial metrics

---
