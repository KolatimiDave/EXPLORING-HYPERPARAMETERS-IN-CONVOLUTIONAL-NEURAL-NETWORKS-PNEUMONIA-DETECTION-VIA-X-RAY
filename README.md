# Exploring Hyperparameters in Convolutional Neural Networks: Pneumonia Detection via X-ray

This repository contains the complete source code, configurations, and experimental scripts for the research project **"Exploring Hyperparameters in Convolutional Neural Networks: Pneumonia Detection via X-ray"**.

The project systematically investigates the impact of various hyperparameters on CNN performance for pneumonia detection using chest X-ray images.

---

## 📄 Project Overview

This work explores the effect of learning rate, batch size, dropout rate, number of epochs, and optimizer choice on CNN performance. Multiple architectures were evaluated, including **VGG16, ResNet50, InceptionV3, and MobileNetV2**, using transfer learning.

All experiments were executed on **Kaggle Notebooks** with a **NVIDIA Tesla P100 GPU** runtime for accelerated training.

The full Kaggle notebooks, along with this GitHub repository, are publicly available for full reproducibility.

**Hyperparameter optimization techniques used:**
- **Random Search**
- **Grid Search**
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

## 📂 Repository Structure

```
├── LICENSE                                             # License information (MIT)
├── README.md                                           # Project documentation
├── requirements.txt                                    # Python dependencies
├── exploratory-data-analysis.ipynb                    # Initial dataset exploration and visualization
├── Single-hyperparameter_exploration-architecture.ipynb
│   # One-factor-at-a-time (OFAT) analysis across CNN architectures
├── multi-factor-exploration-architecture.ipynb
│   # Multi-factor (two or more hyperparameters) exploration across architectures
├── analysis-of-results.ipynb                          # Statistical analysis, performance comparison, and result interpretation
├── combine_results.py                                  # Python script to merge results from multiple experiments
└── combined_results.csv                                # Aggregated results file from all experiments
```

---

## 🚀 Usage

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Access the Kaggle Notebook version

The repository is also mirrored on Kaggle for direct execution with GPU acceleration:

- [Exploratory Data Analysis](https://www.kaggle.com/code/kolatimidavid/exploratory-data-analysis)
- [Single Hyperparameter Exploration](https://www.kaggle.com/code/kolatimidavid/single-hyperparameter-exploration-architecture)
- [Multi-factor Exploration](https://www.kaggle.com/code/kolatimidavid/multi-factor-exploration-architecture)
- [Analysis of Results](https://www.kaggle.com/code/kolatimidavid/analysis-of-results)

### 4. Run the experiments on Kaggle

Clone the Kaggle notebooks and run them to get the results.

---

## 📊 Results

All experiment results, including confusion matrices and metrics, are saved to `/kaggle/working/`.

---

## 📖 How to Cite This Repository

If you use this code in your research or project, please cite it as follows:

**IEEE format:**

[1] D. Olukolatimi, "Exploring Hyperparameters in Convolutional Neural Networks – Complete Source Code," GitHub repository, Aug. 13, 2025. [Online]. Available: https://github.com/KolatimiDave/EXPLORING-HYPERPARAMETERS-IN-CONVOLUTIONAL-NEURAL-NETWORKS-PNEUMONIA-DETECTION-VIA-X-RAY (Accessed: 13 August 2025).
