# High-Performance Image Pre-processing & Feature Extraction

## 📌 Overview
This repository hosts an advanced, GPU-accelerated pipeline designed for sophisticated image processing and feature extraction. Unlike standard vision scripts, this project integrates industrial-grade tools to ensure high-fidelity data preparation, making it ideal for large-scale Content-Based Image Retrieval (CBIR) or deep learning training workflows.

The system leverages **GPU acceleration** via ONNX Runtime to handle complex tasks like automated background removal and wavelet-based denoising with minimal latency.

## ✨ Key Features
* **GPU-Accelerated Background Removal:** Utilizing `rembg` with `onnxruntime-gpu` for precise foreground extraction at scale.
* **Advanced Denoising:** Implementation of **Discrete Wavelet Transform (DWT)** via `PyWavelets` for multi-resolution signal decomposition and noise reduction while preserving edge details.
* **Optimized Pipeline:** A structured `process_image_pipeline` that handles everything from raw input to refined feature vectors.
* **Ensemble Feature Analytics:** Integrated support for `XGBoost` and `LightGBM` for downstream classification or ranking tasks based on extracted descriptors.
* **Hybrid Processing:** Seamless switching between CPU and GPU to maximize hardware utilization.

## 🛠 Tech Stack
* **Hardware Acceleration:** `ONNX Runtime GPU`, `CUDA`
* **Computer Vision:** `OpenCV`, `Scikit-image`, `Rembg`
* **Signal Processing:** `PyWavelets` (Wavelet Denoising)
* **Machine Learning:** `Scikit-learn`, `XGBoost`, `LightGBM`
* **Performance:** `Joblib` for parallel processing, `TQDM` for progress tracking

## 🏗 Pipeline Architecture
1.  **Ingestion:** Batch loading of image datasets.
2.  **Denoising:** Multi-level Wavelet thresholding to remove sensor noise.
3.  **Segmentation:** Automated background removal to focus on the Region of Interest (ROI).
4.  **Feature Extraction:** Generating high-dimensional descriptors for similarity matching or classification.
5.  **Scaling:** Robust scaling of features for ensemble model compatibility.

## 🚀 Getting Started

### Prerequisites
Ensure you have a CUDA-enabled GPU and the appropriate drivers installed.

### Installation
```bash
pip install numpy opencv-python-headless scipy scikit-image rembg[gpu] onnxruntime-gpu pywavelets tqdm joblib scikit-learn
