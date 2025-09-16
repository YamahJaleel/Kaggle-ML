# Kaggle-ML

Machine Learning experiments & models using datasets from Kaggle, focused on voice / gender classification.

---

## Table of Contents

- [About](#about)  
- [Contents](#contents)  
- [Why `voice_gender1.py` is the Best Version](#why-voice_gender1py-is-the-best-version)  
- [Requirements](#requirements)  
- [Usage](#usage)  
- [Model Details](#model-details)  
- [Examples](#examples)  
- [Contributing](#contributing)  
- [License](#license)  

---

## About

This repository contains Python scripts, sample audio files, and a trained model for classifying voice by gender. The goal is to experiment with different model approaches, feature extraction, and compare performance, especially on Kaggle’s datasets.

---

## Contents

| File | Description |
|---|---|
| `voice_gender.py` | Base version for voice gender classification. |
| `voice_gender1.py` / `voice_gender2.py` | Enhanced / experimental versions—particularly `voice_gender1.py` is the most advanced. |
| `voice_gender_model.pkl` | Serialized model (pickle format) for inference. |
| `Sample_Voice.wav` / `Sample_Voice2.mp3` | Example audio files used for testing / demonstration. |

---

## Why `voice_gender1.py` is the Best Version

`voice_gender1.py` is considered the best iteration because it:

- Uses a richer set of libraries for both model building and audio processing (including **`librosa`**, **`sklearn`**, **`xgboost`**, etc.), which allows more robust modeling and feature extraction.  
- Includes hyperparameter tuning (GridSearchCV) on the RandomForest model to find best parameters (`n_estimators`, `max_depth`, `max_features`).  
- Compares two strong model families (Random Forest and XGBoost) and selects the better one based on test accuracy.  
- Has a detailed audio-feature extraction pipeline:  
    – Fourier transform features (power spectrum, frequency bins up to ~280 Hz)  
    – Statistical moments: mean, standard deviation, skewness, kurtosis  
    – Information-theoretic features like entropy, flatness  
    – Temporal features and spectral peaks etc.  
- Provides the ability to make predictions on new audio files via `predict_gender(...)`, taking an audio file, extracting the same features, loading the trained model, and returning both predicted label and confidence.

These make `voice_gender1.py` more robust, flexible, and generally higher performing than simpler versions.

---

## Requirements

Make sure you have the following installed:

- Python 3.x  
- `numpy`  
- `pandas`  
- `librosa` (for audio feature extraction)  
- `scikit-learn`  
- `xgboost`  
- Any other dependencies (like `kagglehub`, if used)  

You can install via pip:

```bash
pip install numpy pandas scikit-learn librosa xgboost
