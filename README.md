# 🛰️ Sensor Anomaly Detection - Celebal Challenge

This repository presents a pipeline built to identify anomalies in sensor readings. Leveraging **XGBoost**, this project encapsulates a complete flow - from data preprocessing and feature engineering to model tuning and prediction generation.

---

## 🚀 Objective

Develop a high-performing anomaly detection model with a strong focus on **F1-score optimization**. This solution aims to be generalizable and effective on unseen data, following structured stages:

-  Data preprocessing and exploratory analysis  
-  Feature creation and enhancement  
-  Model development with **XGBoost**  
-  Hyperparameter tuning using **Optuna**  
-  Handling class imbalance with **SMOTE**  
-  Generating submission-ready predictions  

---

## 📂 Dataset Description

- `train.parquet` → Labeled training data with sensor features and anomaly flags.  
- `test.parquet` → Unlabeled sensor data for which predictions are to be made.

The dataset includes features like `X1`, `X2`, ..., `X5`, and additional engineered time-related features (e.g., `year`, `month`, `weekend`).

---

##  Tech Stack & Libraries

- `Python 3.11`
- `pandas`, `numpy`
- `xgboost`
- `optuna` (for hyperparameter optimization)
- `scikit-learn`
- `imbalanced-learn` (for SMOTE)
- `matplotlib`, `seaborn` (for visualization)

---

## ⚙️ Feature Engineering Highlights

Features derived from the `Date` column include:

```python
df['year'] = pd.to_datetime(df['Date']).dt.year
df['month'] = pd.to_datetime(df['Date']).dt.month
df['day'] = pd.to_datetime(df['Date']).dt.day
df['dayofweek'] = pd.to_datetime(df['Date']).dt.dayofweek
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
```

---

## ⚙️ Model Pipeline

### 🔄 Preprocessing
- Extracted features from datetime columns  
- Dropped unnecessary columns (e.g., `Date`)  
- Applied `RobustScaler` for feature scaling  

### 🧬 Resampling
- Used **SMOTE** to address class imbalance in the dataset  

### 🧠 Model Training
- Trained an `XGBClassifier`  
- Optimized hyperparameters using **Optuna**  
- Evaluated using **F1-Score** with 3-fold cross-validation  

---

## 📊 Results

- ✅ **Final F1-Score**: 0.78 *(example)*
- ✅ Validated on resampled and tuned data  
- ✅ Predictions generated on `test.parquet` and saved to CSV  

