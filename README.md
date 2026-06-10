<div align="center">

# 🚗 OptiCar — Used Car Price Prediction

**An end-to-end machine learning pipeline for predicting pre-owned car prices.**

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-purple?style=flat-square&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📌 Overview

**OptiCar** is an end-to-end machine learning project that predicts the resale price of used cars from real-world listing data. It covers the full ML workflow — from raw data cleaning and feature engineering to model training, evaluation, and persistence.

The project benchmarks a **Linear Regression** baseline against a **Random Forest Regressor**, demonstrating the impact of feature engineering and non-linear modeling.

> Developed as a final-week capstone project inspired by the NPTEL course **"Python for Data Science"**.  
> Problem selection, feature engineering, modeling strategy, and project structure were independently designed and implemented.

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | Pre-owned car listings |
| Size (post-cleaning) | 42,772 records |
| Target Variable | `price` (log-transformed) |

**Key features used:**

- `age` — Vehicle age (engineered from registration year/month)
- `powerPS` — Engine power (in PS)
- `kilometer` — Odometer reading
- `brand`, `model` — Manufacturer and model name
- `fuelType` — Petrol, diesel, electric, etc.
- `gearbox` — Manual or automatic
- `notRepairedDamage` — Damage history flag

---

## 🛠️ Project Structure

```
opticar/
│
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned & engineered data
│
├── notebooks/
│   └── opticar_pipeline.ipynb
│
├── models/
│   ├── linear_regression.pkl
│   └── random_forest.pkl
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Data Preprocessing & Feature Engineering

- **Removed** irrelevant columns and duplicate records
- **Filtered** outliers using domain-driven value ranges (e.g. realistic price, age, and power bounds)
- **Handled missing values** via:
  - Row omission for small gaps
  - Median imputation for numerical features
  - Mode imputation for categorical features
- **Engineered `age`** from registration year and month relative to the listing date
- **Log-transformed `price`** to reduce right skew and stabilise regression
- **One-hot encoded** all categorical variables

---

## 📈 Exploratory Data Analysis

EDA focused on understanding feature relationships with price:

| Analysis | Insight |
|---|---|
| Age vs Price | Newer cars command significantly higher prices |
| Power vs Price | Strong positive correlation — higher PS = higher price |
| Kilometer vs Price | Expected negative correlation |
| Brand comparison | Luxury brands (e.g. BMW, Mercedes) skew prices high |
| Fuel type | Diesel cars generally priced higher than petrol |
| Damage status | Damaged cars see a substantial price drop |

---

## 🤖 Models

### 1. Linear Regression (Baseline)
- Assumes linear relationships between features and log-price
- Fast to train; interpretable coefficients
- Used as the performance baseline

### 2. Random Forest Regressor
- Ensemble of decision trees
- Handles non-linearity and feature interactions automatically
- Outperformed Linear Regression across all metrics

---

## 📉 Results

| Model | RMSE ↓ | R² ↑ |
|---|---|---|
| Baseline (Mean Predictor) | — | 0.000 |
| Linear Regression | Lower | Higher |
| **Random Forest** | **Lowest** | **Highest** |

> Random Forest achieved the best generalisation on unseen data with lower RMSE and higher R² compared to both baselines.

---

## 💾 Model Persistence

Trained models are saved using `joblib` for reuse and deployment:

```python
import joblib

# Load a saved model
model = joblib.load("models/random_forest.pkl")

# Predict on new data
predictions = model.predict(X_new)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/opticar.git
cd opticar
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline

```bash
jupyter notebook notebooks/opticar_pipeline.ipynb
```

Or run individual modules:

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

---

## 🧰 Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisations |
| `joblib` | Model serialisation |

---

## 📚 Inspiration

This project was built upon concepts from the NPTEL course **"Python for Data Science"** (IIT Madras). The dataset choice, feature engineering approach, model comparison framework, and overall pipeline architecture were independently conceived and implemented.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
