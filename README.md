# Used Car Price Prediction 🚗💰

## Overview

This project builds an end-to-end machine learning pipeline to predict the prices of pre-owned cars using real-world data. The workflow includes data cleaning, feature engineering, exploratory data analysis (EDA), and regression modeling.

The project compares baseline performance with Linear Regression and Random Forest models to evaluate improvements achieved through feature engineering and non-linear modeling techniques.

This project was developed as a final-week inspired project based on concepts learned from the NPTEL course **“Python for Data Science.”**  
While the foundational ideas were motivated by the coursework, the problem selection, feature engineering, modeling approach, and overall project structure were independently designed and implemented as a complete, end-to-end machine learning pipeline.


## Dataset
- Source: Used car listings dataset
- Size after cleaning: **42,772 records**
- Target variable: `price`


## Key Features
- Removed duplicates and unrealistic values
- Handled missing values using statistical imputation
- Engineered **vehicle age** from registration year and month
- Applied **log transformation** to reduce target skewness
- Compared baseline, Linear Regression, and Random Forest models


## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn


## Dataset

- **Source:** Pre-owned car listings  
- **Key Features:**
  - Age of vehicle
  - Engine power (PS)
  - Kilometer driven
  - Brand, model, fuel type
  - Gearbox type
  - Damage history  

- **Target Variable:** `price` (log-transformed for modeling)

---

## Data Preprocessing & Feature Engineering

- Removed irrelevant columns and duplicate records
- Filtered unrealistic values using domain-driven working ranges
- Handled missing values using:
  - Row omission
  - Median (numerical) and mode (categorical) imputation
- Created a new **Age** feature from registration year and month
- Applied one-hot encoding to categorical variables

---

##  Exploratory Data Analysis (EDA)

- Distribution and box plots for numerical variables
- Relationship analysis:
  - Age vs Price
  - Power vs Price
  - Kilometer vs Price
- Category-wise price comparison (brand, fuel type, gearbox, damage status)

---

## Models Used

### 1️⃣ Linear Regression
- Baseline regression model
- Assumes linear relationships
- Evaluated using RMSE and R²

### 2️⃣ Random Forest Regressor
- Ensemble-based, non-linear model
- Handles feature interactions automatically
- Outperformed Linear Regression in all scenarios

---

## 📈 Evaluation Metrics

- **RMSE (Root Mean Squared Error)**
- **R² Score**
- Compared against a **baseline mean predictor**

📌 Random Forest achieved:
- Lower RMSE
- Higher R²
- Better generalization on unseen data

---

## Model Persistence

Trained models are saved using `joblib`:

- `models/linear_regression.pkl`
- `models/random_forest.pkl`

These can be directly loaded for inference or deployment.

---

## How To Run This Project
## 🔽 Clone This Repository

Open **Git Bash** (or Terminal) and run:

```bash
git clone https://github.com/<your-username>/used-car-prediction.git
cd used-car-prediction
