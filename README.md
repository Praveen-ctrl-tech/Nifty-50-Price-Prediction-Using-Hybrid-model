
# NIFTY 50 – Hybrid Forecasting Model

### XGBoost + LSTM + Fusion Ensemble (Auto-Updating Dataset)

## Overview

This project implements a **hybrid machine learning + deep learning ensemble model** for forecasting **NIFTY 50 index closing prices**.
It combines:

1. **XGBoost** – Captures short-term price patterns and engineered features
2. **LSTM** – Learns long-term temporal dependencies from sequential price data
3. **Fusion Ensemble Layer (Linear Regression)** – Optimally blends both predictions

The pipeline automatically downloads and updates **25 years of historical NIFTY 50 data** using *Yahoo Finance (yfinance)*.
This project is built entirely in Python and provides a complete, error-free forecasting workflow.

---

## Key Features

### 1. **Automatic Data Update**

* Downloads full historical NIFTY 50 data (25 years)
* If a local dataset exists, only **new rows** are fetched and appended

### 2. **Feature Engineering**

* Moving averages: **MA7**, **MA21**
* **Daily return**
* Cleaned and chronologically sorted dataset

### 3. **XGBoost Regression Model**

* Predicts closing prices using numerical + engineered features
* Handles non-linear patterns well

### 4. **LSTM Deep Learning Model**

* Sequence length: 60 days
* Predicts next-day price based on past 60-day close values
* Architecture:

  * LSTM(64) → LSTM(32) → Dense(1)

### 5. **Fusion Ensemble Layer**

Final prediction = Linear regression combining:

* XGBoost prediction
* LSTM prediction

This usually reduces variance and improves robustness.

### 6. **Evaluation Metrics**

The project calculates:

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error
* **MAPE** – Mean Absolute Percentage Error
* **R² Score**
* **Directional Accuracy** (up/down prediction accuracy)

### 7. **Visualization**

Generates a plot showing:

* Actual closing prices
* Hybrid model predictions

---


## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Minimum libraries required:

```
numpy
pandas
matplotlib
xgboost
scikit-learn
tensorflow
yfinance
```

---

## How It Works

### Step 1: Load / Update NIFTY Data

* Checks if `nifty.csv` exists
* Downloads 25 years if missing
* Otherwise only new daily rows are appended

### Step 2: Clean & Engineer Features

* Calculate MA7, MA21, returns
* Remove missing values
* Prepare feature matrix for ML

### Step 3: Train XGBoost

Predicts `Close` using engineered predictors.

### Step 4: Train LSTM

Predicts the close value for day *t* using a 60-day window.

### Step 5: Fusion Ensemble

A linear regression model is trained on the combined predictions:

```
final_pred = a * pred_ml + b * pred_lstm + c
```

### Step 6: Evaluate

Generates metrics and visual comparison.

---

## Sample Output (Console)

```
================== MODEL PERFORMANCE ==================
MAE  : 34.851
RMSE : 56.412
MAPE : 0.98 %
R²   : 0.9847
Directional Accuracy : 63.20 %
=======================================================
```

---

## Visualization

A line plot compares actual vs hybrid predicted prices.

---

## Use Cases

* Financial time series forecasting
* Stock market analytics
* Portfolio decision support
* Research on ML + DL hybrid models
* Resume-ready project showcasing:

  * Feature engineering
  * XGBoost
  * LSTM
  * Ensemble modeling
  * Data pipeline automation

---

## Future Enhancements

* Add GRU model
* Add hyperparameter tuning (Optuna / Bayesian Optimization)
* Deploy using Flask / FastAPI
* Create dashboard using Streamlit
* Add multi-step forecasting

---

