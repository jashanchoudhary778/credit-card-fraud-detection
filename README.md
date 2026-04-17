# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using Random Forest. The dataset is highly imbalanced — only 0.17% of transactions are fraud — making this a realistic and challenging classification problem.

---

## Problem Statement

Credit card fraud causes billions of dollars in losses every year. The challenge is identifying the rare fraudulent transaction (1 in 578) without flagging too many legitimate ones. Standard accuracy is misleading here — a model that predicts "not fraud" every time still gets 99.8% accuracy. The real metric is **F1-score on the fraud class**.

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud cases:** 492 (0.17%)
- **Features:** 30 — Time, Amount, and V1–V28 (PCA-transformed for confidentiality)

---

## Approach

### Preprocessing
- Split features and target (`Class`: 0 = normal, 1 = fraud)
- 80/20 train-test split with `stratify=y` to preserve class ratio
- Applied `StandardScaler` on all features (fit on train, transform on test)

### EDA
- Compared transaction amount distributions for normal vs fraud
- Analyzed fraud rate by hour of day (extracted from `Time` column)
- Visualized amount distributions using histograms and box plots

### Model
- **Random Forest Classifier**
  - `n_estimators=50`
  - `max_depth=10`
  - `class_weight='balanced_subsample'` — handles class imbalance without oversampling
  - `n_jobs=-1` — parallel training

---

## Results

| Metric | Normal (0) | Fraud (1) |
|---|---|---|
| Precision | 1.00 | 0.82 |
| Recall | 1.00 | 0.83 |
| F1-Score | 1.00 | 0.82 |

**Confusion Matrix:**
```
[[56846    18]
 [   17    81]]
```

- Caught **81 out of 98 fraud cases** (83% recall)
- Only **18 false alarms** out of 56,864 normal transactions
- Overall accuracy: **99.9%**

---

## Key Insight from EDA

Fraud transactions are more evenly distributed across hours of the day compared to normal transactions, which peak during business hours. This time-based pattern is a useful signal for detection.

---

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn

---

## How to Run

```bash
git clone https://github.com/jashanchoudhary778/credit-card-fraud-detection
cd credit-card-fraud-detection
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
```

> Download the dataset from Kaggle and place `creditcard.csv` in the root folder before running.

---

## What I Learned

- Handling severe class imbalance using `class_weight` instead of oversampling
- Why accuracy is a misleading metric on imbalanced datasets
- How to use stratified splitting to preserve class distribution
- Extracting time-based features from raw timestamp columns
