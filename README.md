# Credit-Card-Fraud-Detection-Using-Machine-Learning-model-LogisticRegression

# 💳 Credit Card Fraud Detection System

A Machine Learning based Credit Card Fraud Detection system built using Logistic Regression and deployed using Flask.

This project detects fraudulent credit card transactions using supervised machine learning techniques and provides real-time predictions through a web interface.

---

# 📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small percentage of total transactions.

This project:

- Handles class imbalance
- Performs data preprocessing
- Trains a Logistic Regression model
- Saves preprocessing objects and model
- Deploys the model using Flask
- Provides real-time fraud prediction via a web app

The focus of the system is on maximizing **Recall for fraud class** to reduce missed fraud cases.

---

# 🚀 Key Features

- Data Cleaning & Preprocessing
- Missing Value Imputation
- Feature Scaling using RobustScaler
- Logistic Regression with `class_weight='balanced'`
- Model Serialization using Pickle
- Flask Web Application for Prediction
- Organized & Production-ready Project Structure

---

# 🧠 Machine Learning Details

### Algorithm Used:
- Logistic Regression

### Why Logistic Regression?
- Interpretable
- Fast training
- Strong baseline for binary classification
- Works well with scaled numerical features

### Imbalance Handling:
- `class_weight='balanced'`

### Evaluation Metrics:
- Accuracy
- Precision
- Recall (Fraud Class Focus)
- F1-Score
- ROC-AUC Score
- Confusion Matrix

Fraud detection prioritizes **Recall over Accuracy** because missing a fraud transaction is more costly than flagging a normal transaction.

---

# 📂 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── app/
│   ├── static/
│   ├── templates/
│   └── app.py
├── data/
│   ├── raw/
|      ├── creditcard.csv
|
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── imputer.pkl
│   └── columns.pkl
│
├── notebooks/
│   ├── Feature_Engineering.ipynb
│   ├── Credit Card Fraud Detection Decision Tree.ipynb
│   └── Credit Card Fraud Detection Support Vector.ipynb
│
├── tests/
|  ├── train_model.py
|
├── venv/
|
├── creditcard.csv
├── requirements.txt
└── .gitignore
```

