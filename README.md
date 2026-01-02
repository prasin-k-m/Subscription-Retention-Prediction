# Subscription Retention (Customer Churn) Prediction

## Problem Statement

Subscription-based businesses often lose revenue due to customer churn.
The goal of this project is to predict whether a customer is likely to churn using historical customer data, enabling businesses to take proactive retention actions.

# Business Use Case

Identify customers at high risk of churn

Support targeted retention campaigns

Improve customer lifetime value (CLV)

Assist business and marketing teams in decision-making

# Project Overview

This project implements a supervised machine learning classification pipeline to predict customer churn.
Multiple models were trained and evaluated, and the final model was selected based on generalization performance rather than accuracy alone.

The project also includes a Python-based application for real-time churn prediction.

# Dataset Information

Source: Kaggle

Type: Subscription-based customer dataset

Target Variable: Churn

Class Distribution: Imbalanced (churned customers are the minority class)

# Machine Learning Workflow

Data cleaning and preprocessing

Handling categorical variables (encoding)

Feature scaling

Train–test split

Training multiple classification models

Model evaluation using classification metrics

Overfitting analysis (train vs test performance)

Hyperparameter tuning of the final model

# Models Trained & Evaluated

The following models were trained and compared:

K-Nearest Neighbors (KNN)

Decision Tree

Logistic Regression

Support Vector Classifier (SVC)

Random Forest

AdaBoost

Gaussian Naive Bayes

# Overfitting Analysis

KNN, Decision Tree, SVC, Random Forest, AdaBoost

Very high training accuracy

Significantly lower test performance

Identified as overfitting models

Logistic Regression & Gaussian Naive Bayes

Comparable train and test performance

Better generalization on unseen data

# Final Model Selection

After comparing all models using precision, recall, F1-score, and confusion matrices,
Logistic Regression was selected as the final model because:

It showed good generalization

Balanced bias–variance tradeoff

Better interpretability

More reliable performance on the minority (churn) class

Hyperparameter tuning was applied to further optimize the Logistic Regression model, and the final model was saved using pickle.

# Model Evaluation (Final Model – Logistic Regression)
Test Set Performance

Accuracy: ~0.67

Recall (Churn class): ~0.61

F1-Score (Churn class): ~0.27

📌 Recall was prioritized over accuracy, as identifying churned customers is more important than predicting non-churn.

# Application Interface

The project includes a Python application that allows users to:

Manually input customer features

Receive real-time churn predictions

# Planned Enhancement

Customer ID–based prediction, where input features are automatically retrieved for existing customers.

# Tech Stack

Python

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

# How to Run the Project
1. Clone the repository
git clone https://github.com/prasin-k-m/Subscription-Retention-Prediction.git
cd Subscription-Retention-Prediction

2. Install dependencies
pip install -r requirements.txt

3. Run the application
python app_manual_input.py

4. Explore the model development notebook
jupyter notebook model.ipynb

# Project Structure
Subscription-Retention-Prediction/
│
├── app_manual_input.py
├── model.ipynb
├── requirements.txt
├── README.md
├── .gitignore

📌 Notes on Model Artifacts

Trained model files, encoders, and scalers are excluded from version control.

This follows ML engineering best practices, as model artifacts are reproducible and environment-dependent.

# Author

Prasin K M
Data Science | Machine Learning Projects
