#  Customer Churn Prediction for Subscription-Based Businesses

An **end-to-end Machine Learning project** that predicts customer churn for subscription-based online businesses using behavioral, transactional, and engagement data, and deploys predictions via **Streamlit web applications**.

---

##  Project Objective

To build a **production-ready churn prediction system** that identifies customers likely to cancel their subscription, enabling businesses to take **proactive retention actions**.

This project focuses on **model generalization, class imbalance handling, and business-relevant evaluation metrics** rather than accuracy alone.

---

##  Business Problem & Impact

Customer churn directly impacts revenue in subscription-based businesses (SaaS, OTT, EdTech, FinTech).

This solution helps organizations to:

* Identify **high-risk churn customers**
* Improve **customer retention rate**
* Reduce revenue leakage
* Optimize **marketing and support strategies**
* Enhance **Customer Lifetime Value (CLV)**

---

##  Dataset Overview

* **Source**: Kaggle
* **Type**: Subscription business customer dataset
* **Target Variable**: `churn` (Binary Classification)
* **Class Imbalance**: Churned customers are the minority class

### Feature Domains

* Demographics (Age, Gender, Country, City)
* Usage & Engagement (Logins, Session Time, Features Used)
* Billing & Payments (Monthly Fee, Total Revenue, Payment Failures)
* Customer Support (Tickets, Escalations, Resolution Time)
* Customer Satisfaction (CSAT, NPS, Survey Response)
* Marketing Engagement (Email Open Rate, Click Rate, Referrals)

---

##  Exploratory Data Analysis (EDA)

Performed EDA to understand customer behavior and churn patterns using:

* Distribution plots
* Bar charts & count plots
* Box plots (outlier analysis)
* Correlation heatmaps

EDA insights guided **feature selection and model choice**.

---

##  Machine Learning Pipeline

1. Data cleaning & preprocessing
2. Handling categorical variables using **OneHotEncoder**
3. Feature scaling using **StandardScaler**
4. Train–test split
5. Model training using multiple classifiers
6. Model evaluation using:

   * Precision
   * Recall
   * F1-score
   * Confusion Matrix
7. Overfitting analysis (train vs test performance)
8. Hyperparameter tuning
9. Model persistence using **pickle**

---

##  Models Implemented

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree Classifier
* Support Vector Classifier (SVC)
* Random Forest Classifier
* AdaBoost Classifier
* Gaussian Naive Bayes

---

##  Model Selection Strategy

The final model was selected based on:

* **Recall (Churn class)**
* **F1-score**
* Generalization ability
* Bias–variance tradeoff

###  Final Model: **Logistic Regression**

**Why Logistic Regression?**

* Better recall for churned customers
* Stable train–test performance
* Interpretable and business-friendly
* Lower risk of overfitting

---

##  Application Deployment (Streamlit)

### 1️ Manual Input Interface

* User manually enters customer features
* Real-time churn prediction
* Actionable retention recommendations

📄 File: `app_manual_input.py`

---

### 2️ Customer ID–Based Interface (Production Simulation)

* User inputs **Customer ID**
* Customer data auto-fetched from dataset
* Churn prediction + recommendations generated

📄 File: `app_customer_id_input.py`

✔ Simulates real-world enterprise ML systems

---

##  Recommendation Engine

A rule-based recommendation system provides **business-aligned retention strategies**, including:

* Engagement improvement suggestions
* Pricing & discount strategies
* Support escalation handling
* Marketing optimization
* Loyalty and referral programs

Recommendations are generated **independently of the ML prediction** to improve interpretability.

---

##  Tech Stack & Tools

* **Programming**: Python
* **Data Analysis**: Pandas, NumPy
* **Machine Learning**: Scikit-learn
* **Visualization**: Matplotlib, Seaborn
* **Deployment**: Streamlit
* **Model Persistence**: Pickle

---

##  How to Run the Project

### Clone Repository

```bash
git clone https://github.com/prasin-k-m/Subscription-Retention-Prediction.git
cd Subscription-Retention-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Applications

```bash
streamlit run app_manual_input.py
streamlit run app_customer_id_input.py
```

### Model Development Notebook

```bash
jupyter notebook model.ipynb
```

---

##  Project Structure

```
Subscription-Retention-Prediction/
│
├── app_manual_input.py
├── app_customer_id_input.py
├── model.ipynb
├── customer_churn_business_dataset.csv
├── requirements.txt
├── README.md
├── .gitignore
```

---

##  Model Artifacts Note

Trained models, encoders, and scalers are **excluded from version control** in line with ML engineering best practices.
Artifacts are reproducible via the notebook.

---

##  Future Enhancements

* SHAP-based explainability
* Cloud deployment (AWS / GCP)
* REST API integration
* Automated model retraining
* Business KPI dashboards

---

##  Author

**Prasin K M**
Data Science | Machine Learning | Predictive Analytics

