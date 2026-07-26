📊 Telecom Customer Churn Prediction using Machine Learning

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn, based on their account and service details. The project covers the full pipeline — from raw data to a live, deployed web application.

🔗 Live Demo: telecom-churn-prediction-k5z3.onrender.com

📌 Project Overview

Customer churn — when a customer stops using a company's service — is a major concern for telecom providers. This project builds a machine learning model to predict churn risk in advance, so businesses can take proactive retention action.

The project follows a 5-stage pipeline:

Stage	Description
1️⃣ Data Understanding & EDA	Explore the raw dataset and identify key churn drivers
2️⃣ Data Preprocessing & Feature Engineering	Clean data, encode features, handle class imbalance
3️⃣ Model Development & Optimization	Train, compare, and tune ML models
4️⃣ Application Development	Build an interactive prediction web app with Streamlit
5️⃣ Deployment & Serving	Deploy the app live on the web (Render)

🗂️ Project Structure
telecom-churn/
├── data/                  
├── notebooks/
│   ├── 01_eda.ipynb             
│   ├── 02_preprocessing.ipynb    
│   └── 03_modeling.ipynb        
├── models/               
├── app/
│   └── app.py              
├── requirements.txt         
├── runtime.txt               
└── README.md

📊 Dataset
Source: Telco Customer Churn Dataset (Kaggle)
The dataset contains customer-level information including demographics, account details, subscribed services, billing information, and whether the customer churned.


Key features used:

Customer profile: gender, senior citizen status, partner, dependents
Account details: tenure, contract type, payment method
Services: internet service, online security, tech support
Billing: monthly charges, total charges

🔍 Stage 1: Exploratory Data Analysis
Analyzed churn distribution (~27% churn rate — an imbalanced dataset)
Identified key churn drivers:
Customers on month-to-month contracts churn significantly more than those on 1–2 year contracts
Low-tenure (newer) customers are at higher risk of churning
Higher monthly charges correlate with increased churn

🧹 Stage 2: Data Preprocessing & Feature Engineering
Removed non-predictive identifier columns
Converted and cleaned improperly typed numeric fields
Encoded categorical variables into numeric form
Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)

🤖 Stage 3: Model Development & Optimization
Trained and compared multiple models: Logistic Regression, Decision Tree, and Random Forest
Selected Random Forest as the best-performing model
Tuned hyperparameters using Optuna for optimal performance
Simplified the final feature set to the most important predictors for a cleaner user experience
Evaluation metrics used: Accuracy, Precision, Recall, F1 Score

💻 Stage 4: Application Development
Built an interactive web application using Streamlit, allowing users to:
Input customer details through a simple form
Get an instant churn prediction with probability score
View results in a clear, visual format

☁️ Stage 5: Deployment
<br>
The application is deployed and publicly accessible via Render, connected directly to this GitHub repository for continuous deployment — any update pushed to the main branch automatically redeploys the live app.

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Programming Language** | Python |
| **Data Analysis** | pandas, numpy |
| **Data Visualization** | matplotlib, seaborn |
| **Machine Learning** | scikit-learn |
| **Imbalanced Data Handling** | imbalanced-learn (SMOTE) |
| **Hyperparameter Tuning** | Optuna |
| **Web Framework** | Streamlit |
| **Deployment** | Render |
| **Version Control** | Git & GitHub |
