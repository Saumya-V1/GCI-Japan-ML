# Customer Churn Prediction & Retention Strategy

## Overview

This project develops a machine learning solution for predicting customer churn in the telecommunications industry. By analyzing customer demographics, service usage patterns, billing information, and behavioral indicators, the model identifies subscribers at risk of leaving and supports proactive retention campaigns.

The project combines predictive analytics with business recommendations, enabling telecom operators to reduce churn, improve customer lifetime value, and increase revenue retention.

---

## Business Problem

Customer churn is one of the most significant challenges for telecom companies. Acquiring a new customer typically costs several times more than retaining an existing one. The goal of this project is to identify customers likely to churn within the near future and enable targeted intervention before revenue is lost.

### Key Objectives

* Predict customer churn probability
* Identify behavioral signals preceding churn
* Segment customers into risk categories
* Estimate business impact and ROI of retention campaigns
* Provide interpretable model insights for decision-makers

---

## Dataset

The project uses two datasets:

### Client.csv

Contains demographic and device-related information.

### Record.csv

Contains customer usage, billing, revenue, and service behavior metrics.

Both datasets are merged using:

```python
Customer_ID
```

The target variable is:

```python
churn
```

where:

* 1 = Customer churned
* 0 = Customer retained

---

## Feature Engineering

Several business-driven features were engineered to improve predictive performance:

### Usage & Quality Metrics

* Call Quality
* Completion Rate
* Overage Ratio
* Data Overage Percentage
* Revenue per Minute of Usage

### Trend Features

* Revenue Trend
* Usage Trend

### Customer Behavior Metrics

* Customer Care Calls per Usage
* Roaming Intensity
* Credit Card Usage Ratio
* Device Value-to-Age Ratio

### Service Reliability Metrics

* Unanswered Call Rate
* Blocked Call Rate

Outliers are clipped using the 1st and 99th percentiles to improve robustness.

---

## Data Preprocessing

### Numerical Features

* Median Imputation
* Standard Scaling

### Categorical Features

* Most Frequent Imputation
* One-Hot Encoding

### Missing Data Handling

Features with more than 25% missing values are removed to reduce noise.

Examples:

* numbcars
* dwllsize
* HHstatin
* ownrent
* dwlltype

---

## Machine Learning Model

### Algorithm

XGBoost Classifier

### Model Configuration

* 500 Estimators
* Learning Rate: 0.03
* Max Depth: 5
* Subsample: 0.8
* Column Sample Rate: 0.7
* Early Stopping Enabled
* Class Imbalance Handling via Scale Pos Weight

### Validation Strategy

* Stratified Train/Test Split
* Early Stopping Validation Set
* 5-Fold Stratified Cross Validation

---

## Model Explainability

The project uses SHAP (SHapley Additive exPlanations) to:

* Understand feature importance
* Explain individual predictions
* Increase stakeholder trust
* Support business decision-making

Visual outputs include:

* SHAP Summary Plot
* Feature Importance Analysis

---

## Performance

### Evaluation Metrics

* ROC-AUC
* Precision
* Recall
* Accuracy
* Cross-Validation ROC-AUC

Results reported in the accompanying business proposal:

* Test ROC-AUC ≈ 0.70
* Cross-Validation ROC-AUC ≈ 0.703
* Accuracy ≈ 64%

---

## Risk Segmentation

Customers are grouped into three categories:

| Risk Tier | Churn Probability |
| --------- | ----------------- |
| Low       | < 0.30            |
| Medium    | 0.30 – 0.60       |
| High      | > 0.60            |

This segmentation enables targeted retention campaigns and resource allocation.

---

## Business Recommendations

### High-Risk Customers

* Loyalty team outreach
* Personalized retention offers
* Device upgrade assessments

### Medium-Risk Customers

* Promotional discounts
* Plan optimization recommendations
* Usage notifications

### Rising-Risk Customers

* Usage summaries
* Plan-fit recommendations
* Early engagement campaigns

---

## Project Structure

```text
.
├── model.py
├── Client.csv
├── Record.csv
├── README.md
└── requirements.txt
```

---

## Installation

```bash
pip install pandas numpy matplotlib shap scikit-learn xgboost
```

---

## Usage

```bash
python model.py
```

The script will:

1. Load and merge customer datasets
2. Engineer predictive features
3. Train an XGBoost churn model
4. Evaluate performance
5. Generate visualizations
6. Produce SHAP explanations
7. Create business-focused churn risk reports

---

## Business Impact

The accompanying business proposal estimates that a targeted retention program based on model predictions could:

* Reduce churn rates
* Improve customer retention
* Increase revenue preservation
* Deliver positive ROI through proactive intervention strategies

---

## Future Improvements

* Hyperparameter optimization using Optuna
* Automated model retraining pipeline
* Real-time churn scoring API
* Ensemble learning approaches
* Customer lifetime value integration
* Deployment as a production microservice

---

## Author

Saumya Vijayvargiya

Customer Churn Prediction using XGBoost, Feature Engineering, Explainable AI (SHAP), and Business-Driven Retention Analytics.
