# 🎗️ Breast Cancer Prediction System

## Overview

The Breast Cancer Prediction System is an end-to-end Machine Learning healthcare application developed using Python, XGBoost, and Streamlit. The application analyzes breast tumor characteristics and predicts whether a tumor is classified as Benign (Non-Cancerous) or Malignant (Cancerous).

The project combines Machine Learning, Data Analysis, Data Visualization, Model Deployment, and Interactive User Interface development to provide a complete healthcare prediction solution.

---

## Key Features

### Authentication & User Management

* User Login System
* Session Management
* Logout Functionality

### Machine Learning Prediction

* Breast Cancer Classification
* Benign vs Malignant Prediction
* Confidence Score Calculation
* Risk Assessment System
* Probability Analysis

### Data Visualization

* Probability Charts
* Tumor Feature Comparison
* Radar Chart Visualization
* Risk Meter Visualization

### Reporting & Storage

* Prediction History Tracking
* CSV Report Download
* PDF Report Generation
* Timestamp Recording
* Patient Information Storage

### User Assistance

* Feature Guide
* Interactive Healthcare Chatbot
* Risk Recommendations
* Clinical Follow-Up Suggestions

### User Interface

* Professional Streamlit Dashboard
* Responsive Layout
* Clean Navigation System
* Healthcare-Focused Design

---

## Technologies Used

### Programming Language

* Python

### Machine Learning

* XGBoost Classifier
* Scikit-Learn

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Streamlit Charts

### Web Application Development

* Streamlit
* HTML
* CSS

### Report Generation

* ReportLab

### Model Serialization

* Pickle

---

## Dataset

### Dataset Name

Breast Cancer Wisconsin Diagnostic Dataset

### Domain

Healthcare / Medical Diagnosis

### Dataset Characteristics

* Real-world healthcare dataset
* 569 patient records
* 30 tumor measurement features
* Binary classification target

### Target Classes

* Benign (Non-Cancerous)
* Malignant (Cancerous)

---

## Machine Learning Model

### Algorithm

XGBoost Classifier

### Model Type

Supervised Machine Learning

### Classification Type

Binary Classification

### Evaluation Techniques

* Train-Test Split
* Confusion Matrix
* Classification Report
* Cross Validation
* Accuracy Measurement

---

## Project Workflow

### Phase 1: Data Collection

* Load Breast Cancer Dataset
* Explore Dataset Structure

### Phase 2: Data Cleaning

* Handle Unnecessary Columns
* Verify Missing Values
* Validate Data Quality

### Phase 3: Exploratory Data Analysis

* Dataset Visualization
* Feature Understanding
* Correlation Analysis

### Phase 4: Model Development

* Train-Test Split
* Model Training using XGBoost
* Model Evaluation

### Phase 5: Model Persistence

* Save Model using Pickle
* Load Model for Prediction

### Phase 6: Web Application Development

* Streamlit Interface
* User Authentication
* Prediction Dashboard

### Phase 7: Additional Features

* Chatbot Integration
* Prediction History
* PDF Report Generation
* CSV Export
* Feature Guide

### Phase 8: Deployment

* GitHub Repository
* Streamlit Cloud Deployment

---

## How to Run

### Install Dependencies

pip install -r requirements.txt

### Run Application

streamlit run app.py

---

## Project Structure

breast_cancer_detection/

├── app.py

├── xgboost_breast_cancer_model.pkl

├── breast_cancer_dataframe.csv

├── cleaned_breast_cancer_dataset.csv

├── prediction_history.csv

├── requirements.txt

├── README.md

├── screenshots/

│ ├── login_page.png

│ ├── prediction_page.png

│ ├── result_page.png

│ ├── history_page.png

│ └── chatbot_page.png

└── breast_cancer_model_training.ipynb

---

## Future Enhancements

* Multi-Model Comparison
* Cloud Database Integration
* Doctor Dashboard
* Email Report Delivery
* Mobile Application Version
* Real-Time Healthcare Analytics

---

