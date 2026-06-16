# Credit Score Classification: From Local Machine Learning to Cloud Deployment ☁️

This project was developed as part of my final Model Deployment project, where I built and deployed an end-to-end credit scoring system capable of classifying customers into **Poor**, **Standard**, and **Good** credit categories.

What started as a machine learning model running locally eventually evolved into a cloud-based application deployed on **AWS SageMaker** and served through a **Streamlit web application hosted on Amazon EC2**.

Beyond model performance, this project focused on understanding the challenges of bringing machine learning systems into production environments.

---

## ✨ Project Overview

### Business Problem

Financial institutions process large numbers of loan and credit applications every day. Evaluating customer creditworthiness manually can be time-consuming, inconsistent, and difficult to scale.

The challenge is determining:

* Which customers represent higher credit risk
* Which financial and behavioral factors influence credit score classifications
* How machine learning can support faster and more consistent credit assessment

### Objective

Develop a machine learning solution that can automatically classify customer credit scores and deploy the model as a real-time prediction service accessible through a web application.

---

## 📊 Dataset

The dataset contains customer financial and behavioral information, including:

* Income and salary information
* Credit utilization metrics
* Outstanding debt
* Credit history characteristics
* Payment behavior patterns
* Loan information
* Credit mix indicators

Target Classes:

* Good
* Standard
* Poor

---

## 🔍 Development Process

### 1. Data Preparation & Feature Engineering

Performed preprocessing using Scikit-Learn Pipelines to ensure reproducibility and consistent transformations during both training and inference.

Key preprocessing steps included:

* Missing value handling
* Numerical feature scaling
* Categorical feature encoding
* Feature selection and transformation

---

### 2. Model Development

Several machine learning approaches were evaluated before selecting the final model.

Models were compared using:

* Accuracy
* Precision
* Recall
* F1 Score
* Macro F1 Score

The final model utilized:

**LightGBM Classifier**

with hyperparameter optimization using:

**Optuna**

to maximize macro F1 performance across classes.

---

### 3. Local Machine Learning Pipeline

Built a reusable local training pipeline that supports:

* Dataset loading
* Preprocessing
* Hyperparameter tuning
* Model training
* Evaluation
* Artifact packaging

This pipeline enables retraining and experimentation with minimal manual intervention.

---

### 4. AWS Cloud Deployment

The trained model was packaged and deployed using:

* Amazon S3 (Model Storage)
* Amazon SageMaker (Real-Time Endpoint)
* Amazon EC2 (Streamlit Hosting)

Architecture:

Dataset

↓

Training Pipeline

↓

LightGBM Model

↓

model.tar.gz

↓

Amazon S3

↓

SageMaker Endpoint

↓

Streamlit Application

↓

End User

---


## 📈 Results

Final Model:

* Accuracy: 73%
* Macro F1 Score: 72%

The optimized LightGBM model provided the strongest balance across all target classes and was selected for deployment.

---

## ⚡ Challenges & Solutions

### SageMaker Deployment Failure Due to Dependency Mismatch

One of the most challenging parts of this project was not building the model itself—but making it work reliably in the cloud.

After deploying the trained model to SageMaker, the endpoint repeatedly failed during inference despite working perfectly in the training environment.

#### Root Cause

A version mismatch existed between:

* Local training environment
* SageMaker inference environment

The model was trained using a newer Scikit-Learn version while the SageMaker container was running a different version, causing serialization incompatibilities.

#### Solution

* Investigated CloudWatch logs
* Traced the issue to Scikit-Learn version differences
* Aligned package versions across environments
* Rebuilt the model artifact
* Redeployed the endpoint

The endpoint successfully returned predictions after the dependency issue was resolved.

This experience taught me that production deployment challenges often come from infrastructure and environment differences rather than model performance itself.

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM
* Optuna
* Joblib

### Deployment

* Streamlit
* AWS SageMaker
* AWS EC2
* AWS S3
* Boto3
* IAM
* CloudWatch

---

## 🌟 What Made This Project Special

This project was my first experience deploying a machine learning system on AWS.

Before this project, I mainly focused on building models and evaluating metrics locally. Through the deployment process, I learned that production machine learning involves much more than model training—it requires understanding cloud services, dependency management, infrastructure configuration, and system integration.

Watching a model move from a Jupyter notebook into a publicly accessible cloud application was one of the most rewarding parts of the experience.

---

## 🎯 Key Takeaway

A machine learning model is only valuable when people can actually use it.

Building the model was important, but learning how to deploy, troubleshoot, and operate it in a cloud environment provided the biggest growth opportunity throughout this project.

---

## 👩‍💻 Author

**Luna Alexa**

Data Science Student — BINUS University
