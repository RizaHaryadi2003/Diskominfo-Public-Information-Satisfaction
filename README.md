# 📊 Public Information Satisfaction Analysis (Diskominfo)

*A Data-Driven Machine Learning Framework for Evaluating Community Satisfaction in West Kalimantan.*

---

## 🧠 Overview

This project presents a comprehensive machine learning approach to evaluate and understand public satisfaction toward information services provided by the Communication and Informatics Office (**Diskominfo**).

In the context of digital governance, traditional survey analysis often stops at descriptive statistics. This research goes further by uncovering **hidden behavioral patterns**, identifying **key dissatisfaction drivers**, and transforming raw survey data into **actionable policy insights**.

By combining **Unsupervised Learning** and **Explainable AI (XAI)**, this system not only segments the population based on real experiences but also explains *why* those segments exist.

---

## 🎯 Objectives

The primary goals of this research are:

- To segment public satisfaction levels using data-driven clustering
- To identify the most influential factors affecting satisfaction
- To provide interpretable insights for policy and system improvement
- To support data-backed decision making in public service modernization

---

## 🔄 End-to-End Data Science Pipeline

This project is built on a structured 10-stage pipeline designed to ensure robustness, interpretability, and reliability:

### 1. Data Acquisition  
Collection and integration of raw survey data (Q1–Q15) from public information service platforms.

### 2. Exploratory Data Analysis (EDA)  
Understanding data distributions, detecting anomalies, and analyzing inter-feature relationships.

### 3. Data Preprocessing  
Cleaning the dataset by handling missing values, removing outliers, and excluding irrelevant metadata.

### 4. Feature Engineering  
Selecting and refining features that directly influence the Public Information Satisfaction Index.

### 5. Data Standardization  
Applying `StandardScaler` to normalize feature values and prevent bias in distance-based algorithms.

### 6. Optimal Cluster Determination  
Using:
- **Elbow Method** → to evaluate inertia (within-cluster variance)  
- **Silhouette Score** → to measure cluster separation quality  

This ensures the chosen number of clusters is mathematically justified.

### 7. Unsupervised Clustering (K-Means)  
Segmenting the population into distinct satisfaction groups (e.g., High, Medium, Low satisfaction).

### 8. Model Validation (Random Forest)  
Training a supervised model to validate cluster consistency and ensure predictability.

### 9. Explainable AI (SHAP)  
Using SHAP values to:
- Interpret model decisions  
- Identify feature importance  
- Explain satisfaction drivers at both global and individual levels  

### 10. Insight Generation & Reporting  
Transforming model outputs into strategic recommendations for improving systems such as **SIKEDIP**.

---

## 🛠️ Tech Stack

| Category            | Tools & Libraries |
|--------------------|------------------|
| Language           | Python 3.x       |
| Data Processing    | Pandas, NumPy    |
| Machine Learning   | Scikit-learn     |
| Visualization      | Matplotlib, Seaborn |
| Explainability     | SHAP             |
| Environment        | Jupyter Notebook / Google Colab |

---

## 📈 Key Findings

The analysis reveals several critical insights:

- **3 Distinct Clusters Identified**  
  The population naturally segments into three levels of satisfaction.

- **Primary Driver — Digital Accessibility (Q10)**  
  Ease of access to digital systems (e.g., SIKEDIP) is the strongest predictor of satisfaction.

- **Secondary Driver — Staff Professionalism (Q15)**  
  Responsiveness, ethics, and service quality significantly influence public perception.

These findings highlight that **technology usability + human interaction** jointly define service satisfaction.

---

## 💡 Strategic Implications

- Improving system accessibility will yield the highest impact on satisfaction
- Training staff in communication and responsiveness is equally critical
- Data-driven segmentation enables targeted policy interventions

---

## ⚙️ Installation & Usage



```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap
