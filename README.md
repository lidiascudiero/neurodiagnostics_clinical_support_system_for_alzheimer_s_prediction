#  NeuroDiagnostics-AD: Alzheimer’s Disease Prediction Pipeline

This repository hosts a comprehensive Machine Learning pipeline designed to support the early detection of Alzheimer's Disease. By combining high-performance ensemble methods with Explainable AI (XAI), this project provides a robust framework for clinical decision support.

---

## The "Clinical Integrity" Angle 
> **Author's Note:** Leveraging my background in **Cognitive Neuroscience**, I approached this clinical dataset not just as rows of data, but as dynamic patient profiles. In medical diagnostics, a model is only as good as its generalizability; for this reason, I prioritized **clinical integrity over pure metric-chasing**. By validating a Baseline LightGBM over a fine-tuned version, I ensured a model that avoids overfitting and remains grounded in medical reality. My focus was to bridge the gap between "black-box" algorithms and transparent, SHAP-driven insights that clinicians can actually trust.

---

##  Project Workflow

The project is structured into four specialized phases:

1.  **Exploratory Data Analysis (EDA):** Identifying demographic biases and addressing the significant class imbalance (Demented vs. Non-Demented).
2.  **Model Benchmarking:** Comparative analysis of ensemble architectures (Random Forest, XGBoost, CatBoost). **LightGBM** emerged as the champion for its balance of precision and clinical utility.
3.  **Validation & Interpretation:** A critical phase where the **Baseline model** was selected over fine-tuned versions to ensure better generalization. Implementation of **SHAP Waterfall plots** for local interpretability.
4.  **Clinical Dashboard:** Integration of statistical benchmarks (means/modes) to allow real-time comparison of patient profiles against the training population.

## Interactive Live Demo

Experience the diagnostic pipeline through the dedicated Streamlit application. This dashboard allows for real-time risk assessment and feature impact visualization.

 [**Access the Alzheimer's Diagnosis Dashboard**](URL)

### Key Features:
* **Patient Profile vs. Average:** Interactive radar charts comparing user input with statistical population benchmarks.
* **SHAP Transparency:** Real-time generation of Waterfall plots to explain individual predictions.
* **Clinical Glossary:** Quick access to definitions of medical metrics like MMSE and Functional Assessment.

---
*Disclaimer: This tool is intended for research purposes and clinical support simulation. It does not replace professional medical judgment.*


##  Tech Stack & Libraries

### **Core Data Science & Neuro-Signal Processing**
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)

### **ML & Explainable AI (XAI)**
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-3E8E41?style=for-the-badge&logo=python&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-black?style=for-the-badge&logo=python&logoColor=white)

### **Visualization & Deployment**
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=Plotly&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

---
