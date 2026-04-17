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

 [**Access the Alzheimer's Diagnosis Dashboard**](https://neurodiagnosticsclinicalsupportsystemforalzheimersprediction-4.streamlit.app/)

### Key Features:
* **Patient Profile vs. Average:** Interactive radar charts comparing user input with statistical population benchmarks.
* **SHAP Transparency:** Real-time generation of Waterfall plots to explain individual predictions.
* **Clinical Glossary:** Quick access to definitions of medical metrics like MMSE and Functional Assessment.

---
*Disclaimer: This tool is intended for research purposes and clinical support simulation. It does not replace professional medical judgment.*


## Tech Stack & Libraries

### **Core Data Science & Clinical Analysis**
* **Python:** Programming language and core logic.
* **Pandas & NumPy:** Data manipulation and matrix operations.
* **SciPy:** Advanced statistical testing and analysis.

### **ML & Explainable AI (XAI)**
* **Scikit-Learn:** Standardized preprocessing and model benchmarking.
* **LightGBM:** Champion gradient boosting model (Baseline version).
* **SHAP (Shapley Additive Explanations):** Model interpretability and feature impact analysis.

### **Visualization & Deployment**
* **Streamlit:** Interactive web dashboard for clinical support.
* **Plotly:** Dynamic radar charts for patient profiling.
* **Matplotlib:** Static visualizations for SHAP waterfall plots.
