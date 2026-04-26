#  NeuroDiagnostics-AD: Alzheimer’s Disease Prediction Pipeline

This repository hosts a comprehensive Machine Learning pipeline designed to support the early detection of Alzheimer's Disease. By combining high-performance ensemble methods with Explainable AI (XAI), this project provides a robust framework for clinical decision support.

---

## The "Clinical Integrity" Angle 
> **Author's Note:** Leveraging my background in **Cognitive Neuroscience**, I approached this clinical dataset not just as rows of data, but as dynamic patient profiles. In medical diagnostics, a model is only as good as its generalizability; for this reason, I prioritized **clinical integrity over pure metric-chasing**. By **validating** a Baseline LightGBM over a **fine-tuned** version, I ensured a model that avoids overfitting and remains grounded in medical reality. My focus was to bridge the gap between "black-box" algorithms and transparent, SHAP-driven insights that clinicians can actually trust.

---

##  Project Workflow

The project is structured into four specialized phases:

- **Data Analysis:** Identified demographic biases and severe class imbalance in dementia classification (Demented vs Non-Demented)

- **Model Benchmarking:** Evaluated ensemble models (Random Forest, XGBoost, CatBoost, LightGBM) using **AUROC, F1-score, and confusion matrix analysis**.

- **Model Validation:** Prioritized **baseline model over fine-tuned variant** to reduce overfitting and improve generalization on unseen patient profiles  

- **Explainability:** Applied **SHAP (Waterfall plots)** to provide patient-level interpretability and support clinical decision-making  

- **Output:**  Selected  **LightGBM** for its superior balance between sensitivity and clinical reliability. Achieving a **94.4% AUC-ROC** and a **93.7% F1-score**, the model effectively **minimized diagnostic errors**, ensuring **high predictive power and generalizability on unseen patient data**.  Built an interactive **dashboard** to **compare** individual **patients against population-level statistical benchmarks**.
  
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
