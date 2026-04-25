import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np 

# 1. Page Configuration
st.set_page_config(page_title="Alzheimer's Diagnosis Support", layout="wide")

# 2. Load the Baseline Model
@st.cache_resource
def load_model():
    return joblib.load("best_model_lightgbm.joblib")

pipeline = load_model()

def plot_radar(patient_data, defaults):
    # Added DietQuality to match clinical profile
    categories = ['MMSE', 'Functional Assessment', 'ADL', 'Physical Activity', 'Sleep Quality', 'Diet Quality']
    
    # Normalization (MMSE /30, others /10)
    patient_val = [
        patient_data['MMSE']/30, 
        patient_data['FunctionalAssessment']/10, 
        patient_data['ADL']/10, 
        patient_data['PhysicalActivity']/10, 
        patient_data['SleepQuality']/10,
        patient_data['DietQuality']/10
    ]
    
    avg_val = [
        defaults['MMSE']/30, 
        defaults['FunctionalAssessment']/10, 
        defaults['ADL']/10, 
        defaults['PhysicalActivity']/10, 
        defaults['SleepQuality']/10,
        defaults['DietQuality']/10
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=patient_val, theta=categories, fill='toself', name='Current Patient', line_color='teal'))
    fig.add_trace(go.Scatterpolar(r=avg_val, theta=categories, fill='toself', name='Average Patient', line_color='gray'))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, 
        height=450,
        margin=dict(l=50, r=50, t=20, b=20)
    )
    return fig

# 3. App Header
st.title("Alzheimer's Disease Prediction Dashboard")
st.markdown("""
This tool uses a **Baseline LightGBM** model to estimate the probability of Alzheimer's. 
The model was selected for its high generalization capability and robustness.
""")
st.divider()

# 4. Calculated Default Values
defaults = {
    'ADL': 4.9545, 
    'Age': 74.9495, 
    'AlcoholConsumption': 10.0606, 
    'BMI': 27.6959,
    'BehavioralProblems': 0, 
    'CardiovascularDisease': 0, 
    'CholesterolHDL': 59.1440,
    'CholesterolLDL': 124.1461, 
    'CholesterolTotal': 225.0974, 
    'CholesterolTriglycerides': 229.2381,
    'Confusion': 0, 
    'Depression': 0, 
    'Diabetes': 0, 
    'DiastolicBP': 90.2684,
    'DietQuality': 4.9961, 
    'DifficultyCompletingTasks': 0, 
    'Disorientation': 0,
    'EducationLevel': 1, 
    'Ethnicity': 0, 
    'FamilyHistoryAlzheimers': 0, 
    'Forgetfulness': 0,
    'FunctionalAssessment': 5.0797, 
    'Gender': 1, 
    'HeadInjury': 0, 
    'Hypertension': 0,
    'MMSE': 14.8854, 
    'MemoryComplaints': 0, 
    'PersonalityChanges': 0,
    'PhysicalActivity': 4.9278, 
    'SleepQuality': 7.0576, 
    'Smoking': 0, 
    'SystolicBP': 133.8339,
    'DoctorInCharge': 'Unknown' 
}

# 5. User Interface - Sidebar
st.sidebar.header("Primary Clinical Factors")
st.sidebar.info("These variables have the highest impact on the prediction.")

# High Impact Features (SHAP)
mmse = st.sidebar.slider("MMSE Score (Cognitive)", 0.0, 30.0, float(defaults['MMSE']))
func_ast = st.sidebar.slider("Functional Assessment", 0.0, 10.0, float(defaults['FunctionalAssessment']))
adl = st.sidebar.slider("ADL (Activities of Daily Living)", 0.0, 10.0, float(defaults['ADL']))
age = st.sidebar.number_input("Patient Age", 60, 90, int(defaults['Age']))

# Categorical Inputs
memory = st.sidebar.selectbox("Memory Complaints", [0, 1], index=int(defaults['MemoryComplaints']), format_func=lambda x: "Yes" if x==1 else "No")
behavior = st.sidebar.selectbox("Behavioral Problems", [0, 1], index=int(defaults['BehavioralProblems']), format_func=lambda x: "Yes" if x==1 else "No")

st.sidebar.divider()

# Lifestyle Factors (For Radar Plot interaction)
with st.sidebar.expander("Lifestyle and Prevention"):
    st.write("These factors influence the Radar Plot and long-term risk profile.")
    phys_act = st.slider("Physical Activity (hrs/week)", 0.0, 10.0, float(defaults['PhysicalActivity']))
    sleep_qual = st.slider("Sleep Quality (Rating 0-10)", 0.0, 10.0, float(defaults['SleepQuality']))
    diet_qual = st.slider("Diet Quality (Rating 0-10)", 0.0, 10.0, float(defaults['DietQuality']))

# 6. Building the Input DataFrame
patient_data = defaults.copy()
patient_data.update({
    'MMSE': mmse,
    'FunctionalAssessment': func_ast,
    'ADL': adl,
    'Age': age,
    'MemoryComplaints': memory,
    'BehavioralProblems': behavior,
    'PhysicalActivity': phys_act,
    'SleepQuality': sleep_qual,
    'DietQuality': diet_qual
})

input_df = pd.DataFrame([patient_data])

# 7. Prediction Display
st.subheader("Diagnostic Analysis")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Generate Prediction", type="primary"):
        probability = pipeline.predict_proba(input_df)[0][1]
        prediction = pipeline.predict(input_df)[0]

        if prediction == 1:
            st.error("### POSITIVE: High Risk Detected")
        else:
            st.success("### NEGATIVE: Low Risk Detected")
        
        st.metric("Probability of Alzheimer's", f"{probability:.1%}")
        st.progress(probability)

with col2:
    st.write("**Prediction Explanation (SHAP):**")
    
    model_step = pipeline.named_steps['model']
    preprocessor_step = pipeline.named_steps['preprocessor']
    
    input_transformed = preprocessor_step.transform(input_df)
    feature_names = preprocessor_step.get_feature_names_out()
    input_transformed_df = pd.DataFrame(input_transformed, columns=feature_names)
    
    explainer = shap.TreeExplainer(model_step)
    shap_values = explainer.shap_values(input_transformed_df)
    
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap.Explanation(
        values=sv[0], 
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        data=input_transformed_df.iloc[0],
        feature_names=feature_names
    ), max_display=10, show=False)
    
    plt.title("Impact of Features on this Prediction")
    st.pyplot(fig_shap)

st.divider()
c_radar, c_info = st.columns([1, 1])

with c_radar:
    st.subheader("Patient Profile vs Average")
    st.plotly_chart(plot_radar(patient_data, defaults), use_container_width=True)

with c_info:
    st.subheader("Clinical Metrics Glossary")
    with st.expander("See definitions of scores"):
        st.markdown("""
        - **MMSE:** Cognitive test (0-30). Lower scores indicate higher impairment.
        - **ADL:** Activities of Daily Living. Measures independence in daily tasks.
        - **Functional Assessment:** Evaluation of lifestyle management.
        - **Lifestyle Factors:** Physical Activity, Sleep, and Diet are modifiable risks.
        - **SHAP:** Red bars increase predicted risk; blue bars decrease it.
        """)

st.caption("Baseline Model with XAI Integration - This tool is for portfolio purposes and should not be used for real clinical decisions.")
