import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Neurodegenerative Bimodal Clinical Diagnostics Engine",
    layout="wide"
)

# --- USER CREDIT SIGNATURE LINE ---
st.sidebar.markdown("### Platform Signature")
st.sidebar.info("Designed and Developed by Amara Ranjit Kamble")
st.sidebar.markdown("---")

st.title("Neurodegenerative Bimodal Clinical Diagnostics Engine")
st.markdown("##### Powered by Gradient Boosted Decision Tree (XGBoost) Architectures")
st.markdown("---")
st.sidebar.header("Clinical Navigation Control")

# --- MASTER TRANSLATION MAP ---
parameter_dictionary = {
    'vcv000013401': 'Genetic Variation Pathology Risk Index',
    'total_time25': 'Total Clinical Progression Score',
    'air_time': 'Movement Interruption Duration (Air Time)',
    'disp_index1': 'Spatial Displacement Variability Index',
    'gmrt_in_air1': 'Geometric Mean Reaction Time (Air)',
    'gmrt_on_paper1': 'Task Execution Motor Response Speed (Active)',
    'max_x_extension1': 'Maximum Horizontal Extension Range',
    'max_y_extension1': 'Maximum Vertical Extension Range',
    'mean_acc_in_air1': 'Average Tremor Acceleration (Resting)',
    'mean_acc_on_paper1': 'Average Active Movement Acceleration',
    'mean_gmrt1': 'Unified Mean Response Coordinate Speed',
    'mean_jerk_in_air1': 'Resting Tremor Incoordination (Mean Jerk in Air)',
    'mean_jerk_on_paper1': 'Active Movement Shakiness (Mean Jerk on Paper)',
    'mean_speed_in_air1': 'Resting Limb Movement Velocity',
    'mean_speed_on_paper1': 'Active Writing/Drawing Velocity'
}

@st.cache_resource
def load_suite():
    try:
        return joblib.load('prion_clinical_suite(2).pkl')
    except:
        try:
            return joblib.load('prion_clinical_suite.pkl')
        except:
            st.error("Could not locate the model database asset file. Please ensure it is uploaded.")
            return None

suite = load_suite()

if suite:
    raw_keys = list(suite.keys())
    pnrp_key = [k for k in raw_keys if 'prnp' in k.lower() or 'variant' in k.lower()]
    motor_key = [k for k in raw_keys if 'data.csv' in k.lower()]
    
    available_modules = {}
    if pnrp_key:
        available_modules['PRNP Genetic Mutation Clinical Registry'] = pnrp_key
    if motor_key:
        available_modules['Motor Function & Clinical Severity Evaluation Matrix'] = motor_key
        
    selected_display = st.sidebar.selectbox("Choose Disease or Lab Analysis Type", list(available_modules.keys()))
    
    # FIXED: Extracting the actual first string element out of the matching dictionary list structure safely
    internal_file_key = available_modules[selected_display][0]
    
    model_data = suite[internal_file_key]
    features = model_data['features']
    target = model_data['target']
    mae = model_data['mae']
    
    # INTERACTIVE INFO EXPANDER
    with st.expander(f"Show Module Description for: {selected_display}", expanded=True):
        if "PRNP" in selected_display:
            st.write(
                "This diagnostic network evaluates verified human PRNP gene variations and genomic profiles. "
                "It computes multi-parameter non-linear risk projections to determine relative pathogenicity "
                "bounds mapped to structural structural mutations."
            )
        else:
            st.write(
                "This predictive module tracks physical motor control degradation, stability boundaries, and "
                "biomechanical coordination vectors. It evaluates fine-motor velocity variations and tremor acceleration profiles "
                "to determine overall functional progression markers."
            )
        
    st.markdown("---")
    friendly_target = parameter_dictionary.get(str(target).lower(), str(target).upper().replace('_', ' '))
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="AI Clinical Prediction Goal", value=friendly_target)
    col2.metric(label="Engine Validation Deviation (MAE)", value=f"{mae:.4f}")
    col3.metric(label="Computational Architecture", value="XGBoost Ensemble")
    
    st.markdown("### Clinical Parameter & Feature Settings")
    st.write("Modify the functional metrics below to evaluate real-time architectural estimations:")
    
    input_values = []
    feature_labels = []
    display_limit = min(12, len(features))
    grid_cols = st.columns(2)
    
    for idx, feat in enumerate(features[:display_limit]):
        col_slot = grid_cols[idx % 2]
        clean_key = str(feat).lower().strip()
        friendly_label = parameter_dictionary.get(clean_key, str(feat).replace('_', ' ').title())
        
        if len(friendly_label) > 60:
            friendly_label = friendly_label[:57] + "..."
            
        if "PRNP" in selected_display:
            choice = col_slot.selectbox(f"{friendly_label}", ["0 - Mutation Absent (Normal)", "1 - Mutation Present (Variant)"], index=0)
            val = 1.0 if "1 -" in choice else 0.0
        else:
            val = col_slot.number_input(f"{friendly_label}", value=0.0, step=0.01, format="%.4f")
            
        input_values.append(val)
        feature_labels.append(friendly_label)
        
    for feat in features[display_limit:]:
        input_values.append(0.0)
        
    st.markdown("---")
    
    if st.button("Compute Diagnostic Projection", type="primary"):
        input_array = np.array([input_values])
        prediction_raw = model_data['model'].predict(input_array)
        prediction_scalar = float(prediction_raw)
        
        st.success(f"### AI Predicted Outcome Value for [{friendly_target}]: **{prediction_scalar:.4f}**")
        st.info(f"Cross-Validation Guidance Window: +/- {mae:.4f} deviation interval bounds.")
        
        # --- CLINICAL INTERPRETATION LAYER ---
        st.markdown("### Clinical Interpretation of Results")
        if "PRNP" in selected_display:
            if prediction_scalar == 0.0:
                st.write(
                    "**Interpretation:** The AI engine has generated a baseline classification score. "
                    "This indicates that the current combination of mutation vectors matches profiles with "
                    "low localized structural disruption within the modeled genetic variants catalog."
                )
            else:
                st.write(
                    "**Interpretation:** The AI engine indicates a non-zero variant risk probability tracking index. "
                    "This implies structural deviations or alterations corresponding to known pathogenic genomic configurations."
                )
        else:
            if prediction_scalar > 35000:
                st.write(
                    "**Interpretation:** **Advanced Functional Impairment Profile.** The combination of high physical tremors "
                    "(Mean Jerk) and severe task velocity slowing places the computed score in the upper severity continuum. "
                    "This profile correlates heavily with advanced neurodegenerative motor degradation markers."
                )
            else:
                st.write(
                    "**Interpretation:** **Early-Stage / Baseline Functional Profile.** The biomechanical parameters "
                    "indicate relatively preserved fine-motor response coordinates or mild, localized resting tremor markers."
                )

        # --- MANDATORY MEDICAL DISCLAIMER BOX ---
        st.warning(
            "**IMPORTANT MEDICAL NOTICE & LEGAL DISCLAIMER:** "
            "This application is a computational biology modeling research tool and is provided strictly for educational "
            "and scientific simulation purposes. Results generated by this machine learning system are mathematical projections "
            "based on historical training data sets and are **NOT guaranteed to be accurate, complete, or reflective of real-world "
            "clinical conditions.** This tool does not provide medical diagnoses, treatment advice, or formal diagnostic evaluations. "
            "All users must refer to a qualified medical professional, neurologist, or physician for any health concerns or "
            "clinical decision-making guidance."
        )

        st.markdown("### Patient Benchmark Distribution Spectrum")
        chart_data = pd.DataFrame(
            [prediction_scalar, 16650.0, 35000.0],
            index=['Current Profile Score', 'Normal Cohort Baseline Variance', 'Advanced Severity Threshold'],
            columns=['Score Value']
        )
        st.bar_chart(chart_data)
        
        # --- GENERATE EXPORTABLE SUMMARY TEXT REPORT ---
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_text = f"""======================================================
NEURODEGENERATIVE BIMODAL CLINICAL INTEL PROJECTION REPORT
======================================================
Generated Timestamp: {timestamp}
Analysis Module Focus: {selected_display}
Platform Authorship: Designed and Developed by Amara Ranjit Kamble

RESULTS SUMMARY:
------------------------------------------------------
Target Variable Prediction Goal: {friendly_target}
Calculated AI Inference Value  : {prediction_scalar:.4f}
Cross-Validation Error Window  : +/- {mae:.4f} units

MEDICAL DISCLAIMER:
Projections are mathematical approximations from research registries and are not guaranteed to be clinically accurate. Refer to a medical professional.
"""
        for lbl, val in zip(feature_labels, input_values[:display_limit]):
            report_text += f"- {lbl}: {val}\n"
        report_text += "======================================================\n"
        
        st.download_button(
            label="Download Structured Clinical Summary Report",
            data=report_text,
            file_name=f"Clinical_AI_Diagnostic_Report.txt",
            mime="text/plain"
        )
else:

