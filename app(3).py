import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Neurodegenerative Bimodal Clinical Diagnostics Engine",
    layout="wide"
)

st.title("Neurodegenerative Bimodal Clinical Diagnostics Engine")
st.markdown("---")
st.sidebar.header("Clinical Navigation Control")

# --- MASTER TRANSLATION MAP ---
parameter_dictionary = {
    # Objectives / Targets
    'vcv000013401': 'Genetic Variation Pathology Risk Index',
    'total_time25': 'Total Clinical Progression Score',
    
    # Motor metrics (data.csv)
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
        available_modules['PRNP Genetic Mutation Clinical Registry'] = pnrp_key[0]
    if motor_key:
        available_modules['Motor Function & Clinical Severity Evaluation Matrix'] = motor_key[0]
        
    selected_display = st.sidebar.selectbox("Choose Disease or Lab Analysis Type", list(available_modules.keys()))
    internal_file_key = available_modules[selected_display]
    
    model_data = suite[internal_file_key]
    features = model_data['features']
    target = model_data['target']
    mae = model_data['mae']
    
    st.subheader(selected_display)
    if "PRNP" in selected_display:
        st.write(
            "This diagnostic network evaluates verified human PRNP gene variations and genomic profiles. "
            "It computes multi-parameter non-linear risk projections to determine relative pathogenicity "
            "bounds mapped to structural mutations."
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
        
    for feat in features[display_limit:]:
        input_values.append(0.0)
        
    st.markdown("---")
    
    if st.button("Compute Diagnostic Projection", type="primary"):
        input_array = np.array([input_values])
        prediction = model_data['model'].predict(input_array)
        
        st.success(f"### AI Predicted Outcome Value for [{friendly_target}]: **{prediction:.4f}**")
        st.info(f"Cross-Validation Guidance Window: +/- {mae:.4f} deviation interval bounds.")
else:
    st.warning("Failed to link model properties. Please check file integrity.")
