# ============================================================================
# STREAMLIT + KERAS: It's Just Python!
# ============================================================================
# Run with: streamlit run streamlit_diabetes_app.py
# ============================================================================

import streamlit as st
import numpy as np
import joblib
from tensorflow import keras


# ============================================================================
# THE KEY INSIGHT: Streamlit is just Python with UI widgets
# ============================================================================
#
#   Normal Python:              Streamlit:
#   ─────────────────           ─────────────────────────────
#   bmi = 25.0                  bmi = st.slider("BMI", 18, 40)
#   print(f"Risk: {risk}")      st.write(f"Risk: {risk}")
#
#   That's literally it. Everything else is normal Python/Keras code!
#
# ============================================================================

# ----------------------------------------------------------------------------
# STEP 1: Load model (same as any Python script!)
# ----------------------------------------------------------------------------
@st.cache_resource  # Cache so it doesn't reload on every interaction
def load_model():
    """Load model once and keep in memory"""
    model = keras.models.load_model('health_predictor_production.keras')
    scaler = joblib.load('health_scaler.pkl')
    return model, scaler


# Try to load, show error if files missing
try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_msg = str(e)

# ----------------------------------------------------------------------------
# STEP 2: Build the UI (Streamlit widgets)
# ----------------------------------------------------------------------------
st.title("🏥 Diabetes Risk Predictor")
st.write("Adjust the patient values and click Predict")

if not model_loaded:
    st.error(f"⚠️ Could not load model: {error_msg}")
    st.info("Run the training script first to create the model files!")
    st.stop()

# Create input widgets - these return the current values!
st.sidebar.header("Patient Data")

age = st.sidebar.slider(
    "Age (normalized)",
    min_value=-0.10,
    max_value=0.15,
    value=0.05,
    help="0.0 = ~48 years, +0.05 = ~58 years"
)

sex = st.sidebar.selectbox(
    "Sex",
    options=[0.05, -0.04],
    format_func=lambda x: "Male" if x > 0 else "Female"
)

bmi = st.sidebar.slider(
    "BMI (normalized)",
    min_value=-0.10,
    max_value=0.15,
    value=0.03,
    help="0.0 = BMI ~26, +0.05 = BMI ~30 (obese)"
)

bp = st.sidebar.slider(
    "Blood Pressure (normalized)",
    min_value=-0.10,
    max_value=0.15,
    value=0.02,
    help="0.0 = ~94 mmHg"
)

# Cholesterol panel
st.sidebar.subheader("Cholesterol Panel")

s1 = st.sidebar.slider("Total Cholesterol (s1)", -0.10, 0.15, 0.03)
s2 = st.sidebar.slider("LDL - Bad (s2)", -0.10, 0.15, 0.02)
s3 = st.sidebar.slider("HDL - Good (s3)", -0.10, 0.15, -0.02,
                       help="⚠️ LOWER = WORSE for HDL!")
s4 = st.sidebar.slider("TC/HDL Ratio (s4)", -0.10, 0.15, 0.03)
s5 = st.sidebar.slider("Triglycerides (s5)", -0.10, 0.15, 0.03)

# Blood sugar
s6 = st.sidebar.slider(
    "Blood Sugar (s6)",
    -0.10, 0.15, 0.04,
    help="0.0 = ~91 mg/dL, +0.05 = ~105 (pre-diabetic)"
)

# ----------------------------------------------------------------------------
# STEP 3: Make prediction (normal Keras code!)
# ----------------------------------------------------------------------------
if st.button("🔮 Predict Risk", type="primary"):

    # Package input (same as any script)
    patient_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])

    # Scale (same as any script)
    patient_scaled = scaler.transform(patient_data)

    # Predict (same as any script!)
    risk_probability = model.predict(patient_scaled, verbose=0)[0][0]

    # --------------------------------------------------------------------
    # STEP 4: Display results (Streamlit makes it pretty)
    # --------------------------------------------------------------------
    st.markdown("---")
    st.header("📊 Prediction Results")

    # Big number display
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Risk Probability",
            value=f"{risk_probability:.1%}"
        )

    with col2:
        if risk_probability < 0.3:
            st.success("🟢 LOW RISK")
        elif risk_probability < 0.6:
            st.warning("🟡 MODERATE RISK")
        else:
            st.error("🔴 HIGH RISK")

    # Progress bar visualization
    st.progress(float(risk_probability))

    # Clinical interpretation
    st.subheader("Clinical Interpretation")

    if risk_probability < 0.3:
        st.write("✅ Continue routine monitoring. Maintain healthy lifestyle.")
    elif risk_probability < 0.6:
        st.write("⚠️ Schedule follow-up in 3 months. Consider lifestyle modifications.")
    else:
        st.write("🚨 Immediate clinical consultation recommended. Further testing needed.")

    # Show the input values
    with st.expander("View Patient Data"):
        st.json({
            "age": float(age),
            "sex": "Male" if sex > 0 else "Female",
            "bmi": float(bmi),
            "bp": float(bp),
            "cholesterol_total": float(s1),
            "ldl": float(s2),
            "hdl": float(s3),
            "tc_hdl_ratio": float(s4),
            "triglycerides": float(s5),
            "glucose": float(s6)
        })

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown("---")
st.caption("⚠️ This is a demo model for educational purposes only. Not for clinical use.")

# ============================================================================
# THE POINT: Streamlit is just Python with pretty widgets
# ============================================================================
#
#   BEFORE (command line):
#   ─────────────────────────────────────────
#   bmi = float(input("Enter BMI: "))
#   risk = model.predict([[bmi]])
#   print(f"Risk: {risk}")
#
#   AFTER (Streamlit):
#   ─────────────────────────────────────────
#   bmi = st.slider("BMI", 18, 40)          # ← Pretty slider instead of input()
#   risk = model.predict([[bmi]])            # ← SAME!
#   st.write(f"Risk: {risk}")               # ← Pretty output instead of print()
#
#   Your Keras code stays EXACTLY the same!
#   Streamlit just replaces input() and print() with widgets.
#
# ============================================================================