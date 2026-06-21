import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Possum Age Predictor",
    page_icon="🐨",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2d6a4f;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #d8f3dc;
        border-left: 5px solid #2d6a4f;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
    }
    .result-age {
        font-size: 3rem;
        font-weight: 700;
        color: #1b4332;
    }
    .result-label {
        font-size: 1rem;
        color: #40916c;
        font-weight: 500;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #343a40;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stSlider > div > div > div > div {
        background-color: #2d6a4f;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = "knn_possum_pipeline.pkl"
    if not os.path.exists(model_path):
        st.error("Model file `knn_possum_pipeline.pkl` not found. Please ensure it is in the same directory as app.py.")
        st.stop()
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

st.markdown('<div class="main-header">🐨 Possum Age Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Estimate a Mountain Brushtail Possum\'s age from its body measurements using K-Nearest Neighbors regression.</div>', unsafe_allow_html=True)

st.divider()

col_inputs, col_result = st.columns([1.2, 1], gap="large")

with col_inputs:
    st.markdown('<div class="section-title">Possum Characteristics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        pop = st.selectbox("Population", options=["Vic", "other"], help="Vic = Victoria, other = remaining Australian regions")
    with c2:
        sex = st.selectbox("Sex", options=["m", "f"], format_func=lambda x: "Male" if x == "m" else "Female")

    st.markdown("**Head & Skull**")
    c1, c2 = st.columns(2)
    with c1:
        hdlngth = st.slider("Head length (mm)", min_value=82.5, max_value=103.1, value=92.5, step=0.1)
    with c2:
        skullw = st.slider("Skull width (mm)", min_value=50.0, max_value=68.6, value=56.9, step=0.1)

    st.markdown("**Body Dimensions**")
    c1, c2 = st.columns(2)
    with c1:
        totlngth = st.slider("Total length (cm)", min_value=75.0, max_value=96.5, value=87.0, step=0.5)
    with c2:
        taill = st.slider("Tail length (cm)", min_value=32.0, max_value=43.0, value=37.0, step=0.5)

    st.markdown("**Extremities**")
    c1, c2, c3 = st.columns(3)
    with c1:
        footlgth = st.slider("Foot length (mm)", min_value=60.0, max_value=76.0, value=68.0, step=0.5)
    with c2:
        earconch = st.slider("Ear conch (mm)", min_value=38.0, max_value=58.0, value=48.5, step=0.5)
    with c3:
        eye = st.slider("Eye diameter (mm)", min_value=12.0, max_value=18.0, value=14.9, step=0.1)

    st.markdown("**Girth**")
    c1, c2 = st.columns(2)
    with c1:
        chest = st.slider("Chest (cm)", min_value=22.0, max_value=38.0, value=28.0, step=0.5)
    with c2:
        belly = st.slider("Belly (cm)", min_value=22.0, max_value=44.0, value=31.0, step=0.5)

    predict_btn = st.button("Predict Age", type="primary", use_container_width=True)

with col_result:
    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            "Pop":      pop,
            "sex":      sex,
            "hdlngth":  hdlngth,
            "skullw":   skullw,
            "totlngth": totlngth,
            "taill":    taill,
            "footlgth": footlgth,
            "earconch": earconch,
            "eye":      eye,
            "chest":    chest,
            "belly":    belly,
        }])

        predicted_age = model.predict(input_data)[0]
        predicted_age = max(1.0, predicted_age)

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Age</div>
            <div class="result-age">{predicted_age:.1f} <span style="font-size:1.5rem">years</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        age_int = int(round(predicted_age))
        if age_int <= 2:
            stage = "🐣 Young juvenile"
            note  = "Very young possum — still developing key physical traits."
        elif age_int <= 4:
            stage = "🌿 Sub-adult"
            note  = "Most common age range in the dataset; model is most reliable here."
        elif age_int <= 6:
            stage = "🌳 Adult"
            note  = "Fully grown adult possum with stable body proportions."
        else:
            stage = "🏔️ Mature adult"
            note  = "Older possums are rare in the dataset; estimate carries higher uncertainty."

        st.info(f"**{stage}**  \n{note}")

        with st.expander("View input summary"):
            summary = input_data.copy()
            summary.columns = ["Population", "Sex", "Head (mm)", "Skull (mm)",
                                "Total length (cm)", "Tail (cm)", "Foot (mm)",
                                "Ear conch (mm)", "Eye (mm)", "Chest (cm)", "Belly (cm)"]
            st.dataframe(summary, use_container_width=True)

    else:
        st.markdown("""
        <div style="padding: 2rem; text-align: center; color: #6c757d; border: 2px dashed #dee2e6; border-radius: 8px; margin-top: 1rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem">🐨</div>
            <div style="font-weight: 600; margin-bottom: 0.25rem;">Ready to predict</div>
            <div style="font-size: 0.875rem;">Adjust the measurements on the left and click <strong>Predict Age</strong>.</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">Model Info</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Algorithm", "KNN Regressor")
    with m2:
        st.metric("Test R²", "0.29")
    with m3:
        st.metric("Test MAE", "~1.49 yrs")

    st.caption("Model trained on 102 possum records. Hyperparameters tuned via GridSearchCV with 5-fold cross-validation. Preprocessing (imputation, scaling, encoding) is bundled inside the pipeline.")

st.divider()
st.markdown(
    "<div style='text-align:center; color:#adb5bd; font-size:0.8rem;'>"
    "Built by <strong>Noureldeen Bassem</strong> · KNN Regression · Possum Dataset"
    "</div>",
    unsafe_allow_html=True
)
