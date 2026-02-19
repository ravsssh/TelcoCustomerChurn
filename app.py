"""
Telco Customer Churn — Streamlit App (Home Page)

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📡 Telco Customer Churn Predictor")
st.markdown(
    "A **full ML lifecycle** application — train, evaluate, and deploy a "
    "neural network for customer churn prediction."
)
st.divider()

# ---------------------------------------------------------------------------
# Artifact / model status banner
# ---------------------------------------------------------------------------
required_files = ["churn_model.keras", "scaler.pkl", "encoder.pkl", "config.json"]
model_ready = all((ARTIFACTS_DIR / f).exists() for f in required_files)

if model_ready:
    config = json.loads(CONFIG_PATH.read_text())
    trained_at = config.get("trained_at", "unknown")
    st.success(f"Model is ready — last trained at **{trained_at}**")
else:
    st.warning(
        "No trained model found. Go to **Train Model** in the sidebar to train the model first."
    )

st.divider()

# ---------------------------------------------------------------------------
# KPI cards (only shown when model exists)
# ---------------------------------------------------------------------------
if model_ready:
    st.subheader("Model Performance at a Glance")
    m = config["metrics"]
    t = config["threshold"]
    cm = config["confusion_matrix"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Recall", f"{m['recall']:.1%}", help="Proportion of actual churners caught")
    col2.metric("Precision", f"{m['precision']:.1%}", help="Proportion of flagged customers who actually churn")
    col3.metric("ROC-AUC", f"{m['auc_roc']:.3f}", help="Ranking quality of the model")
    col4.metric("F1-Score", f"{m['f1']:.3f}")
    col5.metric("Threshold", f"{t['optimal']:.2f}", help="Classification threshold (not the default 0.5)")

    st.divider()

    # Quick confusion matrix summary
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(
            f"At threshold **{t['optimal']}**, the model catches "
            f"**{cm['tp']}** out of **{cm['tp'] + cm['fn']}** actual churners "
            f"({cm['tp'] / (cm['tp'] + cm['fn']):.1%} recall), "
            f"with **{cm['fp']}** false alarms."
        )
    with col_b:
        tr = config.get("training", {})
        st.info(
            f"Trained on **{tr.get('train_size', '?')}** customers "
            f"({tr.get('epochs_trained', '?')} epochs). "
            f"Test set: **{tr.get('test_size', '?')}** customers."
        )

    st.divider()

# ---------------------------------------------------------------------------
# Navigation guide
# ---------------------------------------------------------------------------
st.subheader("Navigation")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 1️⃣ Train Model")
    st.markdown(
        "Trigger a full training run — fetches fresh data from IBM/GitHub, "
        "runs preprocessing, SMOTE-Tomek, trains the neural network, "
        "and saves all artifacts."
    )

with col2:
    st.markdown("### 2️⃣ Model Performance")
    st.markdown(
        "Explore the confusion matrix, ROC curve, PR curve, feature importance, "
        "training history, and an interactive business impact calculator."
    )

with col3:
    st.markdown("### 3️⃣ Single Prediction")
    st.markdown(
        "Enter a customer's details via form and get their churn probability, "
        "risk segment badge, and personalised retention recommendations."
    )

with col4:
    st.markdown("### 4️⃣ Batch Prediction")
    st.markdown(
        "Upload a CSV of customer records and download a results file with "
        "churn probabilities, predictions, and risk segments for every row."
    )

st.divider()

# ---------------------------------------------------------------------------
# Dataset and model info
# ---------------------------------------------------------------------------
with st.expander("About this project"):
    st.markdown("""
**Dataset**: IBM Telco Customer Churn — 7,043 customers × 21 features.

**Model**: Keras Sequential neural network
- Dense(128, relu, L2) → BatchNorm → Dropout(0.3)
- Dense(64, relu, L2) → BatchNorm → Dropout(0.2)
- Dense(32, relu) → Dropout(0.1) → Dense(1, sigmoid)

**Key design choices**:
- Threshold **0.20** (not the default 0.5) — tuned for maximum recall
- SMOTE-Tomek resampling to address class imbalance
- EarlyStopping monitors `val_recall` (not val_loss)

**Top churn drivers** (from SHAP analysis in the notebook):
`tenure`, `Contract_Month-to-month`, `InternetService_Fiber optic`

**EDA and full analysis**: [AnalyticsAndModelling.ipynb](https://github.com/ravsssh/TelcoCustomerChurn/blob/master/AnalyticsAndModelling.ipynb)
    """)
