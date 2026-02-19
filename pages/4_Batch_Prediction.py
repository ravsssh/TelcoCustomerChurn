"""
Page 4 — Batch Prediction

Upload a CSV of customer records → predictions table + download as CSV.
"""

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.inference import artifacts_exist, predict_batch
from src.preprocessing import REQUIRED_INPUT_COLS

st.set_page_config(page_title="Batch Prediction — Telco Churn", page_icon="📂", layout="wide")
st.title("📂 Batch Prediction")
st.markdown(
    "Upload a CSV file containing customer records and get churn predictions for every row. "
    "Optionally include a `customerID` column — it will be preserved in the output."
)
st.divider()

# ---------------------------------------------------------------------------
# Guard: artifacts must exist
# ---------------------------------------------------------------------------
if not artifacts_exist():
    st.warning("No trained model found. Go to **Train Model** to train the model first.")
    st.stop()


@st.cache_resource
def get_artifacts():
    from src.inference import load_artifacts
    return load_artifacts()


artifacts = get_artifacts()

# ---------------------------------------------------------------------------
# Sample CSV template download
# ---------------------------------------------------------------------------
SAMPLE_DATA = [
    {
        "customerID": "SAMPLE-001",
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 1, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.70, "TotalCharges": 70.70,
    },
    {
        "customerID": "SAMPLE-002",
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 34, "PhoneService": "Yes",
        "MultipleLines": "Yes", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "No",
        "DeviceProtection": "Yes", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "One year", "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 56.95, "TotalCharges": 1889.50,
    },
    {
        "customerID": "SAMPLE-003",
        "gender": "Female", "SeniorCitizen": 1, "Partner": "No",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 53.85, "TotalCharges": 108.15,
    },
    {
        "customerID": "SAMPLE-004",
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "Yes", "tenure": 45, "PhoneService": "No",
        "MultipleLines": "No phone service", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "No",
        "DeviceProtection": "Yes", "TechSupport": "Yes",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "One year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 42.30, "TotalCharges": 1840.75,
    },
    {
        "customerID": "SAMPLE-005",
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "Yes", "tenure": 60, "PhoneService": "Yes",
        "MultipleLines": "Yes", "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Two year", "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 112.95, "TotalCharges": 6844.50,
    },
]

sample_df = pd.DataFrame(SAMPLE_DATA)
sample_csv = sample_df.to_csv(index=False).encode("utf-8")

col_dl, col_info = st.columns([1, 2])
with col_dl:
    st.download_button(
        label="Download sample CSV template",
        data=sample_csv,
        file_name="churn_sample_template.csv",
        mime="text/csv",
    )
with col_info:
    st.caption(
        f"Required columns: {', '.join(REQUIRED_INPUT_COLS)}  \n"
        "Optional: `customerID` (preserved in output), `Churn` (ignored/used for comparison)."
    )

st.divider()

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload customer CSV", type=["csv"], help="Must contain the 19 required input columns."
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        st.stop()

    st.markdown(f"**Uploaded:** {uploaded_file.name}  ·  {len(df_raw):,} rows  ·  {df_raw.shape[1]} columns")

    # Validate required columns
    missing_cols = [c for c in REQUIRED_INPUT_COLS if c not in df_raw.columns]
    if missing_cols:
        st.error(
            f"The uploaded CSV is missing **{len(missing_cols)}** required column(s):\n\n"
            + "  \n".join(f"- `{c}`" for c in missing_cols)
        )
        st.stop()

    has_ground_truth = "Churn" in df_raw.columns
    if has_ground_truth:
        st.info("Ground truth `Churn` column detected — will be shown alongside predictions for comparison.")

    with st.expander("Preview uploaded data (first 5 rows)"):
        st.dataframe(df_raw.head())

    # ---- Run predictions ----
    if st.button("Run Predictions", type="primary", use_container_width=True):
        with st.spinner(f"Running predictions on {len(df_raw):,} customers..."):
            try:
                result_df = predict_batch(df_raw, artifacts)
            except ValueError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        # If ground truth present, compute metrics for comparison
        if has_ground_truth:
            from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
            import numpy as np

            churn_col = df_raw["Churn"]
            if churn_col.dtype == object:
                y_true = churn_col.map({"Yes": 1, "No": 0, "True": 1, "False": 0}).fillna(0).astype(int)
            else:
                y_true = churn_col.astype(int)

            y_pred = result_df["ChurnPrediction"].values
            y_proba = result_df["ChurnProbability"].values

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.1%}")
            m_col2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.1%}")
            m_col3.metric("F1", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")
            try:
                m_col4.metric("AUC-ROC", f"{roc_auc_score(y_true, y_proba):.3f}")
            except Exception:
                m_col4.metric("AUC-ROC", "N/A")

        st.divider()
        st.subheader("Prediction Results")

        # ---- Summary charts ----
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            seg_counts = result_df["RiskSegment"].value_counts().reset_index()
            seg_counts.columns = ["RiskSegment", "Count"]
            color_map = {
                "High Risk": "#e74c3c",
                "Medium Risk": "#e67e22",
                "Low-Medium Risk": "#f1c40f",
                "Low Risk": "#2ecc71",
            }
            fig_pie = px.pie(
                seg_counts, names="RiskSegment", values="Count",
                color="RiskSegment", color_discrete_map=color_map,
                title="Risk Segment Distribution",
            )
            fig_pie.update_layout(height=320)
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            fig_hist = go.Figure(go.Histogram(
                x=result_df["ChurnProbability"],
                nbinsx=30,
                marker_color="#3498db",
                opacity=0.8,
            ))
            threshold = artifacts["config"]["threshold"]["optimal"]
            fig_hist.add_vline(
                x=threshold, line_dash="dash", line_color="#e74c3c",
                annotation_text=f"Threshold {threshold}",
                annotation_position="top right",
            )
            fig_hist.update_layout(
                title="Churn Probability Distribution",
                xaxis_title="Churn Probability",
                yaxis_title="Count",
                height=320,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ---- Results table with color coding ----
        def _color_risk(val):
            colors = {
                "High Risk": "background-color: #fadbd8",
                "Medium Risk": "background-color: #fdebd0",
                "Low-Medium Risk": "background-color: #fef9e7",
                "Low Risk": "background-color: #d5f5e3",
            }
            return colors.get(val, "")

        display_df = result_df.copy()
        if has_ground_truth:
            # Insert actual churn column after customerID for easy comparison
            actual_col = df_raw["Churn"].values
            display_df.insert(
                display_df.columns.get_loc("ChurnProbability") + 1,
                "ActualChurn",
                actual_col,
            )

        st.dataframe(
            display_df.style.applymap(_color_risk, subset=["RiskSegment"]),
            use_container_width=True,
        )

        # ---- Download button ----
        csv_output = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download predictions CSV",
            data=csv_output,
            file_name="churn_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ---- Priority table: top churners ----
        with st.expander("Top 10 highest-risk customers"):
            top10 = result_df.nlargest(10, "ChurnProbability")
            if "customerID" in top10.columns:
                show_cols = ["customerID", "ChurnProbability", "ChurnPrediction", "RiskSegment"]
            else:
                show_cols = ["ChurnProbability", "ChurnPrediction", "RiskSegment"]
            st.dataframe(top10[show_cols].reset_index(drop=True))
