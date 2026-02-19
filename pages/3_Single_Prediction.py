"""
Page 3 — Single Customer Prediction

Customer input form → churn probability gauge + risk badge + retention recommendations.
"""

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.inference import artifacts_exist, generate_recommendations, predict_single

st.set_page_config(page_title="Single Prediction — Telco Churn", page_icon="🔍", layout="wide")
st.title("🔍 Single Customer Prediction")
st.markdown("Fill in the customer's details and click **Predict** to get their churn probability.")
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
# Input form — 3 column layout
# ---------------------------------------------------------------------------
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    # ---- Column 1: Customer profile & account ----
    with col1:
        st.markdown("**Customer Profile**")
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        senior = st.checkbox("Senior Citizen (65+)")
        partner = st.radio("Has Partner", ["Yes", "No"], horizontal=True)
        dependents = st.radio("Has Dependents", ["Yes", "No"], horizontal=True)

        st.markdown("**Account**")
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0, max_value=10000.0,
            value=round(tenure * monthly_charges, 2),
            step=1.0,
        )

    # ---- Column 2: Services ----
    with col2:
        st.markdown("**Phone & Internet**")
        phone_service = st.radio("Phone Service", ["Yes", "No"], horizontal=True)

        if phone_service == "Yes":
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        else:
            multiple_lines = "No phone service"
            st.info("Multiple Lines: No phone service")

        internet_service = st.selectbox(
            "Internet Service", ["Fiber optic", "DSL", "No"], index=0
        )

        st.markdown("**Add-on Services**")
        if internet_service != "No":
            online_security = st.selectbox("Online Security", ["No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        else:
            online_security = online_backup = device_protection = "No internet service"
            tech_support = streaming_tv = streaming_movies = "No internet service"
            st.info("Add-on services: No internet service")

    # ---- Column 3: Contract & Billing ----
    with col3:
        st.markdown("**Contract & Billing**")
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
            index=0,
        )
        paperless_billing = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        st.markdown("---")
        st.markdown("**Quick risk indicators**")
        if contract == "Month-to-month":
            st.warning("Month-to-month contract — higher churn risk")
        if internet_service == "Fiber optic":
            st.warning("Fiber optic — associated with higher churn")
        if tenure < 12:
            st.info("Tenure < 1 year — early churn window")
        if payment_method == "Electronic check":
            st.info("Electronic check — correlated with churn")

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------
if submitted:
    raw_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    with st.spinner("Running prediction..."):
        result = predict_single(raw_dict, artifacts)

    prob = result["probability"]
    risk = result["risk_segment"]

    st.divider()
    col_gauge, col_info = st.columns([1, 1])

    # ---- Gauge chart ----
    with col_gauge:
        risk_color = {
            "High Risk": "#e74c3c",
            "Medium Risk": "#e67e22",
            "Low-Medium Risk": "#f1c40f",
            "Low Risk": "#2ecc71",
        }.get(risk, "#3498db")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": "Churn Probability", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": risk_color, "thickness": 0.25},
                "steps": [
                    {"range": [0, 20], "color": "#d5f5e3"},
                    {"range": [20, 40], "color": "#fef9e7"},
                    {"range": [40, 60], "color": "#fdebd0"},
                    {"range": [60, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": artifacts["config"]["threshold"]["optimal"] * 100,
                },
            },
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ---- Risk badge and details ----
    with col_info:
        badge_color = {
            "High Risk": "red",
            "Medium Risk": "orange",
            "Low-Medium Risk": "yellow",
            "Low Risk": "green",
        }.get(risk, "blue")

        st.markdown(f"### :{badge_color}[{risk}]")
        threshold = artifacts["config"]["threshold"]["optimal"]
        prediction_label = "**Churn predicted**" if prob >= threshold else "**No churn predicted**"
        st.markdown(
            f"{prediction_label} at threshold **{threshold}**  \n"
            f"Churn probability: **{prob:.1%}**"
        )

        st.markdown("---")
        st.markdown("**Retention Recommendations**")
        recommendations = generate_recommendations(raw_dict, prob)
        for rec in recommendations:
            st.markdown(f"• {rec}")

    # ---- Customer summary ----
    with st.expander("Customer input summary"):
        import pandas as pd
        summary = {k: str(v) for k, v in raw_dict.items()}
        st.dataframe(pd.DataFrame([summary]).T.rename(columns={0: "Value"}))
