"""
Telco Customer Churn — Business Dashboard
Early Warning System for Customer Retention
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import artifacts_exist, load_artifacts, predict_batch
from src.preprocessing import load_raw_data

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Guard: model artifacts must exist
# ---------------------------------------------------------------------------
if not artifacts_exist():
    st.error(
        "Model not found. Please run `python train.py` to generate the model first."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner="Analyzing customer churn risk across all accounts...")
def get_all_predictions():
    artifacts = get_artifacts()
    df_raw = load_raw_data()
    result = predict_batch(df_raw, artifacts)
    result["ActualChurn"] = df_raw["Churn"].map({True: "Yes", False: "No"})

    # Count active add-on services (value == "Yes")
    _service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    result["ServicesUsed"] = result[_service_cols].apply(
        lambda row: int(sum(v == "Yes" for v in row)), axis=1
    )
    return result


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
artifacts = get_artifacts()
df = get_all_predictions()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Customer Churn Dashboard")
st.markdown(
    "Monitor churn risk across all customers. "
    "Customers are sorted from **highest to lowest risk** so your team can prioritise action. "
    "**Click a row** to see the full customer profile."
)
st.divider()

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
total    = len(df)
high     = int((df["RiskSegment"] == "High Risk").sum())
medium   = int((df["RiskSegment"] == "Medium Risk").sum())
low_med  = int((df["RiskSegment"] == "Low-Medium Risk").sum())
low      = int((df["RiskSegment"] == "Low Risk").sum())
avg_prob = df["ChurnProbability"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Customers",      f"{total:,}")
c2.metric("🔴 High Risk",         f"{high:,}",    f"{high/total:.1%}",    delta_color="inverse")
c3.metric("🟠 Medium Risk",       f"{medium:,}",  f"{medium/total:.1%}",  delta_color="inverse")
c4.metric("🟡 Low-Medium Risk",   f"{low_med:,}", f"{low_med/total:.1%}", delta_color="inverse")
c5.metric("Avg Churn Probability", f"{avg_prob:.1%}")

st.divider()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
f1, f2, f3 = st.columns([2, 2, 3])
with f1:
    risk_filter = st.selectbox(
        "Risk Level",
        ["All Customers", "🔴 High Risk", "🟠 Medium Risk", "🟡 Low-Medium Risk", "✅ Low Risk"],
    )
with f2:
    contract_filter = st.selectbox(
        "Contract Type",
        ["All Contracts", "Month-to-month", "One year", "Two year"],
    )
with f3:
    search = st.text_input("Search by Customer ID", placeholder="e.g. 1234-ABCD")

# ---------------------------------------------------------------------------
# Filter logic
# ---------------------------------------------------------------------------
_RISK_MAP = {
    "🔴 High Risk":       "High Risk",
    "🟠 Medium Risk":     "Medium Risk",
    "🟡 Low-Medium Risk": "Low-Medium Risk",
    "✅ Low Risk":        "Low Risk",
}

view = df.copy()
if risk_filter != "All Customers":
    view = view[view["RiskSegment"] == _RISK_MAP[risk_filter]]
if contract_filter != "All Contracts":
    view = view[view["Contract"] == contract_filter]
if search.strip():
    view = view[view["customerID"].str.contains(search.strip(), case=False, na=False)]

# ---------------------------------------------------------------------------
# Prepare display table
# ---------------------------------------------------------------------------
_WARNING = {
    "High Risk":       "🔴 Act Now",
    "Medium Risk":     "🟠 Monitor",
    "Low-Medium Risk": "🟡 Watch",
    "Low Risk":        "✅ Safe",
}

view = view.copy()
view["Early Warning"]  = view["RiskSegment"].map(_WARNING)
view["Churn Risk (%)"] = (view["ChurnProbability"] * 100).round(1)
view = view.sort_values("ChurnProbability", ascending=False).reset_index(drop=True)

TABLE_COLS = {
    "customerID":      "Customer ID",
    "Early Warning":   "Early Warning",
    "Churn Risk (%)":  "Churn Risk (%)",
    "Contract":        "Contract",
    "tenure":          "Tenure (months)",
    "MonthlyCharges":  "Monthly Bill ($)",
    "ServicesUsed":    "Services Used",
    "InternetService": "Internet",
}

st.markdown(f"**{len(view):,} of {total:,} customers shown** — sorted by highest churn risk")

event = st.dataframe(
    view[list(TABLE_COLS.keys())].rename(columns=TABLE_COLS),
    use_container_width=True,
    hide_index=True,
    height=480,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Churn Risk (%)": st.column_config.ProgressColumn(
            "Churn Risk (%)",
            min_value=0,
            max_value=100,
            format="%.1f%%",
        ),
        "Tenure (months)": st.column_config.NumberColumn(
            "Tenure (months)",
            format="%d mo",
        ),
        "Monthly Bill ($)": st.column_config.NumberColumn(
            "Monthly Bill ($)",
            format="$%.2f",
        ),
        "Services Used": st.column_config.NumberColumn(
            "Services Used",
            format="%d / 7",
            help="Number of active add-on services (out of 7 possible)",
        ),
    },
)

# ---------------------------------------------------------------------------
# Customer detail panel (shown when a row is selected)
# ---------------------------------------------------------------------------
selected_rows = event.selection.rows if event and event.selection else []

if selected_rows:
    idx = selected_rows[0]
    row = view.iloc[idx]

    st.divider()
    st.subheader(f"Customer Profile — {row['customerID']}")

    risk_colour = {
        "High Risk":       "red",
        "Medium Risk":     "orange",
        "Low-Medium Risk": "orange",
        "Low Risk":        "green",
    }.get(row["RiskSegment"], "blue")
    warning_label = _WARNING.get(row["RiskSegment"], row["RiskSegment"])
    st.markdown(
        f"**Churn Risk: {row['Churn Risk (%)']:.1f}%** &nbsp;|&nbsp; "
        f":{risk_colour}[{warning_label}]"
    )

    d1, d2, d3 = st.columns(3)

    # ── Demographics ──────────────────────────────────────────────────────
    with d1:
        st.markdown("#### 👤 Demographics")
        senior_label = "Yes (65+)" if row["SeniorCitizen"] else "No"
        st.markdown(f"""
| Field | Value |
|---|---|
| **Gender** | {row['gender']} |
| **Senior Citizen** | {senior_label} |
| **Has Partner** | {row['Partner']} |
| **Has Dependents** | {row['Dependents']} |
""")

    # ── Services ─────────────────────────────────────────────────────────
    with d2:
        st.markdown(f"#### 📡 Services ({int(row['ServicesUsed'])} / 7 active)")
        _service_labels = {
            "PhoneService":      "Phone Service",
            "MultipleLines":     "Multiple Lines",
            "InternetService":   "Internet Service",
            "OnlineSecurity":    "Online Security",
            "OnlineBackup":      "Online Backup",
            "DeviceProtection":  "Device Protection",
            "TechSupport":       "Tech Support",
            "StreamingTV":       "Streaming TV",
            "StreamingMovies":   "Streaming Movies",
        }
        for col, label in _service_labels.items():
            val = row[col]
            if val == "Yes":
                st.markdown(f"✅ {label}")
            elif val == "No":
                st.markdown(f"☐ {label}")
            else:
                st.markdown(f"➖ {label} *({val})*")

    # ── Account & Contract ────────────────────────────────────────────────
    with d3:
        st.markdown("#### 📋 Account & Contract")
        paperless = "Yes" if row["PaperlessBilling"] == "Yes" else "No"
        st.markdown(f"""
| Field | Value |
|---|---|
| **Contract** | {row['Contract']} |
| **Payment Method** | {row['PaymentMethod']} |
| **Paperless Billing** | {paperless} |
| **Tenure** | {int(row['tenure'])} months |
| **Monthly Bill** | ${row['MonthlyCharges']:.2f} |
| **Total Charges** | ${row['TotalCharges']:.2f} |
""")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
threshold = artifacts["config"]["threshold"]["optimal"]
st.caption(
    f"Churn is predicted when probability ≥ **{threshold:.0%}**. "
    "Risk levels: 🔴 ≥ 60%  ·  🟠 40–60%  ·  🟡 20–40%  ·  ✅ < 20%. "
    "Use **Single Prediction** in the sidebar to analyse an individual customer."
)
