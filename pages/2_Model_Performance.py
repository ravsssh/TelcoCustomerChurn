"""
Page 2 — Model Performance

Displays confusion matrix, ROC / PR curves, threshold analysis,
feature importance, training history, and a business impact calculator.
All data is read exclusively from artifacts/config.json — no data re-fetch.
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"

st.set_page_config(page_title="Model Performance — Telco Churn", page_icon="📊", layout="wide")
st.title("📊 Model Performance")
st.divider()

# ---------------------------------------------------------------------------
# Guard: model must exist
# ---------------------------------------------------------------------------
if not CONFIG_PATH.exists():
    st.warning("No trained model found. Go to **Train Model** to train the model first.")
    st.stop()

config = json.loads(CONFIG_PATH.read_text())
m = config["metrics"]
t = config["threshold"]
cm = config["confusion_matrix"]
hist = config.get("training_history", {})
curves = config.get("curves", {})
fi_list = config.get("feature_importance", [])
tr = config.get("training", {})

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
st.subheader("Metrics Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Recall", f"{m['recall']:.1%}", help="Proportion of actual churners caught by the model")
c2.metric("Precision", f"{m['precision']:.1%}", help="Proportion of flagged customers who actually churn")
c3.metric("ROC-AUC", f"{m['auc_roc']:.3f}", help="Model's ranking quality (1.0 = perfect)")
c4.metric("F1-Score", f"{m['f1']:.3f}")
c5.metric("Threshold", f"{t['optimal']:.2f}", help="Classification threshold (tuned for recall, not default 0.5)")

st.caption(
    f"Threshold {t['optimal']} selected by maximising recall with precision ≥ {t['min_precision_constraint']:.0%}  ·  "
    f"Test set: {tr.get('test_size', '?')} customers  ·  Trained: {config.get('trained_at', '?')}"
)
st.divider()

# ---------------------------------------------------------------------------
# Row 1: Confusion Matrix + ROC + PR
# ---------------------------------------------------------------------------
st.subheader("Classification Results")
col_cm, col_roc, col_pr = st.columns([1, 1.3, 1.3])

with col_cm:
    st.markdown("**Confusion Matrix**")
    tn, fp, fn, tp_ = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    z = [[tn, fp], [fn, tp_]]
    text = [
        [f"TN\n{tn}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TP\n{tp_}"],
    ]
    fig_cm = go.Figure(go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 16},
        x=["Predicted: No Churn", "Predicted: Churn"],
        y=["Actual: No Churn", "Actual: Churn"],
        colorscale=[[0, "#d5f5e3"], [1, "#e74c3c"]],
        showscale=False,
    ))
    fig_cm.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_cm, use_container_width=True)

    caught = tp_
    total_churners = tp_ + fn
    st.caption(
        f"Catches **{caught}/{total_churners}** churners ({caught/max(total_churners,1):.1%} recall). "
        f"**{fp}** false alarms out of {tn + fp} non-churners."
    )

with col_roc:
    st.markdown("**ROC Curve**")
    roc = curves.get("roc", {})
    if roc:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"],
            mode="lines", name=f"AUC = {m['auc_roc']:.3f}",
            line=dict(color="#3498db", width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random", line=dict(color="grey", dash="dash", width=1),
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(x=0.6, y=0.05),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

with col_pr:
    st.markdown("**Precision-Recall Curve**")
    pr = curves.get("pr", {})
    if pr:
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=pr["recall"], y=pr["precision"],
            mode="lines", name="PR Curve",
            line=dict(color="#9b59b6", width=2),
        ))
        # Mark operating point
        fig_pr.add_trace(go.Scatter(
            x=[m["recall"]], y=[m["precision"]],
            mode="markers", name="Operating point",
            marker=dict(color="#e74c3c", size=12, symbol="star"),
        ))
        # Min precision line
        fig_pr.add_hline(
            y=t["min_precision_constraint"],
            line_dash="dash", line_color="orange",
            annotation_text=f"Min precision = {t['min_precision_constraint']:.0%}",
            annotation_position="bottom right",
        )
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Row 2: Threshold analysis
# ---------------------------------------------------------------------------
st.subheader("Threshold Analysis")
threshold_results = t.get("results", [])
if threshold_results:
    import pandas as pd
    thr_df = pd.DataFrame(threshold_results)
    fig_thr = go.Figure()
    fig_thr.add_trace(go.Scatter(
        x=thr_df["threshold"], y=thr_df["recall"],
        mode="lines+markers", name="Recall", line=dict(color="#9b59b6", width=2),
    ))
    fig_thr.add_trace(go.Scatter(
        x=thr_df["threshold"], y=thr_df["precision"],
        mode="lines+markers", name="Precision", line=dict(color="#e67e22", width=2),
    ))
    fig_thr.add_trace(go.Scatter(
        x=thr_df["threshold"], y=thr_df["f1"],
        mode="lines+markers", name="F1", line=dict(color="#2ecc71", width=2, dash="dot"),
    ))
    fig_thr.add_vline(
        x=t["optimal"], line_dash="dash", line_color="#e74c3c",
        annotation_text=f"Chosen: {t['optimal']}",
        annotation_position="top right",
    )
    fig_thr.add_hline(
        y=t["min_precision_constraint"], line_dash="dash", line_color="orange",
        annotation_text=f"Min precision = {t['min_precision_constraint']:.0%}",
    )
    fig_thr.update_layout(
        xaxis_title="Threshold",
        yaxis_title="Score",
        height=350,
        hovermode="x unified",
    )
    st.plotly_chart(fig_thr, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Row 3: Feature importance
# ---------------------------------------------------------------------------
if fi_list:
    st.subheader("Feature Importance (Permutation)")

    import pandas as pd
    fi_df = pd.DataFrame(fi_list).head(20)

    # Colour by feature category
    def _category_color(feat: str) -> str:
        if feat in ("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
                    "gender", "Partner", "Dependents"):
            return "#3498db"
        if "Contract" in feat:
            return "#e74c3c"
        if "InternetService" in feat or "PhoneService" in feat:
            return "#9b59b6"
        if "PaymentMethod" in feat or "PaperlessBilling" in feat:
            return "#e67e22"
        return "#2ecc71"

    colors = [_category_color(f) for f in fi_df["feature"]]

    fig_fi = go.Figure(go.Bar(
        x=fi_df["importance"],
        y=fi_df["feature"],
        orientation="h",
        marker_color=colors,
        error_x=dict(type="data", array=fi_df["std"].tolist()),
    ))
    fig_fi.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Mean decrease in recall (permutation importance)",
        height=max(350, len(fi_df) * 22),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    with st.expander("Legend"):
        cols = st.columns(4)
        cols[0].markdown("🔵 Customer profile / charges")
        cols[1].markdown("🔴 Contract type")
        cols[2].markdown("🟣 Internet / phone service")
        cols[3].markdown("🟠 Payment / billing")

st.divider()

# ---------------------------------------------------------------------------
# Row 4: Training history
# ---------------------------------------------------------------------------
if hist:
    st.subheader("Training History")
    epochs_x = list(range(1, len(hist.get("loss", [])) + 1))

    fig_hist = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Loss", "Accuracy", "Precision", "Recall"),
    )
    _map = [
        ("loss", "val_loss", 1, 1, "#e74c3c"),
        ("accuracy", "val_accuracy", 1, 2, "#2ecc71"),
        ("precision", "val_precision", 2, 1, "#3498db"),
        ("recall", "val_recall", 2, 2, "#9b59b6"),
    ]
    for train_key, val_key, row, col, color in _map:
        if train_key in hist:
            fig_hist.add_trace(
                go.Scatter(x=epochs_x, y=hist[train_key], name=f"train",
                           line=dict(color=color), legendgroup=train_key, showlegend=row == 1 and col == 1),
                row=row, col=col,
            )
        if val_key in hist:
            fig_hist.add_trace(
                go.Scatter(x=epochs_x, y=hist[val_key], name=f"val",
                           line=dict(color=color, dash="dash"), legendgroup=val_key, showlegend=row == 1 and col == 1),
                row=row, col=col,
            )

    # Annotate early stop epoch
    epochs_trained = tr.get("epochs_trained", len(epochs_x))
    if epochs_trained < tr.get("max_epochs", epochs_trained + 1):
        for r in [1, 2]:
            for c in [1, 2]:
                fig_hist.add_vline(
                    x=epochs_trained, line_dash="dot", line_color="grey",
                    annotation_text="Early stop" if r == 1 and c == 1 else None,
                    row=r, col=c,
                )

    fig_hist.update_layout(height=480, hovermode="x unified")
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Business Impact Calculator
# ---------------------------------------------------------------------------
st.subheader("Business Impact Calculator")
st.markdown("Adjust the sliders to model the financial impact of deploying this churn model.")

biz = config.get("business", {})
default_avg_revenue = float(biz.get("churner_avg_monthly_revenue", 74.44))
total_churners_test = int(biz.get("total_actual_churners_test", tp_ + fn))

with st.sidebar:
    st.markdown("### Business Parameters")
    avg_monthly_revenue = st.slider(
        "Avg monthly revenue per churner ($)",
        min_value=20, max_value=200, value=int(default_avg_revenue), step=5,
    )
    intervention_cost = st.slider(
        "Intervention cost per contacted customer ($)",
        min_value=5, max_value=150, value=15, step=5,
    )
    retention_rate = st.slider(
        "Retention success rate",
        min_value=0.10, max_value=0.90, value=0.60, step=0.05,
        format="%.0f%%",
    )
    months_horizon = st.slider(
        "Revenue horizon (months)",
        min_value=1, max_value=24, value=12, step=1,
    )

annual_revenue_per_churner = avg_monthly_revenue * months_horizon
revenue_saved = tp_ * retention_rate * annual_revenue_per_churner
intervention_total_cost = (tp_ + fp) * intervention_cost
revenue_missed = fn * annual_revenue_per_churner
net_benefit = revenue_saved - intervention_total_cost

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Revenue Saved",
    f"${revenue_saved:,.0f}",
    help=f"{tp_} caught churners × {retention_rate:.0%} success × ${annual_revenue_per_churner:,.0f}/customer",
)
col2.metric(
    "Intervention Cost",
    f"${intervention_total_cost:,.0f}",
    delta=f"-${intervention_total_cost:,.0f}",
    delta_color="inverse",
    help=f"({tp_} TP + {fp} FP) × ${intervention_cost}/customer",
)
col3.metric(
    "Net Benefit",
    f"${net_benefit:,.0f}",
    delta=f"{'▲' if net_benefit > 0 else '▼'} vs $0 baseline",
    delta_color="normal" if net_benefit > 0 else "inverse",
)
col4.metric(
    "Revenue at Risk (Missed)",
    f"${revenue_missed:,.0f}",
    delta=f"-${revenue_missed:,.0f}",
    delta_color="inverse",
    help=f"{fn} missed churners × ${annual_revenue_per_churner:,.0f} each",
)

if net_benefit > 0:
    st.success(
        f"At these parameters, deploying the model yields a **net benefit of ${net_benefit:,.0f}** "
        f"over {months_horizon} months on a test cohort of {tr.get('test_size', '?')} customers."
    )
else:
    st.warning(
        "Under these parameters the intervention costs exceed revenue saved. "
        "Consider reducing intervention cost or increasing retention success rate."
    )
