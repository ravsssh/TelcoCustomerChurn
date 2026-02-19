"""
Inference utilities for Telco Customer Churn.

This module is intentionally free of Streamlit imports so it can be
imported by both train.py (subprocess, no Streamlit) and Streamlit pages.

Each Streamlit page that calls load_artifacts() should wrap it in its own
@st.cache_resource decorator to avoid reloading on every page render:

    @st.cache_resource
    def get_artifacts():
        return load_artifacts()

After a new training run completes, call get_artifacts.clear() to force
a fresh reload from disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_single_customer, preprocess_batch

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

_REQUIRED_ARTIFACT_FILES = [
    "churn_model.keras",
    "scaler.pkl",
    "encoder.pkl",
    "config.json",
]


def artifacts_exist() -> bool:
    """Return True if all required artifact files are present on disk."""
    return all((ARTIFACTS_DIR / f).exists() for f in _REQUIRED_ARTIFACT_FILES)


def load_artifacts() -> dict:
    """
    Load all artifacts from disk and return them as a dict.

    Returns
    -------
    dict with keys:
        "model"   : keras.Model
        "scaler"  : sklearn.StandardScaler
        "encoder" : sklearn.OneHotEncoder
        "config"  : dict  (parsed config.json)
    """
    if not artifacts_exist():
        raise FileNotFoundError(
            f"Artifacts not found in {ARTIFACTS_DIR}. "
            "Run train.py first to generate model artifacts."
        )

    import keras  # lazy import — avoids TF startup in non-ML contexts

    model = keras.models.load_model(str(ARTIFACTS_DIR / "churn_model.keras"))
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    encoder = joblib.load(ARTIFACTS_DIR / "encoder.pkl")
    with open(ARTIFACTS_DIR / "config.json") as f:
        config = json.load(f)

    return {"model": model, "scaler": scaler, "encoder": encoder, "config": config}


def assign_risk_segment(probability: float) -> str:
    """Map a churn probability to a named risk segment."""
    if probability >= 0.60:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    elif probability >= 0.20:
        return "Low-Medium Risk"
    else:
        return "Low Risk"


def predict_single(raw_dict: dict, artifacts: dict) -> dict:
    """
    Run inference for a single customer.

    Parameters
    ----------
    raw_dict : dict
        Keys = REQUIRED_INPUT_COLS (19 fields, no customerID or Churn).
    artifacts : dict
        Output of load_artifacts().

    Returns
    -------
    dict with keys:
        "probability"   : float  [0, 1]
        "prediction"    : int    0 or 1
        "risk_segment"  : str
    """
    config = artifacts["config"]
    threshold = config["threshold"]["optimal"]
    feature_names = config["feature_names"]

    X_scaled = preprocess_single_customer(
        raw_dict,
        encoder=artifacts["encoder"],
        scaler=artifacts["scaler"],
        feature_names=feature_names,
    )

    probability = float(artifacts["model"].predict(X_scaled, verbose=0).flatten()[0])
    prediction = int(probability >= threshold)
    risk_segment = assign_risk_segment(probability)

    return {
        "probability": round(probability, 4),
        "prediction": prediction,
        "risk_segment": risk_segment,
    }


def predict_batch(df_raw: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Run inference on a batch of raw customer records.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Must contain all columns in REQUIRED_INPUT_COLS.
        May optionally contain 'customerID' (preserved as pass-through).
        'Churn' column is silently ignored if present.

    Returns
    -------
    pd.DataFrame
        Original df_raw columns (excluding 'Churn') plus:
            ChurnProbability  : float
            ChurnPrediction   : int (0/1)
            RiskSegment       : str
    """
    config = artifacts["config"]
    threshold = config["threshold"]["optimal"]
    feature_names = config["feature_names"]

    X_scaled = preprocess_batch(
        df_raw,
        encoder=artifacts["encoder"],
        scaler=artifacts["scaler"],
        feature_names=feature_names,
    )

    probabilities = artifacts["model"].predict(X_scaled, verbose=0).flatten()

    result_df = df_raw.copy()
    if "Churn" in result_df.columns:
        result_df = result_df.drop(columns=["Churn"])

    result_df["ChurnProbability"] = np.round(probabilities, 4)
    result_df["ChurnPrediction"] = (probabilities >= threshold).astype(int)
    result_df["RiskSegment"] = [assign_risk_segment(p) for p in probabilities]

    return result_df


def generate_recommendations(raw_dict: dict, probability: float) -> list[str]:
    """
    Generate rule-based retention recommendations derived from the top
    churn drivers identified by SHAP in the notebook:
    tenure, Contract_Month-to-month, InternetService_Fiber optic,
    PaymentMethod_Electronic check.
    """
    recs = []

    tenure = int(raw_dict.get("tenure", 99))
    contract = str(raw_dict.get("Contract", ""))
    internet = str(raw_dict.get("InternetService", ""))
    payment = str(raw_dict.get("PaymentMethod", ""))
    monthly = float(raw_dict.get("MonthlyCharges", 0))
    security = str(raw_dict.get("OnlineSecurity", ""))
    tech = str(raw_dict.get("TechSupport", ""))

    if probability >= 0.40:
        if contract == "Month-to-month":
            recs.append(
                "Offer a contract upgrade incentive — e.g., 10% discount on a 1-year plan."
            )
        if internet == "Fiber optic":
            recs.append(
                "Fiber optic customers churn at higher rates. Review service satisfaction "
                "and consider a loyalty discount or speed upgrade."
            )
        if tenure < 12:
            recs.append(
                "Early-tenure customer (< 1 year). Trigger a proactive check-in call "
                "and onboarding review."
            )
        if payment == "Electronic check":
            recs.append(
                "Encourage auto-payment setup (bank transfer or credit card) "
                "with a small billing discount."
            )
        if monthly > 80 and security == "No" and tech == "No":
            recs.append(
                "High-value customer without security/support add-ons. "
                "Bundle offer may improve stickiness."
            )

    if not recs:
        if probability >= 0.20:
            recs.append("Moderate risk — include in standard re-engagement campaign.")
        else:
            recs.append("Low churn risk — standard engagement touchpoint is sufficient.")

    return recs
