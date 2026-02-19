"""
Preprocessing pipeline for Telco Customer Churn.

This module is the single source of truth for all data transformation constants
and functions. Both train.py and the inference path use these same definitions
to guarantee consistent column ordering between training and inference.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Constants (extracted from AnalyticsAndModelling.ipynb)
# ---------------------------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/ravsssh/UAS-Machine-Learning/"
    "refs/heads/main/CUSTOMER%20CHURN%20TELCO(in).csv"
)

BINARY_MAPPINGS: dict[str, dict] = {
    "gender": {"Male": 0, "Female": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

MULTIPLE_COLS: list[str] = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

# Columns to drop when building the feature matrix (not fed to the model)
DROP_COLS: list[str] = ["customerID", "Churn", "ServiceUsageScore"]

# Raw input columns required for inference (all columns except customerID and Churn)
REQUIRED_INPUT_COLS: list[str] = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    """Fetch the raw dataset from GitHub and apply type coercions."""
    print(f"Fetching data from {DATA_URL} ...", flush=True)
    df = pd.read_csv(DATA_URL)
    df = _apply_type_coercions(df)
    print(f"Loaded {len(df)} rows x {df.shape[1]} columns.", flush=True)
    return df


def _apply_type_coercions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact type coercions used in the notebook."""
    df = df.copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(bool)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0).astype(float)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["Churn"] = df["Churn"].map({"Yes": True, "No": False})
    return df


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_dataframe(
    df: pd.DataFrame,
    fit: bool = True,
    encoder: OneHotEncoder | None = None,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """
    Encode a DataFrame using binary mappings + one-hot encoding.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after type coercions. Must contain all columns in
        BINARY_MAPPINGS and MULTIPLE_COLS.
    fit : bool
        If True, fit a new OneHotEncoder on df[MULTIPLE_COLS].
        If False, use the provided ``encoder`` (inference path).
    encoder : OneHotEncoder | None
        Pre-fitted encoder to use when fit=False.

    Returns
    -------
    df_encoded : pd.DataFrame
        Encoded DataFrame (43 columns including customerID, Churn).
    encoder : OneHotEncoder
        The fitted encoder (new if fit=True, same object if fit=False).
    """
    df_encoded = df.copy()

    # Binary mappings
    for col, mapping in BINARY_MAPPINGS.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping).astype(float)

    # SeniorCitizen: bool → float
    if "SeniorCitizen" in df_encoded.columns:
        df_encoded["SeniorCitizen"] = df_encoded["SeniorCitizen"].astype(float)

    # Churn: bool/object → float (if present)
    if "Churn" in df_encoded.columns:
        if df_encoded["Churn"].dtype == object:
            df_encoded["Churn"] = df_encoded["Churn"].map({"No": 0, "Yes": 1}).astype(float)
        else:
            df_encoded["Churn"] = df_encoded["Churn"].astype(float)

    # One-hot encoding
    if fit:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        one_hot_encoded = encoder.fit_transform(df_encoded[MULTIPLE_COLS])
    else:
        if encoder is None:
            raise ValueError("encoder must be provided when fit=False")
        one_hot_encoded = encoder.transform(df_encoded[MULTIPLE_COLS])

    one_hot_df = pd.DataFrame(
        one_hot_encoded,
        columns=encoder.get_feature_names_out(MULTIPLE_COLS),
        index=df_encoded.index,
    )

    df_encoded = pd.concat([df_encoded.drop(columns=MULTIPLE_COLS), one_hot_df], axis=1)
    return df_encoded, encoder


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df_encoded: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) from an encoded DataFrame.

    Drops customerID, Churn, and ServiceUsageScore (if present) from X.
    The notebook's model uses 41 features — ServiceUsageScore is computed
    for EDA purposes only and is NOT fed to the model (confirmed by SHAP
    shape (100, 41, 1) in the notebook).

    Returns
    -------
    X : pd.DataFrame  shape (n, 41)
    y : pd.Series     shape (n,)  dtype int
    """
    y = df_encoded["Churn"].astype(int)

    cols_to_drop = [c for c in DROP_COLS if c in df_encoded.columns]
    X = df_encoded.drop(columns=cols_to_drop)

    return X, y


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def preprocess_single_customer(
    raw_dict: dict,
    encoder: OneHotEncoder,
    scaler,  # sklearn StandardScaler
    feature_names: list[str],
) -> np.ndarray:
    """
    Convert a raw customer dict into a scaled feature array for model.predict().

    Parameters
    ----------
    raw_dict : dict
        Keys are the 19 raw input columns (REQUIRED_INPUT_COLS).
        customerID and Churn should NOT be present.
    encoder : OneHotEncoder
        Fitted encoder saved during training.
    scaler : StandardScaler
        Fitted scaler saved during training.
    feature_names : list[str]
        Canonical column order from training (saved in config.json).
        Used to reorder inference columns to match training exactly.

    Returns
    -------
    np.ndarray  shape (1, 41) — ready for model.predict()
    """
    df = pd.DataFrame([raw_dict])

    # Type coercions
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0).astype(float)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["tenure"] = df["tenure"].astype(float)

    # SeniorCitizen: accept bool or int/str
    if df["SeniorCitizen"].dtype == object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map(
            {True: True, False: False, "True": True, "False": False, 1: True, 0: False, "1": True, "0": False}
        )
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(float)

    # Binary mappings
    for col, mapping in BINARY_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype(float)

    # One-hot encode using saved encoder (transform only, NOT fit_transform)
    one_hot_encoded = encoder.transform(df[MULTIPLE_COLS])
    one_hot_df = pd.DataFrame(
        one_hot_encoded,
        columns=encoder.get_feature_names_out(MULTIPLE_COLS),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=MULTIPLE_COLS), one_hot_df], axis=1)

    # Drop columns not in the model (customerID, Churn, ServiceUsageScore if present)
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Reorder columns to match training (guards against any ordering drift)
    df = df.reindex(columns=feature_names, fill_value=0.0)

    # Scale
    X_scaled = scaler.transform(df.values.astype(float))
    return X_scaled


def preprocess_batch(
    df_raw: pd.DataFrame,
    encoder: OneHotEncoder,
    scaler,
    feature_names: list[str],
) -> np.ndarray:
    """
    Preprocess a batch DataFrame of raw customer records for inference.

    Validates required columns, applies the full pipeline, and returns
    a scaled feature matrix.
    """
    # Validate required columns
    missing = [c for c in REQUIRED_INPUT_COLS if c not in df_raw.columns]
    if missing:
        raise ValueError(
            f"Uploaded CSV is missing required columns: {missing}\n"
            f"Required columns: {REQUIRED_INPUT_COLS}"
        )

    df = df_raw.copy()
    df = _apply_type_coercions_batch(df)

    # Binary mappings
    for col, mapping in BINARY_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(float)

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(float)

    # One-hot encode
    one_hot_encoded = encoder.transform(df[MULTIPLE_COLS])
    one_hot_df = pd.DataFrame(
        one_hot_encoded,
        columns=encoder.get_feature_names_out(MULTIPLE_COLS),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=MULTIPLE_COLS), one_hot_df], axis=1)

    # Drop non-feature columns
    for col in DROP_COLS + ["customerID", "Churn"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.reindex(columns=feature_names, fill_value=0.0)
    return scaler.transform(df.values.astype(float))


def _apply_type_coercions_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Type coercions for batch inference (tolerant of string values)."""
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0).astype(float)
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0).astype(float)
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(float)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(float)
    return df
