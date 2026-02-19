"""
Full ML training pipeline for Telco Customer Churn.

Usage
-----
python train.py                          # defaults
python train.py --epochs 50              # limit max epochs
python train.py --min-precision 0.35     # relax precision constraint

Outputs (written to artifacts/)
--------------------------------
churn_model.keras   Native Keras SavedModel format
scaler.pkl          Fitted sklearn StandardScaler
encoder.pkl         Fitted sklearn OneHotEncoder (critical for inference)
config.json         Threshold, metrics, curves, feature names, training history
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Unbuffered output so Streamlit subprocess can stream logs line-by-line
os.environ["PYTHONUNBUFFERED"] = "1"

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.combine import SMOTETomek

from src.preprocessing import (
    load_raw_data,
    encode_dataframe,
    build_feature_matrix,
)
from src.model import create_high_recall_model, get_callbacks, find_optimal_threshold

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
RANDOM_STATE = 42


def _downsample_curve(x: np.ndarray, y: np.ndarray, max_pts: int = 200) -> tuple[list, list]:
    """Downsample a curve to at most max_pts points for compact JSON storage."""
    if len(x) <= max_pts:
        return x.tolist(), y.tolist()
    idx = np.round(np.linspace(0, len(x) - 1, max_pts)).astype(int)
    return x[idx].tolist(), y[idx].tolist()


def train(max_epochs: int = 100, min_precision: float = 0.4) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: Load and encode data
    # ------------------------------------------------------------------
    print("=" * 60, flush=True)
    print("STEP 1: Loading and preprocessing data", flush=True)
    print("=" * 60, flush=True)

    df = load_raw_data()
    df_encoded, encoder = encode_dataframe(df, fit=True)

    print(f"Encoded shape: {df_encoded.shape}", flush=True)

    # ------------------------------------------------------------------
    # STEP 2: Build feature matrix (41 features, no ServiceUsageScore)
    # ------------------------------------------------------------------
    print("\nSTEP 2: Building feature matrix", flush=True)
    X, y = build_feature_matrix(df_encoded)
    feature_names = X.columns.tolist()
    n_features = len(feature_names)
    print(f"Features: {n_features}  |  Samples: {len(X)}  |  Churn rate: {y.mean():.1%}", flush=True)

    # ------------------------------------------------------------------
    # STEP 3: Train/test split
    # ------------------------------------------------------------------
    print("\nSTEP 3: Train/test split (80/20, stratified)", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}", flush=True)

    # ------------------------------------------------------------------
    # STEP 4: Feature scaling
    # ------------------------------------------------------------------
    print("\nSTEP 4: Scaling features (StandardScaler)", flush=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # STEP 5: Class imbalance — SMOTE-Tomek + class weights
    # ------------------------------------------------------------------
    print("\nSTEP 5: Handling class imbalance (SMOTE-Tomek)", flush=True)
    smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)
    print(f"Resampled train size: {len(X_resampled)}", flush=True)

    class_weights_arr = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=y_train.values
    )
    class_weight_dict = {0: class_weights_arr[0], 1: class_weights_arr[1]}
    print(f"Class weights: {class_weight_dict}", flush=True)

    # ------------------------------------------------------------------
    # STEP 6: Build and train model
    # ------------------------------------------------------------------
    print(f"\nSTEP 6: Building neural network (input_dim={n_features})", flush=True)
    model = create_high_recall_model(n_features)
    model.summary(print_fn=lambda s: print(s, flush=True))

    print(f"\nSTEP 7: Training (max_epochs={max_epochs})", flush=True)
    history = model.fit(
        X_resampled,
        y_resampled,
        epochs=max_epochs,
        batch_size=64,
        validation_split=0.2,
        callbacks=get_callbacks(),
        class_weight=class_weight_dict,
        verbose=1,
    )
    epochs_trained = len(history.history["loss"])
    print(f"\nTraining complete. Epochs trained: {epochs_trained}", flush=True)

    # ------------------------------------------------------------------
    # STEP 8: Threshold optimisation
    # ------------------------------------------------------------------
    print("\nSTEP 8: Optimising classification threshold", flush=True)
    y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
    optimal_threshold, threshold_df = find_optimal_threshold(
        y_test.values, y_pred_proba, min_precision=min_precision
    )
    print(f"Optimal threshold: {optimal_threshold}  (min_precision={min_precision})", flush=True)

    # ------------------------------------------------------------------
    # STEP 9: Evaluation metrics
    # ------------------------------------------------------------------
    print("\nSTEP 9: Evaluating model", flush=True)
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel().tolist()

    print(f"Recall:    {recall:.4f}", flush=True)
    print(f"Precision: {precision:.4f}", flush=True)
    print(f"F1:        {f1:.4f}", flush=True)
    print(f"AUC-ROC:   {auc_roc:.4f}", flush=True)
    print(f"Accuracy:  {accuracy:.4f}", flush=True)
    print(f"Confusion matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}", flush=True)

    # ROC and PR curves
    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_pred_proba)
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_pred_proba)
    fpr_ds, tpr_ds = _downsample_curve(fpr_arr, tpr_arr)
    prec_ds, rec_ds = _downsample_curve(prec_arr, rec_arr)

    # ------------------------------------------------------------------
    # STEP 10: Feature importance (permutation, fast ~30s)
    # ------------------------------------------------------------------
    print("\nSTEP 10: Computing permutation feature importance ...", flush=True)

    # Wrap model in a sklearn-compatible scorer
    from sklearn.base import BaseEstimator
    import tensorflow as tf

    class _KerasWrapper(BaseEstimator):
        def __init__(self, keras_model, threshold):
            self.keras_model = keras_model
            self.threshold = threshold

        def fit(self, X, y):
            return self

        def predict(self, X):
            proba = self.keras_model.predict(X, verbose=0).flatten()
            return (proba >= self.threshold).astype(int)

        def score(self, X, y):
            from sklearn.metrics import recall_score as _rs
            return _rs(y, self.predict(X), zero_division=0)

    wrapper = _KerasWrapper(model, optimal_threshold)
    perm_result = permutation_importance(
        wrapper, X_test_scaled, y_test.values,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=1,
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": perm_result.importances_mean,
        "std": perm_result.importances_std,
    }).sort_values("importance", ascending=False).head(20)

    feature_importance_list = [
        {
            "feature": row["feature"],
            "importance": round(float(row["importance"]), 6),
            "std": round(float(row["std"]), 6),
        }
        for _, row in importance_df.iterrows()
    ]
    print("Top 5 features:", [fi["feature"] for fi in feature_importance_list[:5]], flush=True)

    # ------------------------------------------------------------------
    # Business stats
    # ------------------------------------------------------------------
    churner_mask = y_test.values == 1
    total_actual_churners = int(churner_mask.sum())
    churner_avg_monthly = float(
        X_test[y_test == 1]["MonthlyCharges"].mean()
        if "MonthlyCharges" in X_test.columns
        else 74.44
    )

    # ------------------------------------------------------------------
    # STEP 11: Save artifacts
    # ------------------------------------------------------------------
    print("\nSTEP 11: Saving artifacts ...", flush=True)

    model_path = os.path.join(ARTIFACTS_DIR, "churn_model.keras")
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    encoder_path = os.path.join(ARTIFACTS_DIR, "encoder.pkl")
    config_path = os.path.join(ARTIFACTS_DIR, "config.json")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)

    config = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_url": "https://raw.githubusercontent.com/ravsssh/UAS-Machine-Learning/refs/heads/main/CUSTOMER%20CHURN%20TELCO(in).csv",
        "feature_names": feature_names,
        "n_features": n_features,
        "threshold": {
            "optimal": optimal_threshold,
            "min_precision_constraint": min_precision,
            "search_range": [0.10, 0.80],
            "search_step": 0.05,
            "results": threshold_df.to_dict(orient="records"),
        },
        "metrics": {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "auc_roc": round(auc_roc, 4),
            "accuracy": round(accuracy, 4),
        },
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "curves": {
            "roc": {"fpr": fpr_ds, "tpr": tpr_ds},
            "pr": {"precision": prec_ds, "recall": rec_ds},
        },
        "feature_importance": feature_importance_list,
        "training_history": {
            k: [round(float(v), 6) for v in vals]
            for k, vals in history.history.items()
        },
        "training": {
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "smote_resampled_size": int(len(X_resampled)),
            "epochs_trained": int(epochs_trained),
            "max_epochs": int(max_epochs),
            "batch_size": 64,
            "random_state": RANDOM_STATE,
        },
        "class_distribution": {
            "original_churn_rate": round(float(y.mean()), 4),
            "train_churn_rate": round(float(y_train.mean()), 4),
            "test_churn_rate": round(float(y_test.mean()), 4),
        },
        "business": {
            "churner_avg_monthly_revenue": round(churner_avg_monthly, 2),
            "total_actual_churners_test": total_actual_churners,
        },
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  churn_model.keras → {model_path}", flush=True)
    print(f"  scaler.pkl        → {scaler_path}", flush=True)
    print(f"  encoder.pkl       → {encoder_path}", flush=True)
    print(f"  config.json       → {config_path}", flush=True)
    print("\nTraining pipeline complete.", flush=True)
    print(f"FINAL METRICS: Recall={recall:.4f}  Precision={precision:.4f}  AUC={auc_roc:.4f}  Threshold={optimal_threshold}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Telco Churn neural network.")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs (default: 100)")
    parser.add_argument("--min-precision", type=float, default=0.4,
                        help="Minimum precision constraint for threshold selection (default: 0.4)")
    args = parser.parse_args()

    train(max_epochs=args.epochs, min_precision=args.min_precision)


if __name__ == "__main__":
    main()
