"""
Model architecture and training utilities for Telco Customer Churn.

Functions extracted verbatim from AnalyticsAndModelling.ipynb (with minor
refactoring for reusability). The architecture and hyperparameters are
identical to the notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


def create_high_recall_model(input_dim: int) -> Sequential:
    """
    Build and compile the Keras neural network for churn prediction.

    Architecture (matches notebook exactly):
        Input(input_dim)
        → Dense(128, relu, L2=0.001) → BatchNorm → Dropout(0.3)
        → Dense(64,  relu, L2=0.001) → BatchNorm → Dropout(0.2)
        → Dense(32,  relu, L2=0.001) → Dropout(0.1)
        → Dense(1, sigmoid)

    Compiled with Adam(lr=0.001), binary_crossentropy,
    metrics=[accuracy, precision, recall].
    """
    model = Sequential([
        Dense(
            128,
            input_shape=(input_dim,),
            activation="relu",
            kernel_regularizer=l2(0.001),
        ),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.1),

        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"],
    )
    return model


def get_callbacks() -> list:
    """
    Return the training callbacks used in the notebook.

    - EarlyStopping: monitors val_recall, patience=15, restores best weights.
    - ReduceLROnPlateau: monitors val_loss, factor=0.5, patience=5,
      min_lr=0.0001.
    """
    return [
        EarlyStopping(
            monitor="val_recall",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1,
        ),
    ]


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.4,
) -> tuple[float, pd.DataFrame]:
    """
    Grid-search for the threshold that maximises recall while keeping
    precision >= min_precision.

    Search range: np.arange(0.10, 0.80, 0.05)

    Returns
    -------
    best_threshold : float
    results_df : pd.DataFrame
        Columns: threshold, recall, precision, f1
    """
    thresholds = np.arange(0.10, 0.80, 0.05)
    best_threshold = 0.5
    best_recall = 0.0
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "threshold": round(float(threshold), 2),
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "f1": round(float(f1), 4),
        })

        if recall > best_recall and precision >= min_precision:
            best_recall = recall
            best_threshold = round(float(threshold), 2)

    return best_threshold, pd.DataFrame(results)
