"""
Page 1 — Train Model

Triggers the full training pipeline (train.py) as a subprocess and streams
Keras epoch output live to the UI. After completion, displays final metrics
and training history charts.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
TRAIN_SCRIPT = Path(__file__).parent.parent / "train.py"
TRAINING_TIMEOUT_SECONDS = 600  # 10 minutes

# Regex patterns for parsing Keras stdout
_EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)")
_VAL_RECALL_RE = re.compile(r"val_recall:\s*([\d.]+)")
_VAL_LOSS_RE = re.compile(r"val_loss:\s*([\d.]+)")
_LR_RE = re.compile(r"ReduceLROnPlateau reducing learning rate to\s*([\de.+\-]+)")
_FINAL_RE = re.compile(r"FINAL METRICS:.*Recall=([\d.]+).*Precision=([\d.]+).*AUC=([\d.]+).*Threshold=([\d.]+)")

st.set_page_config(page_title="Train Model — Telco Churn", page_icon="🏋️", layout="wide")
st.title("🏋️ Train Model")
st.markdown(
    "Fetches the latest data from GitHub, runs the full preprocessing and "
    "training pipeline, and saves all artifacts. Training typically takes "
    "**2–5 minutes** depending on hardware."
)
st.divider()


# ---------------------------------------------------------------------------
# Check if a model already exists
# ---------------------------------------------------------------------------
required_files = ["churn_model.keras", "scaler.pkl", "encoder.pkl", "config.json"]
model_exists = all((ARTIFACTS_DIR / f).exists() for f in required_files)

if model_exists:
    existing_config = json.loads((ARTIFACTS_DIR / "config.json").read_text())
    st.warning(
        f"A trained model already exists (last trained: **{existing_config.get('trained_at', 'unknown')}**). "
        "Starting a new run will **overwrite** the existing artifacts."
    )

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
with st.expander("Training configuration", expanded=not model_exists):
    col1, col2 = st.columns(2)
    with col1:
        max_epochs = st.number_input(
            "Max epochs", min_value=5, max_value=200, value=100, step=5,
            help="Training will stop earlier if EarlyStopping triggers (patience=15 on val_recall)."
        )
    with col2:
        min_precision = st.slider(
            "Min precision constraint", min_value=0.20, max_value=0.70, value=0.40, step=0.05,
            help="Threshold search selects the threshold with highest recall at this minimum precision."
        )

st.divider()

# ---------------------------------------------------------------------------
# Start training button
# ---------------------------------------------------------------------------
start_btn = st.button("Start Training", type="primary", use_container_width=True)

if start_btn:
    st.divider()
    st.subheader("Training Log")

    progress_bar = st.progress(0, text="Starting training pipeline...")
    col_epoch, col_recall, col_loss, col_lr = st.columns(4)
    epoch_metric = col_epoch.empty()
    recall_metric = col_recall.empty()
    loss_metric = col_loss.empty()
    lr_metric = col_lr.empty()
    log_area = st.empty()

    epoch_metric.metric("Epoch", "—")
    recall_metric.metric("val_recall", "—")
    loss_metric.metric("val_loss", "—")
    lr_metric.metric("Learning rate", "—")

    # Launch training subprocess
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--epochs", str(int(max_epochs)),
        "--min-precision", str(min_precision),
    ]

    start_time = time.time()
    log_lines: list[str] = []
    current_epoch = 0
    current_lr = "0.001000"
    success = False

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(TRAIN_SCRIPT.parent),
        )

        for line in proc.stdout:
            # Timeout guard
            if time.time() - start_time > TRAINING_TIMEOUT_SECONDS:
                proc.terminate()
                st.error("Training timed out after 10 minutes. Try reducing max epochs.")
                break

            line = line.rstrip()
            log_lines.append(line)

            # Update rolling log (last 35 lines)
            log_area.code("\n".join(log_lines[-35:]), language=None)

            # Parse epoch progress
            m_epoch = _EPOCH_RE.search(line)
            if m_epoch:
                current_epoch = int(m_epoch.group(1))
                total_epochs = int(m_epoch.group(2))
                frac = current_epoch / total_epochs
                elapsed = time.time() - start_time
                eta_s = int((elapsed / max(current_epoch, 1)) * (total_epochs - current_epoch))
                progress_bar.progress(
                    min(frac * 0.9, 0.9),  # cap at 90% — last 10% for post-training steps
                    text=f"Epoch {current_epoch}/{total_epochs}  ·  ETA ~{eta_s}s",
                )
                epoch_metric.metric("Epoch", f"{current_epoch}/{total_epochs}")

            # Parse val_recall
            m_recall = _VAL_RECALL_RE.search(line)
            if m_recall:
                recall_metric.metric("val_recall", f"{float(m_recall.group(1)):.4f}")

            # Parse val_loss
            m_loss = _VAL_LOSS_RE.search(line)
            if m_loss:
                loss_metric.metric("val_loss", f"{float(m_loss.group(1)):.4f}")

            # Parse LR reduction
            m_lr = _LR_RE.search(line)
            if m_lr:
                current_lr = m_lr.group(1)
                lr_metric.metric("Learning rate", current_lr)

        proc.wait()
        success = proc.returncode == 0

    except Exception as e:
        st.error(f"Subprocess error: {e}")
        success = False

    progress_bar.progress(1.0 if success else 0.0, text="Done" if success else "Failed")

    if success:
        st.success("Training complete! Artifacts saved to artifacts/")

        # Invalidate cache so other pages reload fresh artifacts
        try:
            from src.inference import load_artifacts
            # The cache wrapper is defined per page — clear via module-level cache
            st.cache_resource.clear()
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Show final metrics from freshly written config.json
        # ------------------------------------------------------------------
        st.divider()
        st.subheader("Final Metrics")
        config = json.loads((ARTIFACTS_DIR / "config.json").read_text())
        m = config["metrics"]
        t = config["threshold"]
        tr = config["training"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Recall", f"{m['recall']:.1%}")
        c2.metric("Precision", f"{m['precision']:.1%}")
        c3.metric("ROC-AUC", f"{m['auc_roc']:.3f}")
        c4.metric("F1-Score", f"{m['f1']:.3f}")
        c5.metric("Threshold", f"{t['optimal']:.2f}")

        st.caption(
            f"Trained for {tr['epochs_trained']}/{tr['max_epochs']} epochs  ·  "
            f"Train size: {tr['train_size']}  ·  Test size: {tr['test_size']}"
        )

        # ------------------------------------------------------------------
        # Training history charts
        # ------------------------------------------------------------------
        st.divider()
        st.subheader("Training History")
        hist = config.get("training_history", {})

        if hist:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Loss", "Accuracy", "Precision", "Recall"),
            )
            epochs_x = list(range(1, len(hist.get("loss", [])) + 1))

            _metrics_map = [
                ("loss", "val_loss", 1, 1, "#e74c3c"),
                ("accuracy", "val_accuracy", 1, 2, "#2ecc71"),
                ("precision", "val_precision", 2, 1, "#3498db"),
                ("recall", "val_recall", 2, 2, "#9b59b6"),
            ]

            for train_key, val_key, row, col, color in _metrics_map:
                if train_key in hist:
                    fig.add_trace(
                        go.Scatter(x=epochs_x, y=hist[train_key], name=f"train_{train_key}",
                                   line=dict(color=color), showlegend=True),
                        row=row, col=col,
                    )
                if val_key in hist:
                    fig.add_trace(
                        go.Scatter(x=epochs_x, y=hist[val_key], name=f"val_{train_key}",
                                   line=dict(color=color, dash="dash"), showlegend=True),
                        row=row, col=col,
                    )

            fig.update_layout(height=500, title_text="Training vs Validation", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.error(
            "Training failed. Check the log above for errors. "
            "Common causes: network issue fetching data, out-of-memory, or import error."
        )
