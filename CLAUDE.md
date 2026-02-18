# CLAUDE.md — Telco Customer Churn Project

This file provides context for AI assistants (e.g., Claude Code) working in this repository.

---

## Project Overview

This is a **data science / machine learning project** focused on predicting customer churn for a telecommunications company. The goal is to generate actionable business insights, build a high-recall churn prediction model, and recommend targeted retention strategies.

- **Dataset**: IBM Telco Customer Churn — 7,043 rows × 21 columns
- **Target variable**: `Churn` (binary: Yes / No)
- **Best model**: Neural Network (Keras Sequential), ROC-AUC ≈ 0.828, Recall ≈ 0.93 at threshold 0.20
- **Key churn drivers**: Tenure, Month-to-month contract type, Fiber Optic internet service
- **Language**: Python 3
- **Primary artifact**: `AnalyticsAndModelling.ipynb` (a single self-contained Jupyter notebook)

---

## Repository Structure

```
TelcoCustomerChurn/
├── AnalyticsAndModelling.ipynb   # Main notebook — all analysis, modelling, and evaluation
├── Presentation Deck[EN].pdf     # Stakeholder summary slides (English)
├── Presentation Deck[ID].pdf     # Stakeholder summary slides (Bahasa Indonesia)
├── requirements.txt              # Pinned Python dependencies
├── README.md                     # Project overview and quickstart
├── CLAUDE.md                     # This file
├── .gitignore                    # Ignores .venv/, out/, data/, models/, artifacts/
└── .gitattributes                # LF line-ending normalization
```

> **Note**: `data/`, `models/`, and `artifacts/` directories are git-ignored. Raw data is fetched directly from a GitHub URL inside the notebook at runtime. Saved model files (`churn_model.h5`, `scaler.pkl`) are also git-ignored.

---

## Notebook Architecture

`AnalyticsAndModelling.ipynb` is the single source of truth. It is structured into five top-level sections and ten numbered modelling steps:

### Top-level sections
1. **Required Modules Installation** — `pip install` block for runtime environments (e.g., Colab)
2. **Data Collection** — Load CSV from IBM/GitHub URL into a Pandas DataFrame
3. **Data Preprocessing** — Type coercion, missing value handling, encoding pipeline
4. **EDA** — Exploratory data analysis with visualizations
5. **Modelling** — Full ML pipeline, evaluation, explainability, and business impact

### Modelling steps (within section 5)
| Step | Description |
|------|-------------|
| 1 | Data preparation (train/test split, stratified 80/20) |
| 2 | Feature scaling (StandardScaler) |
| 3 | Handling class imbalance (SMOTE-Tomek + class weights) |
| 4 | Building the neural network (`create_high_recall_model`) |
| 5 | Setting up training callbacks (EarlyStopping, ReduceLROnPlateau) |
| 6 | Training the model |
| 7 | Optimizing classification threshold for high recall |
| 8 | Model evaluation (metrics, confusion matrix) |
| 9 | Business impact analysis |
| 10 | Probability analysis and per-customer predictions |

---

## Key Functions

| Function | Purpose |
|----------|---------|
| `check_data_quality(df, target_column='Churn')` | Assess missing values, types, and class balance |
| `setup_churn_style()` | Configure Matplotlib/Seaborn plot style |
| `datalabel_bar(ax, fontsize=12)` | Add value labels to bar charts |
| `detect_outliers_iqr(data, column)` | IQR-based outlier detection |
| `detect_outliers_zscore(data, threshold=3)` | Z-score outlier detection |
| `detect_outliers_modified_zscore(data, threshold=3.5)` | Modified Z-score outlier detection |
| `encode_dataframe(df, binary_mappings, multiple_cols)` | Full encoding pipeline (binary + one-hot) |
| `service_distribution_by_usage(score)` | Analyse service subscriptions by usage score |
| `common_service_combinations(df, service_cols, usage_score)` | Find common service bundles |
| `create_churn_histogram(data, var_name, figsize=(12,7))` | Visualise churn distribution for a variable |
| `create_high_recall_model(input_dim)` | Build and compile the Keras neural network |
| `find_optimal_threshold(y_true, y_proba, min_precision=0.4)` | Grid-search threshold for best recall at ≥40% precision |

---

## Data & Features

### Raw dataset (21 columns)
| Category | Columns |
|----------|---------|
| Customer profile | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account info | `tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`, `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Target | `Churn` |

### Engineered features
- `ServiceUsageScore` — count of active add-on services per customer (integer sum)

### Encoding
- **Binary mappings**: `gender` (Male→0, Female→1), `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` (No→0, Yes→1)
- **One-hot encoded**: multi-category columns (e.g., `Contract`, `InternetService`, `PaymentMethod`)
- **Post-encoding**: 43 columns total

---

## Model Configuration

### Neural Network Architecture
```
Input(43) → Dense(128, relu, L2=0.001) → BatchNorm → Dropout(0.3)
          → Dense(64,  relu, L2=0.001) → BatchNorm → Dropout(0.2)
          → Dense(32,  relu)            →            Dropout(0.1)
          → Dense(1, sigmoid)
```

### Training hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr=0.001 |
| Loss | binary_crossentropy |
| Batch size | 64 |
| Max epochs | 100 |
| Validation split | 0.2 |
| Early stopping patience | 15 (monitors `val_recall`) |
| LR reduction patience | 5 (factor=0.5, min_lr=0.0001) |
| Random seed | 42 (all stochastic operations) |

### Class imbalance
- SMOTE-Tomek with `sampling_strategy=0.8`
- `compute_class_weight('balanced')` passed to `fit()`

### Threshold
- Default Keras threshold (0.5) is **not** used
- Optimal threshold: **0.20** — found by grid search over [0.10, 0.80] in steps of 0.05
- Constraint: minimum precision ≥ 0.40

---

## Evaluation Metrics

The project prioritises **recall** (catching churners) over precision due to the cost asymmetry of missing a churning customer vs. a false alarm.

| Metric | Notes |
|--------|-------|
| ROC-AUC | Primary ranking metric |
| Recall | Primary operational metric (target: high) |
| Precision | Minimum constraint in threshold optimisation |
| F1-Score | Reported but secondary |
| Confusion matrix | TP, TN, FP, FN counts |
| Business impact | Caught vs missed churners, cost/benefit analysis |

---

## Explainability

- **SHAP KernelExplainer**: 100 background samples, 100 explanation samples
- **Permutation importance**: fallback when SHAP is slow/unavailable
- Top identified churn drivers: `tenure`, `Contract_Month-to-month`, `InternetService_Fiber optic`

---

## Development Workflow

### Environment setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Running the notebook interactively
```bash
jupyter notebook AnalyticsAndModelling.ipynb
```

### Running the notebook non-interactively (papermill)
```bash
pip install papermill
papermill AnalyticsAndModelling.ipynb out/AnalyticsAndModelling.output.ipynb
```

The `out/` directory is git-ignored; create it locally if needed.

### Google Colab
Click the "Open in Colab" badge in README.md. The notebook's first cell installs all dependencies via `pip install`.

---

## Key Conventions

1. **Single notebook**: All logic lives in `AnalyticsAndModelling.ipynb`. Do not split into separate `.py` modules unless the project explicitly migrates to a script-based pipeline.
2. **Random seed = 42**: Use `random_state=42` for all stochastic operations (train/test split, SMOTE, permutation importance, etc.).
3. **Recall-first**: Model selection and threshold decisions prioritise recall. Do not default to 0.5 threshold.
4. **No raw data in repo**: The dataset is fetched at runtime from a URL. Never commit CSVs or raw data files.
5. **No model artefacts in repo**: `churn_model.h5` and `scaler.pkl` are git-ignored; treat them as local outputs.
6. **Pinned dependencies**: Keep `requirements.txt` pinned. Update versions deliberately and test the full notebook before committing.
7. **Encoding pipeline**: Always use `encode_dataframe()` to ensure consistent column ordering between train and inference. Do not manually one-hot encode outside this function.
8. **Visualisation style**: Call `setup_churn_style()` at the top of any new visualisation section to maintain consistent plot aesthetics.

---

## Dependencies (key packages)

| Package | Version | Role |
|---------|---------|------|
| pandas | 2.3.2 | Data manipulation |
| numpy | 2.2.0 | Numerical computation |
| scikit-learn | 1.7.1 | Preprocessing, metrics, utilities |
| imbalanced-learn | 0.14.0 | SMOTE-Tomek resampling |
| tensorflow / keras | 2.20.0 / 3.11.3 | Neural network |
| xgboost | 3.0.5 | Available but not the primary model |
| shap | 0.48.0 | Model explainability |
| matplotlib | 3.10.6 | Plotting |
| seaborn | 0.13.2 | Statistical visualisation |
| plotly | 6.3.0 | Interactive charts |
| scikeras | 0.13.0 | Keras–scikit-learn wrapper |

Full pinned list: `requirements.txt`

---

## Git Conventions

- **Default branch**: `master`
- **Feature branches**: use `claude/<description>` prefix for AI-assisted work
- Commit messages are plain English, imperative mood
- `.gitignore` covers: `.venv/`, `.env`, `__pycache__/`, `.ipynb_checkpoints/`, `out/`, `data/`, `models/`, `artifacts/`, `.DS_Store`
- Line endings: auto-normalised to LF via `.gitattributes`
