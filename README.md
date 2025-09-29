# Telco Customer Churn —  Data Science Project

I provide in-depth business insight and solve the problem by analyze customer data by customer profiling, predict churn and strategy to improve retention rate.


## TL;DR
- Goal: Business insight, customer churn insight, churn prediction model and recommendation action to customer
- Best model: Neural Netowrk, Churn class accuracy with 0.2 threshold ROC‑AUC: [0.828], Precision@threshold: [0.45], Recall@threshold: [0.93]
- Churn Key drivers(model interpratation): [Tenure, Month-to-month Contract type, Fiber Optic]
- Business impact: At threshold T, expected [e.g., +X% recall of churners] → estimated [e.g., Y customers retained/month] given intervention cost/benefit assumptions.

## Project Assets
- Notebook: [AnalyticsAndModelling.ipynb](https://github.com/ravsssh/TelcoCustomerChurn/blob/main/AnalyticsAndModelling.ipynb)
  - [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ravsssh/TelcoCustomerChurn/blob/main/AnalyticsAndModelling.ipynb)
  - [View in NBViewer](https://nbviewer.org/github/ravsssh/TelcoCustomerChurn/blob/main/AnalyticsAndModelling.ipynb)
- Slides (Stakeholder Summary):
  - PDF: [English (PDF)](https://github.com/ravsssh/TelcoCustomerChurn/blob/main/Presentation%20Deck%5BEN%5D.pdf) · [Bahasa (PDF)](https://github.com/ravsssh/TelcoCustomerChurn/blob/main/Presentation%20Deck%5BID%5D.pdf)

## Problem Statement
Customer churn directly impacts recurring revenue. The objective is to predict churn probability per customer and guide targeted retention actions under a cost/benefit framework.

## Data
- Source: [[cite dataset source](https://www.ibm.com/docs/en/cognos-analytics/12.1.0?topic=samples-telco-customer-churn), IBM Telco Customer Churn]
- Rows/columns: [7043 rows, 21 columns]
- Target: `Churn` (Yes/No)

## Methodology
1. EDA
2. Feature engineering
3. Modeling
4. Evaluation
5. Interpretability
6. Business decisioning

## Reproducibility
- Python: [version]
- Quickstart:
  ```bash
  # Option A: pip
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Run:
  ```bash
  # Execute notebook non-interactively
  pip install papermill
  papermill notebooks/AnalyticsAndModelling.ipynb out/AnalyticsAndModelling.output.ipynb
  ```


## License
[MIT](LICENSE)
