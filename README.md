# Hybrid Loan Risk Decision System

An end-to-end, cost-aware loan risk assessment system combining **Google Cloud Natural Language** (NLP), **Vertex AI AutoML**, and a rule-based offline proxy — with a Streamlit UI and a full error-severity evaluation framework.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
5. [Running the App](#running-the-app)
6. [Running Tests](#running-tests)
7. [Data Pipeline & Leakage Prevention](#data-pipeline--leakage-prevention)
8. [Error Severity Framework](#error-severity-framework)
9. [Environment Variables](#environment-variables)

---

## Project Overview

Given an applicant's financial profile and a free-text loan purpose description, the system produces a three-class risk label: **Low Risk**, **Medium Risk**, or **High Risk**.

Three inference components are blended into a single decision:

| Component | Description | Online? |
|---|---|---|
| **NLP (Google Natural Language)** | Extracts distress/speculative signals from the loan purpose text | Yes (optional) |
| **Vertex AI AutoML** | Deployed AutoML endpoint trained on LendingClub data | Yes (paid, optional) |
| **Proxy (Offline Rules)** | Threshold-based fallback using normalised numeric features | Always available |

The final decision follows a confidence-weighted hierarchy: if NLP confidence ≥ 0.8, the NLP prediction wins; otherwise the AutoML/Proxy result is used.

---

## Architecture

```
streamlit_app.py  /  backend/app.py (CLI)
        │
        ▼
backend/src/inferences/hybrid_decision.py  (predict_hybrid)
        ├── predict_nlp         ← Google NLP signal extractor
        ├── predict_vertex      ← Vertex AI AutoML endpoint
        ├── predict_proxy       ← Offline rule-based fallback
        └── normalize_inputs    ← Feature scaler (0-1 range)

backend/src/evaluations/
        ├── error_severity_dynamic.py  ← CANONICAL severity module
        ├── error_severity.py          ← Batch CSV evaluation
        └── error_analysis.py          ← Cross-model comparison

backend/src/data_cleaning/
        └── data_cleaning.py    ← Preprocessing + leak-free splitting

backend/src/training_model/
        └── main_advanced_2.0.py  ← Learning curve experiments
```

---

## Project Structure

```
ADS_project/
├── streamlit_app.py              # Streamlit web interface
├── requirements.txt
├── README.md
│
├── backend/
│   ├── app.py                    # CLI entry point
│   └── src/
│       ├── data_cleaning/
│       │   ├── data_cleaning.py          # Main preprocessing (Excel → CSVs)
│       │   └── data_for_custom_model.py  # 50k sample for evaluation
│       │
│       ├── evaluations/
│       │   ├── error_severity_dynamic.py # ← Canonical SEVERITY_MAP + compute_error_severity
│       │   ├── error_severity.py         # Batch evaluation (CSV-based)
│       │   ├── error_analysis.py         # Cross-model analysis
│       │   └── merge_automl_predictions.py
│       │
│       ├── inferences/
│       │   ├── hybrid_decision.py   # predict_hybrid — main orchestrator
│       │   ├── automl_proxy.py      # predict_proxy  — offline fallback
│       │   ├── nlp_predict.py       # predict_nlp    — Google NLP
│       │   ├── vertex_ai_predict.py # predict_vertex — Vertex AI
│       │   └── normalization.py     # normalize_inputs
│       │
│       └── training_model/
│           └── main_advanced_2.0.py # Learning curves + feature importance
│
├── tests/
│   ├── conftest.py
│   ├── test_normalization.py
│   ├── test_automl_proxy.py
│   ├── test_error_severity.py
│   └── test_hybrid_decision.py
│
└── backend/data/
    ├── loan_50k_clean.csv
    └── processed/
        ├── loan_train.csv
        ├── loan_val.csv
        └── loan_test.csv
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd ADS_project
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Google Cloud credentials (optional — only needed for NLP / Vertex AI)

```bash
gcloud auth application-default login
```

Or set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to your service-account JSON key path.

---

## Running the App

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

The sidebar lets you toggle NLP and Vertex AI on/off. The offline **Proxy** model is always available — no cloud credentials required.

### CLI

```bash
python -m backend.app
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=backend --cov-report=term-missing
```

The test suite is fully **offline** — Google Cloud NLP and Vertex AI are mocked with `unittest.mock`.

---

## Data Pipeline & Leakage Prevention

The preprocessing pipeline in `data_cleaning.py` is designed to be **leak-free**:

| Step | What happens |
|---|---|
| 1 | Raw LendingClub data loaded |
| 2 | Columns selected & renamed |
| 3 | Missing values imputed |
| 4 | Risk labels assigned from **raw** feature values |
| 5 | **Train / Val / Test split performed** |
| 6 | `MinMaxScaler` fitted **on train only**, then applied to all three sets |
| 7 | Splits saved as `loan_train.csv`, `loan_val.csv`, `loan_test.csv` |

> **Why this matters:** Fitting the scaler on the full dataset before splitting leaks test-set statistics (min/max) into the preprocessing step, making model performance appear better than it really is.

The `loan_purpose_text` field is generated from the raw `purpose` category column only — **not** from the risk label — to avoid NLP having direct access to the target variable.

---

## Error Severity Framework

Risk misclassifications are not all equal. Approving a **High Risk** loan that should have been rejected is far costlier than conservatively flagging a **Low Risk** applicant.

All evaluation code shares a single canonical severity map defined in `error_severity_dynamic.py`:

| True → Predicted | Severity | Critical? |
|---|---|---|
| High Risk → Low Risk | **10** | Yes |
| High Risk → Medium Risk | **7** | Yes |
| Medium Risk → Low Risk | 5 | No |
| Low Risk → High Risk | 4 | No |
| Medium Risk → High Risk | 3 | No |
| Low Risk → Medium Risk | 2 | No |

Critical errors are those with severity ≥ 7 (both cases where a high-risk borrower was classified as lower risk).

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VERTEX_PROJECT_ID` | `prefab-surfer-481011-s6` | Google Cloud project ID |
| `VERTEX_REGION` | `us-central1` | Vertex AI region |
| `VERTEX_ENDPOINT_ID` | `5611292171412963328` | Deployed AutoML endpoint ID |

Set these in a `.env` file or export them before running to override the defaults without modifying source code.
