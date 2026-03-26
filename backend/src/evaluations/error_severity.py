import pandas as pd
import json
from pathlib import Path

from backend.src.evaluations.error_severity_dynamic import compute_error_severity

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_severity():
    """
    Load saved prediction CSVs, compute error severity for NLP and AutoML,
    persist results to data/results/error_severity.json, and return the dict.
    """
    test_df = pd.read_csv("data/processed/loan_test.csv")
    y_true = test_df["loan_risk_label"]

    nlp_preds = pd.read_csv(
        "data/processed/nlp_baseline_predictions.csv"
    )["predicted_risk"]

    automl_preds = pd.read_csv(
        "data/processed/automl_batch_predictions.csv"
    )["loan_risk_label"]

    nlp_metrics    = compute_error_severity(y_true, nlp_preds)
    automl_metrics = compute_error_severity(y_true, automl_preds)

    results = {
        "NLP Baseline": nlp_metrics,
        "AutoML":        automl_metrics,
    }

    with open(RESULTS_DIR / "error_severity.json", "w") as f:
        json.dump(results, f, indent=4)

    return results
