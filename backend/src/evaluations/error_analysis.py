import pandas as pd
import json
from pathlib import Path

from backend.src.evaluations.error_severity_dynamic import compute_error_severity

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_models():
    """
    Merge test labels with batch predictions, run severity analysis for both
    NLP and AutoML models, and persist final_metrics.json + error_summary.csv.
    """
    test_df   = pd.read_csv("data/processed/loan_test.csv")
    automl_df = pd.read_csv("data/processed/automl_batch_predictions.csv")
    nlp_df    = pd.read_csv("data/processed/nlp_baseline_predictions.csv")

    feature_cols = [
        "income", "credit_score", "employment_length",
        "loan_amount", "debt_to_income",
        "loan_term", "loan_purpose_text",
    ]

    merged = test_df.merge(
        automl_df,
        on=feature_cols,
        how="inner",
        suffixes=("_true", "_pred"),
    )

    y_true        = merged["loan_risk_label_true"]
    automl_preds  = merged["loan_risk_label_pred"]
    nlp_preds     = nlp_df["predicted_risk"]

    nlp_metrics    = compute_error_severity(test_df["loan_risk_label"], nlp_preds)
    automl_metrics = compute_error_severity(y_true, automl_preds)

    # ---------------------------------
    # Persist final_metrics.json
    # ---------------------------------
    final_metrics = {
        "nlp_baseline": {"error_severity": nlp_metrics["average_error_severity"]},
        "automl":        {"error_severity": automl_metrics["average_error_severity"]},
        "aligned_rows":  len(merged),
    }

    with open(RESULTS_DIR / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    # ---------------------------------
    # Persist error_summary.csv
    # ---------------------------------
    error_summary = pd.DataFrame([
        ["NLP",    nlp_metrics["average_error_severity"],    nlp_metrics["critical_errors"]],
        ["AutoML", automl_metrics["average_error_severity"], automl_metrics["critical_errors"]],
    ], columns=["model", "avg_error_severity", "critical_errors"])

    error_summary.to_csv(RESULTS_DIR / "error_summary.csv", index=False)

    return final_metrics
