# app.py
# ==========================================
# Loan Risk Prediction & Error Severity App
# ==========================================

from backend.src.inferences.hybrid_decision import predict_hybrid
from backend.src.evaluations.error_severity  import evaluate_severity

# --------------------------
# DEMO-SAFE FLAGS
# --------------------------
RUN_PREDICTION      = True
RUN_ERROR_SEVERITY  = True

# Keep NLP OFF for demo stability (requires live Google Cloud credentials)
RUN_NLP = False


def run_prediction():
    print("\n=== Loan Risk Prediction ===\n")

    text              = input("Loan purpose text: ")
    income            = float(input("Annual income (e.g. 45000): "))
    credit_score      = float(input("Credit score (300–850): "))
    dti               = float(input("Debt-to-income ratio (e.g. 0.86): "))
    loan_amount       = float(input("Requested loan amount (e.g. 30000): "))
    employment_length = input("Employment length in years (e.g. 3 or 10+): ")
    loan_term         = input("Loan term in months (36 or 60): ")

    result = predict_hybrid(
        text=text,
        income=income,
        credit_score=credit_score,
        dti=dti,
        loan_amount=loan_amount,
        employment_length=employment_length,
        loan_term=loan_term,
        run_nlp=RUN_NLP,
    )

    print("\nResult")
    print("------")
    for k, v in result.items():
        print(f"{k}: {v}")


def run_severity_evaluation():
    print("\n=== Error Severity Evaluation ===\n")

    severity_results = evaluate_severity()

    print("Error Severity Results")
    print("----------------------")
    for model, metrics in severity_results.items():
        print(f"{model}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    print("\nResults saved to: data/results/error_severity.json")


def main():
    print("\n===================================")
    print(" LOAN RISK DECISION SYSTEM (FINAL) ")
    print("===================================\n")

    if RUN_PREDICTION:
        run_prediction()

    if RUN_ERROR_SEVERITY:
        run_severity_evaluation()

    print("\nExecution completed successfully.\n")


if __name__ == "__main__":
    main()
