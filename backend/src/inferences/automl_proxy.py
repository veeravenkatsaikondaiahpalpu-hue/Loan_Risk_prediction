def predict_proxy(income, credit_score, debt_to_income, loan_amount):
    """
    Rule-based offline proxy for AutoML predictions.
    All inputs must already be normalised to the 0-1 range.
    """
    if debt_to_income >= 0.6:
        return "High Risk"

    if credit_score < 0.45 and loan_amount > 0.5:
        return "High Risk"

    if debt_to_income > 0.4 or loan_amount > 0.4:
        return "Medium Risk"

    return "Low Risk"
