def normalize_inputs(income, credit_score, dti, loan_amount):
    """
    Normalize raw user inputs into 0–1 range
    """

    # Income: assume 0–100,000
    income_norm = min(income / 100_000, 1.0)

    # Credit score: 300–850
    credit_norm = (credit_score - 300) / (850 - 300)
    credit_norm = min(max(credit_norm, 0), 1.0)

    # DTI: cap at 1
    dti_norm = min(dti, 1.0)

    # Loan amount: assume 0–100,000
    loan_norm = min(loan_amount / 100_000, 1.0)

    return income_norm, credit_norm, dti_norm, loan_norm
