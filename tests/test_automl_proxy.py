"""
Unit tests for backend/src/inferences/automl_proxy.py

All inputs are normalised to the 0-1 range as the function requires.
"""

import pytest
from backend.src.inferences.automl_proxy import predict_proxy

VALID_LABELS = {"Low Risk", "Medium Risk", "High Risk"}


class TestPredictProxy:

    # --------------------------------------------------
    # Happy-path: High Risk branches
    # --------------------------------------------------

    def test_high_risk_high_dti(self):
        """DTI >= 0.6 always returns High Risk regardless of other inputs."""
        result = predict_proxy(
            income=0.8, credit_score=0.9, debt_to_income=0.6, loan_amount=0.1
        )
        assert result == "High Risk"

    def test_high_risk_dti_exactly_threshold(self):
        result = predict_proxy(
            income=0.5, credit_score=0.5, debt_to_income=0.6, loan_amount=0.5
        )
        assert result == "High Risk"

    def test_high_risk_low_credit_high_loan(self):
        """credit_score < 0.45 AND loan_amount > 0.5 → High Risk."""
        result = predict_proxy(
            income=0.5, credit_score=0.44, debt_to_income=0.3, loan_amount=0.51
        )
        assert result == "High Risk"

    # --------------------------------------------------
    # Happy-path: Medium Risk branch
    # --------------------------------------------------

    def test_medium_risk_via_dti(self):
        """DTI > 0.4 but < 0.6 → Medium Risk."""
        result = predict_proxy(
            income=0.5, credit_score=0.7, debt_to_income=0.41, loan_amount=0.3
        )
        assert result == "Medium Risk"

    def test_medium_risk_via_loan_amount(self):
        """Loan amount > 0.4 with safe DTI → Medium Risk."""
        result = predict_proxy(
            income=0.5, credit_score=0.7, debt_to_income=0.2, loan_amount=0.41
        )
        assert result == "Medium Risk"

    # --------------------------------------------------
    # Happy-path: Low Risk branch
    # --------------------------------------------------

    def test_low_risk_all_safe(self):
        result = predict_proxy(
            income=0.8, credit_score=0.9, debt_to_income=0.2, loan_amount=0.1
        )
        assert result == "Low Risk"

    def test_low_risk_boundary_values(self):
        """Just under every Medium-Risk threshold."""
        result = predict_proxy(
            income=0.5, credit_score=0.5, debt_to_income=0.4, loan_amount=0.4
        )
        assert result == "Low Risk"

    # --------------------------------------------------
    # Output contract
    # --------------------------------------------------

    def test_output_is_always_a_valid_label(self):
        test_cases = [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5, 0.5),
            (0.8, 0.9, 0.15, 0.2),
        ]
        for income, credit, dti, loan in test_cases:
            result = predict_proxy(income, credit, dti, loan)
            assert result in VALID_LABELS, f"Unexpected label '{result}' for inputs {(income, credit, dti, loan)}"

    # --------------------------------------------------
    # Edge: DTI just below High Risk threshold
    # --------------------------------------------------

    def test_dti_just_below_high_risk(self):
        """DTI = 0.599 must NOT trigger the first High Risk branch."""
        result = predict_proxy(
            income=0.5, credit_score=0.9, debt_to_income=0.599, loan_amount=0.1
        )
        assert result == "Medium Risk"

    def test_high_risk_condition_2_boundary_credit(self):
        """credit_score == 0.45 does NOT satisfy < 0.45, so no High Risk here."""
        result = predict_proxy(
            income=0.5, credit_score=0.45, debt_to_income=0.3, loan_amount=0.51
        )
        # credit_score not < 0.45 → falls through to Medium Risk (loan > 0.4)
        assert result == "Medium Risk"
