"""
Unit tests for backend/src/inferences/normalization.py
"""

import pytest
from backend.src.inferences.normalization import normalize_inputs


class TestNormalizeInputs:

    # --------------------------------------------------
    # Happy-path: typical applicant values
    # --------------------------------------------------

    def test_typical_values_in_range(self):
        income_n, credit_n, dti_n, loan_n = normalize_inputs(
            income=50_000,
            credit_score=650,
            dti=0.30,
            loan_amount=20_000,
        )
        assert 0.0 <= income_n  <= 1.0
        assert 0.0 <= credit_n  <= 1.0
        assert 0.0 <= dti_n     <= 1.0
        assert 0.0 <= loan_n    <= 1.0

    def test_income_scaling(self):
        income_n, *_ = normalize_inputs(50_000, 650, 0.30, 20_000)
        assert income_n == pytest.approx(0.5, rel=1e-6)

    def test_credit_score_min(self):
        _, credit_n, *_ = normalize_inputs(50_000, 300, 0.30, 20_000)
        assert credit_n == pytest.approx(0.0, abs=1e-9)

    def test_credit_score_max(self):
        _, credit_n, *_ = normalize_inputs(50_000, 850, 0.30, 20_000)
        assert credit_n == pytest.approx(1.0, rel=1e-6)

    def test_dti_passthrough_below_one(self):
        *_, dti_n, _ = normalize_inputs(50_000, 650, 0.75, 20_000)
        assert dti_n == pytest.approx(0.75, rel=1e-6)

    # --------------------------------------------------
    # Edge cases: boundary and extreme values
    # --------------------------------------------------

    def test_zero_income(self):
        income_n, *_ = normalize_inputs(0, 650, 0.30, 20_000)
        assert income_n == pytest.approx(0.0, abs=1e-9)

    def test_income_above_cap_clipped_to_one(self):
        income_n, *_ = normalize_inputs(999_999, 650, 0.30, 20_000)
        assert income_n == pytest.approx(1.0, rel=1e-6)

    def test_dti_above_one_clipped_to_one(self):
        *_, dti_n, _ = normalize_inputs(50_000, 650, 1.5, 20_000)
        assert dti_n == pytest.approx(1.0, rel=1e-6)

    def test_loan_amount_above_cap_clipped_to_one(self):
        *_, loan_n = normalize_inputs(50_000, 650, 0.30, 500_000)
        assert loan_n == pytest.approx(1.0, rel=1e-6)

    def test_credit_score_below_300_clipped_to_zero(self):
        _, credit_n, *_ = normalize_inputs(50_000, 100, 0.30, 20_000)
        assert credit_n == pytest.approx(0.0, abs=1e-9)

    def test_returns_four_values(self):
        result = normalize_inputs(50_000, 650, 0.30, 20_000)
        assert len(result) == 4
