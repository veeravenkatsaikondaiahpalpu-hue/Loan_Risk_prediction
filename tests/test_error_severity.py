"""
Unit tests for backend/src/evaluations/error_severity_dynamic.py

Tests cover:
- All six penalty pairs in SEVERITY_MAP
- Perfect predictions (zero severity)
- Critical error counting (severity >= 7)
- Edge case: empty input
- Return value shape and types
"""

import pytest
from backend.src.evaluations.error_severity_dynamic import (
    compute_error_severity,
    SEVERITY_MAP,
    CRITICAL_THRESHOLD,
)

H = "High Risk"
M = "Medium Risk"
L = "Low Risk"


class TestSeverityMap:

    def test_all_expected_pairs_present(self):
        expected_pairs = {
            (H, L), (H, M),   # under-prediction
            (M, L),
            (L, H), (M, H),   # over-prediction
            (L, M),
        }
        assert expected_pairs.issubset(set(SEVERITY_MAP.keys()))

    def test_worst_penalty_is_high_to_low(self):
        assert SEVERITY_MAP[(H, L)] == max(SEVERITY_MAP.values())

    def test_under_prediction_penalised_more_than_over(self):
        """Missing a High Risk borrower should cost more than falsely flagging one."""
        assert SEVERITY_MAP[(H, L)] > SEVERITY_MAP[(L, H)]
        assert SEVERITY_MAP[(H, M)] > SEVERITY_MAP[(M, H)]

    def test_critical_threshold_triggers_on_severe_errors(self):
        """Both H→L (10) and H→M (7) should be critical."""
        assert SEVERITY_MAP[(H, L)] >= CRITICAL_THRESHOLD
        assert SEVERITY_MAP[(H, M)] >= CRITICAL_THRESHOLD

    def test_over_prediction_is_below_critical_threshold(self):
        assert SEVERITY_MAP[(L, H)] < CRITICAL_THRESHOLD
        assert SEVERITY_MAP[(M, H)] < CRITICAL_THRESHOLD


class TestComputeErrorSeverity:

    # --------------------------------------------------
    # Perfect predictions
    # --------------------------------------------------

    def test_perfect_predictions_zero_severity(self):
        result = compute_error_severity([H, M, L], [H, M, L])
        assert result["average_error_severity"] == 0.0
        assert result["critical_errors"] == 0

    # --------------------------------------------------
    # Single known errors
    # --------------------------------------------------

    def test_single_high_to_low_error(self):
        """One H→L error in a 4-sample batch."""
        result = compute_error_severity([H, L, L, L], [L, L, L, L])
        expected_avg = SEVERITY_MAP[(H, L)] / 4
        assert result["average_error_severity"] == pytest.approx(expected_avg, rel=1e-6)
        assert result["critical_errors"] == 1

    def test_single_high_to_medium_error(self):
        result = compute_error_severity([H], [M])
        assert result["average_error_severity"] == pytest.approx(SEVERITY_MAP[(H, M)], rel=1e-6)
        assert result["critical_errors"] == 1

    def test_single_medium_to_low_error(self):
        result = compute_error_severity([M], [L])
        assert result["average_error_severity"] == pytest.approx(SEVERITY_MAP[(M, L)], rel=1e-6)
        assert result["critical_errors"] == 0

    def test_single_low_to_high_error(self):
        result = compute_error_severity([L], [H])
        assert result["average_error_severity"] == pytest.approx(SEVERITY_MAP[(L, H)], rel=1e-6)
        assert result["critical_errors"] == 0

    def test_single_medium_to_high_error(self):
        result = compute_error_severity([M], [H])
        assert result["average_error_severity"] == pytest.approx(SEVERITY_MAP[(M, H)], rel=1e-6)
        assert result["critical_errors"] == 0

    def test_single_low_to_medium_error(self):
        result = compute_error_severity([L], [M])
        assert result["average_error_severity"] == pytest.approx(SEVERITY_MAP[(L, M)], rel=1e-6)
        assert result["critical_errors"] == 0

    # --------------------------------------------------
    # Multiple errors and critical counting
    # --------------------------------------------------

    def test_multiple_critical_errors_counted(self):
        y_true = [H, H, M]
        y_pred = [L, M, L]
        result = compute_error_severity(y_true, y_pred)
        # H→L = 10 (critical), H→M = 7 (critical), M→L = 5 (not critical)
        assert result["critical_errors"] == 2

    def test_average_severity_calculation(self):
        y_true = [H, L]
        y_pred = [L, H]
        expected = (SEVERITY_MAP[(H, L)] + SEVERITY_MAP[(L, H)]) / 2
        result = compute_error_severity(y_true, y_pred)
        assert result["average_error_severity"] == pytest.approx(expected, rel=1e-6)

    # --------------------------------------------------
    # Edge cases
    # --------------------------------------------------

    def test_empty_input_returns_zero(self):
        result = compute_error_severity([], [])
        assert result["average_error_severity"] == 0.0
        assert result["critical_errors"] == 0

    def test_return_type_is_dict_with_expected_keys(self):
        result = compute_error_severity([H], [L])
        assert isinstance(result, dict)
        assert "average_error_severity" in result
        assert "critical_errors" in result

    def test_severity_rounded_to_three_decimals(self):
        """average_error_severity should be rounded to 3 decimal places."""
        result = compute_error_severity([H, M, L], [M, L, H])
        avg = result["average_error_severity"]
        assert avg == round(avg, 3)
