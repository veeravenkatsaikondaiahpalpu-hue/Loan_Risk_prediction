"""
Integration tests for backend/src/inferences/hybrid_decision.py

Google Cloud NLP and Vertex AI are mocked so these tests run fully
offline without credentials.
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.src.inferences.hybrid_decision import predict_hybrid, NLP_CONFIDENCE_THRESHOLD

VALID_LABELS = {"Low Risk", "Medium Risk", "High Risk"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_INPUTS = dict(
    text="Need a loan for home improvement.",
    income=60_000,
    credit_score=700,
    dti=0.20,
    loan_amount=15_000,
    employment_length="5",
    loan_term="36",
)

def _make_nlp_output(prediction="Low Risk", confidence=0.9, signals=None):
    return {
        "prediction": prediction,
        "confidence": confidence,
        "risk_score": 1.0,
        "signals":    signals or [],
    }


# ---------------------------------------------------------------------------
# Tests: NLP disabled (default)
# ---------------------------------------------------------------------------

class TestPredictHybridNLPOff:

    def test_returns_dict_with_required_keys(self):
        result = predict_hybrid(**BASE_INPUTS, run_nlp=False)
        for key in (
            "nlp_prediction", "nlp_explanation",
            "automl_prediction", "automl_confidence",
            "automl_source", "automl_raw",
            "final_decision", "decision_reason",
        ):
            assert key in result, f"Missing key: {key}"

    def test_final_decision_is_valid_label(self):
        result = predict_hybrid(**BASE_INPUTS, run_nlp=False)
        assert result["final_decision"] in VALID_LABELS

    def test_automl_source_is_proxy_when_vertex_disabled(self):
        result = predict_hybrid(**BASE_INPUTS, run_nlp=False, use_vertex=False)
        assert result["automl_source"] == "Proxy"

    def test_nlp_explanation_is_none_when_nlp_off(self):
        result = predict_hybrid(**BASE_INPUTS, run_nlp=False)
        assert result["nlp_explanation"] is None

    def test_automl_confidence_is_none_for_proxy(self):
        result = predict_hybrid(**BASE_INPUTS, run_nlp=False, use_vertex=False)
        assert result["automl_confidence"] is None

    def test_decision_reason_is_automl_override_when_nlp_off(self):
        result = predict_hybrid(**BASE_INPUTS, run_nlp=False)
        assert result["decision_reason"] == "AutoML override"


# ---------------------------------------------------------------------------
# Tests: NLP enabled (mocked)
# ---------------------------------------------------------------------------

class TestPredictHybridNLPOn:

    @patch("backend.src.inferences.hybrid_decision.predict_nlp")
    def test_high_confidence_nlp_wins(self, mock_nlp):
        """When NLP confidence >= threshold, NLP result should be the final decision."""
        mock_nlp.return_value = _make_nlp_output("High Risk", NLP_CONFIDENCE_THRESHOLD)

        result = predict_hybrid(**BASE_INPUTS, run_nlp=True, use_vertex=False)

        assert result["final_decision"]  == "High Risk"
        assert result["decision_reason"] == "High NLP confidence"

    @patch("backend.src.inferences.hybrid_decision.predict_nlp")
    def test_low_confidence_nlp_defers_to_automl(self, mock_nlp):
        """When NLP confidence < threshold, AutoML result should win."""
        mock_nlp.return_value = _make_nlp_output("High Risk", NLP_CONFIDENCE_THRESHOLD - 0.01)

        result = predict_hybrid(**BASE_INPUTS, run_nlp=True, use_vertex=False)

        assert result["decision_reason"]  == "AutoML override"
        assert result["final_decision"]   == result["automl_prediction"]

    @patch("backend.src.inferences.hybrid_decision.predict_nlp")
    def test_nlp_explanation_populated_when_nlp_on(self, mock_nlp):
        mock_nlp.return_value = _make_nlp_output(signals=["urgency_language"])

        result = predict_hybrid(**BASE_INPUTS, run_nlp=True)

        assert result["nlp_explanation"] is not None
        assert "signals" in result["nlp_explanation"]


# ---------------------------------------------------------------------------
# Tests: Vertex AI enabled (mocked)
# ---------------------------------------------------------------------------

class TestPredictHybridVertexOn:

    @patch("backend.src.inferences.hybrid_decision.predict_vertex")
    def test_vertex_source_label(self, mock_vertex):
        mock_vertex.return_value = {
            "prediction": "Medium Risk",
            "confidence": 0.88,
            "all_scores": {"Low Risk": 0.05, "Medium Risk": 0.88, "High Risk": 0.07},
            "source":     "Vertex AI",
        }

        result = predict_hybrid(**BASE_INPUTS, run_nlp=False, use_vertex=True)

        assert result["automl_source"]     == "Vertex AI"
        assert result["automl_prediction"] == "Medium Risk"
        assert result["automl_confidence"] == 0.88

    @patch("backend.src.inferences.hybrid_decision.predict_vertex")
    def test_vertex_raw_output_preserved(self, mock_vertex):
        vertex_out = {
            "prediction": "Low Risk",
            "confidence": 0.95,
            "all_scores": {"Low Risk": 0.95, "Medium Risk": 0.03, "High Risk": 0.02},
            "source":     "Vertex AI",
        }
        mock_vertex.return_value = vertex_out

        result = predict_hybrid(**BASE_INPUTS, run_nlp=False, use_vertex=True)

        assert result["automl_raw"] == vertex_out


# ---------------------------------------------------------------------------
# Tests: Input validation / edge cases
# ---------------------------------------------------------------------------

class TestPredictHybridEdgeCases:

    def test_zero_income_does_not_crash(self):
        """Zero income should be handled gracefully (normalised to 0.0)."""
        result = predict_hybrid(
            **{**BASE_INPUTS, "income": 0},
            run_nlp=False,
        )
        assert result["final_decision"] in VALID_LABELS

    def test_maximum_credit_score_does_not_crash(self):
        result = predict_hybrid(
            **{**BASE_INPUTS, "credit_score": 850},
            run_nlp=False,
        )
        assert result["final_decision"] in VALID_LABELS

    def test_high_dti_yields_high_risk_via_proxy(self):
        """DTI of 0.95 (normalised from a raw ratio) should push proxy to High Risk."""
        result = predict_hybrid(
            **{**BASE_INPUTS, "dti": 95_000 / 100_000},  # 0.95 after normalisation cap
            run_nlp=False,
            use_vertex=False,
        )
        assert result["automl_prediction"] == "High Risk"
