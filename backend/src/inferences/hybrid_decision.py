from backend.src.inferences.vertex_ai_predict import predict_vertex
from backend.src.inferences.automl_proxy     import predict_proxy
from backend.src.inferences.nlp_predict      import predict_nlp
from backend.src.inferences.normalization    import normalize_inputs

NLP_CONFIDENCE_THRESHOLD = 0.8


def predict_hybrid(
    text,
    income,
    credit_score,
    dti,
    loan_amount,
    employment_length,
    loan_term,
    run_nlp=False,
    use_vertex=False,
):
    """
    Orchestrate NLP + AutoML signals into a single risk decision.

    Decision logic
    --------------
    - If NLP is enabled and its confidence >= NLP_CONFIDENCE_THRESHOLD,
      the NLP prediction wins.
    - Otherwise the AutoML (proxy or Vertex) prediction is used.
    """
    # ---------------------------
    # Normalize numeric inputs
    # ---------------------------
    income_n, credit_n, dti_n, loan_n = normalize_inputs(
        income, credit_score, dti, loan_amount
    )

    # ---------------------------
    # NLP branch
    # ---------------------------
    if run_nlp:
        nlp_out  = predict_nlp(text)
        nlp_pred = nlp_out["prediction"]
        nlp_conf = nlp_out["confidence"]
    else:
        nlp_out  = None
        nlp_pred = "Low Risk"
        nlp_conf = 0.0

    # ---------------------------
    # AutoML branch
    # ---------------------------
    if use_vertex:
        vertex_features = {
            "income":            str(income_n),
            "credit_score":      str(credit_n),
            "debt_to_income":    str(dti_n),
            "loan_amount":       str(loan_n),
            "employment_length": str(employment_length),
            "loan_term":         str(loan_term),
            "loan_purpose_text": text,
        }
        automl_out    = predict_vertex(vertex_features)
        automl_pred   = automl_out["prediction"]
        automl_conf   = automl_out["confidence"]
        automl_source = "Vertex AI"
    else:
        automl_pred   = predict_proxy(income_n, credit_n, dti_n, loan_n)
        automl_conf   = None
        automl_out    = None
        automl_source = "Proxy"

    # ---------------------------
    # Final decision
    # ---------------------------
    if run_nlp and nlp_conf >= NLP_CONFIDENCE_THRESHOLD:
        final_decision  = nlp_pred
        decision_reason = "High NLP confidence"
    else:
        final_decision  = automl_pred
        decision_reason = "AutoML override"

    return {
        "nlp_prediction":    nlp_pred,
        "nlp_explanation":   nlp_out,
        "automl_prediction": automl_pred,
        "automl_confidence": automl_conf,
        "automl_source":     automl_source,
        "automl_raw":        automl_out,
        "final_decision":    final_decision,
        "decision_reason":   decision_reason,
    }
