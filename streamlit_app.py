import sys
import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from backend.src.inferences.hybrid_decision          import predict_hybrid
from backend.src.inferences.automl_proxy             import predict_proxy
from backend.src.evaluations.error_severity_dynamic  import compute_error_severity

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="Hybrid Loan Risk Decision System",
    layout="centered",
)

# =========================================================
# Sidebar Controls
# =========================================================
st.sidebar.title("Model Controls")

run_nlp = st.sidebar.checkbox(
    "Enable NLP analysis (Explainable)",
    value=True,
)

use_vertex = st.sidebar.checkbox(
    "Enable Vertex AI (Live Prediction)",
    value=False,
    help="Uses deployed Vertex AI AutoML endpoint (paid)",
)

st.sidebar.divider()

st.sidebar.markdown("### Active Models")
st.sidebar.write("NLP:", "ON" if run_nlp else "OFF")
st.sidebar.write(
    "AutoML:",
    "Vertex AI (Live)" if use_vertex else "Proxy (Offline)",
)

st.sidebar.caption(
    "Proxy AutoML is used by default for cost control and reproducibility."
)

# =========================================================
# Main UI
# =========================================================
st.title("Hybrid Loan Risk Decision System")
st.caption(
    "Cost-aware, explainable, hybrid ML system using NLP + AutoML + Vertex AI"
)

# =========================================================
# Input Section
# =========================================================
st.subheader("Applicant Information")

text = st.text_area(
    "Loan Purpose Description",
    placeholder="e.g. Urgent loan due to medical bills and job loss",
)

income = st.number_input("Annual Income",        min_value=0.0,   step=1000.0)
credit_score = st.number_input("Credit Score",   min_value=300.0, max_value=850.0)
dti    = st.number_input(
    "Debt-to-Income Ratio", min_value=0.0, max_value=2.0, step=0.01
)
loan_amount = st.number_input("Loan Amount",     min_value=0.0,   step=1000.0)

employment_length = st.selectbox(
    "Employment Length (years)",
    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"],
)

loan_term = st.selectbox("Loan Term (months)", ["36", "60"])

# =========================================================
# Prediction Trigger
# =========================================================
if st.button("Run Risk Assessment"):

    if not text.strip():
        st.error("Loan purpose description is required.")
        st.stop()

    with st.spinner("Running hybrid decision system..."):
        result = predict_hybrid(
            text=text,
            income=income,
            credit_score=credit_score,
            dti=dti,
            loan_amount=loan_amount,
            employment_length=employment_length,
            loan_term=loan_term,
            run_nlp=run_nlp,
            use_vertex=use_vertex,
        )

    # =====================================================
    # Model Outputs
    # =====================================================
    st.subheader("Model Predictions")

    st.write("**NLP Prediction:**", result["nlp_prediction"])
    st.write(
        f"**AutoML Prediction ({result['automl_source']}):**",
        result["automl_prediction"],
    )

    if result["automl_confidence"] is not None:
        st.write("**AutoML Confidence:**", result["automl_confidence"])

    # NLP explainability
    if run_nlp and result["nlp_explanation"]:
        with st.expander("Why did NLP predict this?"):
            for signal in result["nlp_explanation"]["signals"]:
                st.markdown(f"- {signal.replace('_', ' ').title()}")

    # Vertex AI probabilities
    if use_vertex and result["automl_raw"]:
        with st.expander("Vertex AI Class Probabilities"):
            st.json(result["automl_raw"]["all_scores"])

    # =====================================================
    # Final Decision
    # =====================================================
    st.divider()
    st.subheader("Final Decision")

    if result["final_decision"] == "High Risk":
        st.error("High Risk Application")
    elif result["final_decision"] == "Medium Risk":
        st.warning("Medium Risk Application")
    else:
        st.success("Low Risk Application")

    st.write("**Decision Reason:**", result["decision_reason"])

# =========================================================
# Dynamic Error Severity Evaluation
# Cached so the 50k-row loop only runs once per session.
# =========================================================
st.divider()
st.header("Dynamic Error Severity Evaluation")


@st.cache_data
def load_severity_metrics(csv_path: str) -> dict:
    df_val = pd.read_csv(csv_path)
    y_true = df_val["loan_risk_label"]

    y_pred = [
        predict_proxy(
            row["income"],
            row["credit_score"],
            row["debt_to_income"],
            row["loan_amount"],
        )
        for _, row in df_val.iterrows()
    ]

    return compute_error_severity(y_true, y_pred)


try:
    import matplotlib.pyplot as plt

    severity = load_severity_metrics("backend/data/loan_50k_clean.csv")

    st.metric("Average Error Severity", severity["average_error_severity"])
    st.metric("Critical Errors",        severity["critical_errors"])

    fig, ax = plt.subplots()
    ax.bar(["Critical (severity \u2265 7)"], [severity["critical_errors"]])
    ax.set_title("Critical Error Count")
    st.pyplot(fig)

except Exception as e:
    st.info(f"Dynamic evaluation not available: {e}")

# =========================================================
# Footer
# =========================================================
st.caption(
    "Vertex AI predictions are optional and disabled by default to ensure "
    "cost control and reproducibility."
)
