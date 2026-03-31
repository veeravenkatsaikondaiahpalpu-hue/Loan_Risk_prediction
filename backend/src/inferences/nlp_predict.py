# google-cloud-language is imported lazily inside predict_nlp so that
# this module can be imported in offline / CI environments without the
# package installed.

# -----------------------------
# Keyword Dictionaries
# -----------------------------

DISTRESS_TERMS = [
    "missed payment", "late payment", "overdue",
    "behind on bills", "collections", "default",
    "debt", "struggling", "financial difficulty",
]

EMPLOYMENT_RISK_TERMS = [
    "laid off", "unemployed", "contract ended",
    "temporary", "freelance", "gig", "no steady income",
]

URGENCY_TERMS = [
    "urgent", "immediately", "as soon as possible",
    "emergency", "no other option",
]

SPECULATIVE_TERMS = [
    "investment", "crypto", "stocks",
    "market gains", "recover losses",
    "expect returns",
]


def predict_nlp(text: str) -> dict:
    """
    Explainable, rule-weighted NLP risk assessment.
    Uses Google Cloud Natural Language as a signal extractor.

    Returns
    -------
    dict with keys: prediction, confidence, risk_score, signals
    """
    from google.cloud import language_v1  # lazy import — avoids hard dep in CI

    client = language_v1.LanguageServiceClient()

    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT,
    )

    sentiment_score = client.analyze_sentiment(
        request={"document": document}
    ).document_sentiment.score

    entities = client.analyze_entities(
        request={"document": document}
    ).entities

    full_text   = text.lower()
    risk_score  = 0.0
    signals     = []

    if sentiment_score < -0.3:
        risk_score += 1.5
        signals.append("negative_sentiment")

    if any(term in full_text for term in DISTRESS_TERMS):
        risk_score += 2.0
        signals.append("financial_distress")

    if any(term in full_text for term in EMPLOYMENT_RISK_TERMS):
        risk_score += 1.5
        signals.append("employment_instability")

    if any(term in full_text for term in URGENCY_TERMS):
        risk_score += 1.0
        signals.append("urgency_language")

    if any(term in full_text for term in SPECULATIVE_TERMS):
        risk_score += 1.0
        signals.append("speculative_finance")

    if risk_score >= 3.0:
        prediction = "High Risk"
        confidence = 0.9
    elif risk_score >= 1.5:
        prediction = "Medium Risk"
        confidence = 0.7
    else:
        prediction = "Low Risk"
        confidence = 0.5

    return {
        "prediction": prediction,
        "confidence": confidence,
        "risk_score": round(risk_score, 2),
        "signals":    signals,
    }
