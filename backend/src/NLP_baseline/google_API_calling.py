import pandas as pd
from pathlib import Path
from google.cloud import language_v1
from sklearn.metrics import classification_report

# ======================================================
# STEP 5.3 — Load Test Data
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[3]
TEST_PATH = BASE_DIR / "data" / "processed" / "loan_test.csv"

df_test = pd.read_csv(TEST_PATH)

df_test = df_test[["loan_purpose_text", "loan_risk_label"]]

print("Loaded test data:", df_test.shape)

# ======================================================
# STEP 5.4 — Initialize NLP Client
# ======================================================

client = language_v1.LanguageServiceClient()

# ======================================================
# STEP 5.5 — NLP Analysis Function
# ======================================================

def analyze_text(text):
    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT
    )

    sentiment = client.analyze_sentiment(
        request={"document": document}
    ).document_sentiment

    entities = client.analyze_entities(
        request={"document": document}
    ).entities

    return sentiment.score, [e.name.lower() for e in entities]

# ======================================================
# STEP 5.6 — Risk Classification Logic
# ======================================================

def nlp_risk_classifier(text):
    sentiment_score, entities = analyze_text(text)

    risk_score = 0

    if sentiment_score < -0.3:
        risk_score += 1

    risky_entities = ["medical", "rent", "job", "emergency", "unemployed"]
    if any(ent in risky_entities for ent in entities):
        risk_score += 1

    if risk_score >= 2:
        return "High Risk"
    elif risk_score == 1:
        return "Medium Risk"
    else:
        return "Low Risk"

# ======================================================
# STEP 5.7 — Run Baseline Predictions (LIMITED FOR COST)
# ======================================================

predictions = []

for text in df_test["loan_purpose_text"].head(20):
    predictions.append(nlp_risk_classifier(text))

df_eval = df_test.head(20).copy()
df_eval["predicted_risk"] = predictions

# ======================================================
# STEP 5.8 — Evaluation
# ======================================================

print(
    classification_report(
        df_eval["loan_risk_label"],
        df_eval["predicted_risk"]
    )
)

# ======================================================
# STEP 5.9 — Save Results
# ======================================================

OUTPUT_PATH = BASE_DIR / "data" / "processed" / "nlp_baseline_predictions.csv"
df_eval.to_csv(OUTPUT_PATH, index=False)

print("✅ NLP baseline completed successfully")
