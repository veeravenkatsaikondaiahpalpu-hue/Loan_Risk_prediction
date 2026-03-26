from google.cloud import aiplatform

PROJECT_ID = "prefab-surfer-481011-s6"
REGION = "us-central1"
ENDPOINT_ID = "9182662618836369408"  # your endpoint id

aiplatform.init(
    project=PROJECT_ID,
    location=REGION
)

endpoint = aiplatform.Endpoint(ENDPOINT_ID)

# IMPORTANT: normalized inputs (same scale as training)
instance = {
    "income": "0.60",
    "credit_score": "0.55",
    "debt_to_income": "0.40",
    "loan_amount": "0.35",
    "employment_length": "5",
    "loan_term": "36",
    "loan_purpose_text": "Loan needed urgently for debt_consolidation. Recent employment changes and financial pressure."
}

response = endpoint.predict(instances=[instance])

print("Vertex AI response:")
print(response.predictions)
