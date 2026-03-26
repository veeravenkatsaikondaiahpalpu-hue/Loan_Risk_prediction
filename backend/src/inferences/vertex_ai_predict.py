import os

from google.cloud import aiplatform

PROJECT_ID  = os.environ.get("VERTEX_PROJECT_ID",  "prefab-surfer-481011-s6")
REGION      = os.environ.get("VERTEX_REGION",       "us-central1")
ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID",  "5611292171412963328")


def predict_vertex(feature_dict: dict) -> dict:
    """
    Send a feature dict to the deployed Vertex AI endpoint and return
    the top prediction with its confidence score.

    Raises
    ------
    RuntimeError if the API call fails or the response format is unexpected.
    """
    try:
        aiplatform.init(project=PROJECT_ID, location=REGION)
        endpoint = aiplatform.Endpoint(ENDPOINT_ID)
        response = endpoint.predict(instances=[feature_dict])
    except Exception as exc:
        raise RuntimeError(f"Vertex AI prediction failed: {exc}") from exc

    try:
        pred    = response.predictions[0]
        classes = pred["classes"]
        scores  = pred["scores"]
        idx     = scores.index(max(scores))
    except (IndexError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Vertex AI response format: {exc}") from exc

    return {
        "prediction": classes[idx],
        "confidence": round(scores[idx], 3),
        "all_scores": dict(zip(classes, scores)),
        "source":     "Vertex AI",
    }
