# Canonical severity module — single source of truth for all evaluation code.
#
# Scale: 0–10 asymmetric.
# Under-predicting risk is penalised more heavily than over-predicting,
# because approving a high-risk loan is far costlier than rejecting a good one.

SEVERITY_MAP = {
    # Under-predicting risk (most dangerous)
    ("High Risk",   "Low Risk"):    10,  # Critical: missed high-risk borrower
    ("High Risk",   "Medium Risk"):  7,  # Severe:   under-estimated high risk
    ("Medium Risk", "Low Risk"):     5,  # Moderate: missed medium-risk signal
    # Over-predicting risk (costly for applicants / business)
    ("Low Risk",    "High Risk"):    4,  # False rejection — severe
    ("Medium Risk", "High Risk"):    3,  # Over-estimated medium risk
    ("Low Risk",    "Medium Risk"):  2,  # Minor over-caution
}

CRITICAL_THRESHOLD = 7  # severity >= this value counts as a critical error


def compute_error_severity(y_true, y_pred):
    """
    Compute average weighted error severity and critical error count.

    Parameters
    ----------
    y_true : iterable of str  — ground-truth risk labels
    y_pred : iterable of str  — predicted risk labels

    Returns
    -------
    dict with keys:
        average_error_severity : float
        critical_errors        : int
    """
    total_severity = 0
    critical_errors = 0

    pairs = list(zip(y_true, y_pred))
    if not pairs:
        return {"average_error_severity": 0.0, "critical_errors": 0}

    for true, pred in pairs:
        if true != pred:
            severity = SEVERITY_MAP.get((true, pred), 1)
            total_severity += severity
            if severity >= CRITICAL_THRESHOLD:
                critical_errors += 1

    avg_severity = total_severity / len(pairs)

    return {
        "average_error_severity": round(avg_severity, 3),
        "critical_errors": critical_errors,
    }
