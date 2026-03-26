"""
Learning-curve experiments for the Loan Risk Decision System.

Data pipeline
-------------
data_cleaning.py produces three pre-split, MinMaxScaled CSVs:
    data/processed/loan_train.csv
    data/processed/loan_val.csv
    data/processed/loan_test.csv

Because the data is ALREADY scaled (scaler fitted on train only),
we do NOT apply a second scaler here. Doing so would be redundant and
would subtly alter the 0-1 distribution without benefit.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from backend.src.evaluations.error_severity_dynamic import compute_error_severity

# ======================================================
# 1. LOAD PRE-SPLIT DATA
# ======================================================

BASE_DIR   = Path(__file__).resolve().parents[2]
TRAIN_PATH = BASE_DIR / "data" / "processed" / "loan_train.csv"
TEST_PATH  = BASE_DIR / "data" / "processed" / "loan_test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

features = [
    "income",
    "credit_score",
    "employment_length",
    "loan_amount",
    "debt_to_income",
    "loan_term",
]

X_train_full = train_df[features]
y_train_full = train_df["loan_risk_label"]

X_test = test_df[features]
y_test = test_df["loan_risk_label"]

# ======================================================
# 2. STRATIFIED SUBSAMPLING
# ======================================================

def stratified_subset(X_train, y_train, size):
    X_subset, _, y_subset, _ = train_test_split(
        X_train, y_train,
        train_size=size,
        stratify=y_train,
        random_state=42,
    )
    return X_subset, y_subset

# ======================================================
# 3. MODELS
# ======================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
}

train_sizes      = [5000, 10000, 20000, 35000]
learning_results = {name: [] for name in models}

# ======================================================
# 4. LEARNING CURVE (STRATIFIED)
# ======================================================

for size in train_sizes:
    print(f"\nTraining size: {size}")
    X_subset, y_subset = stratified_subset(X_train_full, y_train_full, size)

    for name, model in models.items():
        model.fit(X_subset, y_subset)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        sev = compute_error_severity(y_test, preds)["average_error_severity"]

        learning_results[name].append((size, acc, sev))
        print(f"  {name} | Accuracy: {acc:.4f} | Severity: {sev:.4f}")

# ======================================================
# 5. PLOT LEARNING CURVES
# ======================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

plt.figure()
for name in models:
    sizes      = [r[0] for r in learning_results[name]]
    accuracies = [r[1] for r in learning_results[name]]
    plt.plot(sizes, accuracies, label=name)
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve — Accuracy")
plt.legend()
plt.savefig(BASE_DIR / "data" / f"learning_curve_accuracy_{timestamp}.png")
plt.show()

plt.figure()
for name in models:
    sizes      = [r[0] for r in learning_results[name]]
    severities = [r[2] for r in learning_results[name]]
    plt.plot(sizes, severities, label=name)
plt.xlabel("Training Size")
plt.ylabel("Average Error Severity")
plt.title("Learning Curve — Error Severity")
plt.legend()
plt.savefig(BASE_DIR / "data" / f"learning_curve_severity_{timestamp}.png")
plt.show()

# ======================================================
# 6. FEATURE IMPORTANCE (TREE MODELS) — FULL TRAINING SET
# ======================================================

rf = models["Random Forest"]
gb = models["Gradient Boosting"]

rf.fit(X_train_full, y_train_full)
gb.fit(X_train_full, y_train_full)

plt.figure()
plt.barh(features, rf.feature_importances_)
plt.title("Feature Importance — Random Forest")
plt.savefig(BASE_DIR / "data" / f"rf_feature_importance_{timestamp}.png")
plt.show()

plt.figure()
plt.barh(features, gb.feature_importances_)
plt.title("Feature Importance — Gradient Boosting")
plt.savefig(BASE_DIR / "data" / f"gb_feature_importance_{timestamp}.png")
plt.show()

print("\nAll experiments completed. Plots saved in data folder.")
