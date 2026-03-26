import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ======================================================
# 1. LOAD FULL LENDINGCLUB DATA
# ======================================================

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "original data.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

print("Original dataset shape:", df.shape)

# ======================================================
# 2. SELECT REQUIRED COLUMNS ONLY
# ======================================================

column_mapping = {
    "annual_inc": "income",
    "fico_range_low": "credit_score",
    "emp_length": "employment_length",
    "loan_amnt": "loan_amount",
    "dti": "debt_to_income",
    "term": "loan_term",
    "purpose": "purpose"
}

df = df[list(column_mapping.keys())]
df = df.rename(columns=column_mapping)

print("After column selection:", df.shape)

# ======================================================
# 3. DROP ROWS WITH MISSING VALUES
# ======================================================

required_cols = list(column_mapping.values())
df = df.dropna(subset=required_cols)

print("After dropping missing values:", df.shape)

# ======================================================
# 4. CLEAN EMPLOYMENT LENGTH
# ======================================================

def clean_employment_length(x):
    x = str(x).lower().strip()
    if "<" in x:
        return 0
    if "10+" in x:
        return 10
    try:
        return int(x.replace("years", "").replace("year", "").strip())
    except:
        return 0

df["employment_length"] = df["employment_length"].apply(clean_employment_length)

# ======================================================
# 5. CLEAN LOAN TERM
# ======================================================

df["loan_term"] = (
    df["loan_term"]
    .astype(str)
    .str.replace("months", "", regex=False)
    .str.strip()
)

df["loan_term"] = pd.to_numeric(df["loan_term"], errors="coerce")
df = df.dropna(subset=["loan_term"])
df["loan_term"] = df["loan_term"].astype(int)

# ======================================================
# 6. RANDOM SAMPLE 50,000 ROWS
# ======================================================

df = df.sample(n=50000, random_state=42)

print("Sampled dataset shape:", df.shape)

# ======================================================
# 7. CONVERT DTI FROM % TO RATIO
# ======================================================

df["debt_to_income"] = df["debt_to_income"] / 100

# ======================================================
# 8. NORMALIZE NUMERIC FEATURES (0–1)
# ======================================================

numeric_cols = [
    "income",
    "credit_score",
    "loan_amount",
    "debt_to_income",
    "employment_length",
    "loan_term"
]

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ======================================================
# 9. CREATE LOAN RISK LABEL
# ======================================================

def assign_loan_risk(row):
    score = 0

    if row["debt_to_income"] > 0.5:
        score += 2
    elif row["debt_to_income"] > 0.35:
        score += 1

    if row["credit_score"] < 0.4:
        score += 2
    elif row["credit_score"] < 0.6:
        score += 1

    if score >= 3:
        return "High Risk"
    elif score == 2:
        return "Medium Risk"
    else:
        return "Low Risk"

df["loan_risk_label"] = df.apply(assign_loan_risk, axis=1)

# ======================================================
# 10. FINAL COLUMN ORDER
# ======================================================

final_columns = numeric_cols + ["loan_risk_label"]
df = df[final_columns]

# ======================================================
# 11. SAVE CLEAN DATASET
# ======================================================

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "loan_50k_clean.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved cleaned dataset to:", OUTPUT_PATH)
print("\nClass distribution:")
print(df["loan_risk_label"].value_counts())
