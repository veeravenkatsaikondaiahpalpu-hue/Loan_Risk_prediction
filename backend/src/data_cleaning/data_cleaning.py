import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ======================================================
# 1. Load Dataset
# ======================================================

BASE_DIR  = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "data" / "lendingclub_loan_data.xlsx"

df = pd.read_excel(DATA_PATH)
print("Initial shape:", df.shape)

# ======================================================
# 2. Select & Rename Required Columns
# ======================================================

column_mapping = {
    "annual_inc":    "income",
    "fico_range_low": "credit_score",
    "emp_length":    "employment_length",
    "loan_amnt":     "loan_amount",
    "dti":           "debt_to_income",
    "term":          "loan_term",
    "purpose":       "purpose",
}

df = df.rename(columns=column_mapping)
df = df[list(column_mapping.values())]
print("After column selection:", df.shape)

# ======================================================
# 3. Clean Employment Length
# ======================================================

def clean_employment_length(x):
    if pd.isna(x):
        return 0
    x = str(x).lower().strip()
    if "<" in x:
        return 0
    if "10+" in x:
        return 10
    return int(x.replace("years", "").replace("year", "").strip())

df["employment_length"] = df["employment_length"].apply(clean_employment_length)

# ======================================================
# 4. Clean Loan Term
# ======================================================

df["loan_term"] = (
    df["loan_term"]
    .astype(str)
    .str.replace("months", "", regex=False)
    .str.strip()
)
df["loan_term"] = pd.to_numeric(df["loan_term"], errors="coerce")
df["loan_term"] = df["loan_term"].fillna(df["loan_term"].mode()[0]).astype(int)

# ======================================================
# 5. Handle Missing Values
# ======================================================

df = df.fillna({
    "income":           df["income"].median(),
    "credit_score":     df["credit_score"].median(),
    "loan_amount":      df["loan_amount"].median(),
    "debt_to_income":   df["debt_to_income"].median(),
    "employment_length": 0,
})

# ======================================================
# 6. Create Loan Risk Labels (on RAW values)
#
# Thresholds derived from LendingClub raw ranges:
#   debt_to_income : already a ratio (0–1+)
#   credit_score   : FICO 300–850
# ======================================================

def assign_loan_risk(row):
    """
    Labels are assigned using raw feature values BEFORE any scaling
    so that the label-assignment logic is transparent and reproducible
    regardless of the scaler fitted later.
    """
    if row["debt_to_income"] > 0.35 and row["credit_score"] < 620:
        return "High Risk"
    elif row["debt_to_income"] > 0.25:
        return "Medium Risk"
    else:
        return "Low Risk"

df["loan_risk_label"] = df.apply(assign_loan_risk, axis=1)

# ======================================================
# 7. Generate Synthetic Loan Purpose Text
#    ⚠️  Text is generated from the PURPOSE column only,
#    NOT from loan_risk_label, to avoid label leakage into
#    the NLP feature.
# ======================================================

PURPOSE_PHRASES = {
    "debt_consolidation": "consolidate existing debts into one manageable payment",
    "credit_card":        "pay off outstanding credit card balances",
    "home_improvement":   "fund home improvement and renovation work",
    "other":              "cover general personal expenses",
    "major_purchase":     "finance a major planned purchase",
    "small_business":     "support small business operations",
    "car":                "purchase a vehicle",
    "medical":            "cover medical expenses",
    "moving":             "cover relocation and moving costs",
    "vacation":           "fund a planned vacation",
    "house":              "assist with housing costs",
    "wedding":            "cover wedding-related expenses",
    "educational":        "support educational costs",
    "renewable_energy":   "invest in renewable energy upgrades",
}

def generate_loan_text(row):
    purpose_phrase = PURPOSE_PHRASES.get(
        str(row["purpose"]).lower(),
        f"cover expenses related to {row['purpose']}"
    )
    return f"Applying for a loan to {purpose_phrase}."

df["loan_purpose_text"] = df.apply(generate_loan_text, axis=1)

# ======================================================
# 8. Train / Validation / Test Split (BEFORE scaling)
#    Split first so the scaler never sees val/test stats.
# ======================================================

final_columns = [
    "income", "credit_score", "employment_length",
    "loan_amount", "debt_to_income", "loan_term",
    "loan_purpose_text", "loan_risk_label",
]
df = df[final_columns]

train_df, temp_df = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    stratify=df["loan_risk_label"],
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["loan_risk_label"],
)

# ======================================================
# 9. Normalize Numeric Features (fit on TRAIN only)
#    Applying to val/test with train statistics prevents
#    data leakage from test distribution into the scaler.
# ======================================================

numeric_cols = [
    "income", "credit_score", "loan_amount",
    "debt_to_income", "employment_length",
]

scaler = MinMaxScaler()
train_df = train_df.copy()
val_df   = val_df.copy()
test_df  = test_df.copy()

train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
val_df[numeric_cols]   = scaler.transform(val_df[numeric_cols])
test_df[numeric_cols]  = scaler.transform(test_df[numeric_cols])

print("Train:", train_df.shape, "| Val:", val_df.shape, "| Test:", test_df.shape)

# ======================================================
# 10. Save Clean Files
# ======================================================

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUTPUT_DIR / "loan_train.csv", index=False)
val_df.to_csv(OUTPUT_DIR  / "loan_val.csv",   index=False)
test_df.to_csv(OUTPUT_DIR / "loan_test.csv",  index=False)

print("Data cleaning completed successfully.")
