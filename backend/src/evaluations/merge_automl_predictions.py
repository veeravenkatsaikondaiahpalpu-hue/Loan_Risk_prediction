import pandas as pd
from pathlib import Path

PRED_DIR = Path("data/processed/automl_raw_predictions")

files = list(PRED_DIR.glob("prediction.results-*"))

dfs = [pd.read_csv(f) for f in files]

merged = pd.concat(dfs, ignore_index=True)

merged.to_csv("data/processed/automl_batch_predictions.csv", index=False)

print("Merged predictions:", merged.shape)
