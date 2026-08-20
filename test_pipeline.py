"""Quick test of the full feature pipeline."""
import logging
logging.basicConfig(level=logging.INFO)

from mls_predictor.data_loader import load_raw_data
from mls_predictor.elo import compute_elo_history
from mls_predictor.feature_engine import build_all_features

df = load_raw_data()
df = compute_elo_history(df)
df_feat = build_all_features(df)

print(f"\nFinal dataset: {len(df_feat)} rows x {len(df_feat.columns)} columns")

meta_cols = {"Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season"}
feature_list = sorted([c for c in df_feat.columns if c not in meta_cols])
print(f"\nAll {len(feature_list)} feature columns:")
for c in feature_list:
    print(f"  - {c}")
