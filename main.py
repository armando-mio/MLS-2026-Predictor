"""
MLS 2026 Predictor — CLI Entrypoint

Usage:
    python main.py --train       # Train RandomForest on 2022-2024, test on 2025
    python main.py --streamlit   # Launch Streamlit dashboard
"""
import os
import sys
import logging

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mls_predictor")


import numpy as np
import pandas as pd

from mls_predictor.data_loader import load_raw_data
from mls_predictor.elo import compute_elo_history
from mls_predictor.feature_engine import build_all_features, get_feature_columns
from mls_predictor.model_utils import (
    temporal_train_test_split, prepare_features,
    train_model, evaluate_model,
    save_model, save_artifacts,
)


def train_and_evaluate():
    """Full training pipeline: RandomForest on 2022-2024, tested on 2025."""
    print("\n══════════════════════════════════════")
    print("  MLS 2026 Predictor — Training Mode  ")
    print("  Train: 2022-2024 | Test: 2025       ")
    print("══════════════════════════════════════\n")

    # Load data (excludes 2026 by default)
    df = load_raw_data()
    df = compute_elo_history(df)
    df_feat = build_all_features(df)

    feature_cols = get_feature_columns()["all"]

    # Train for each target
    for target, target_col in [("1x2", "target_1x2"), ("ou25", "target_ou25"), ("ggng", "target_ggng")]:
        print(f"\n─── Training: {target.upper()} ───")
        split = temporal_train_test_split(df_feat, target_col, feature_cols)

        X_train_imp, X_test_imp, imputer, scaler = prepare_features(
            split["X_train"], split["X_test"]
        )

        model = train_model(X_train_imp, split["y_train"].values, target)
        metrics = evaluate_model(model, X_test_imp, split["y_test"].values, target)
        print(f"  Accuracy: {metrics['accuracy']*100:.1f}% | Log-Loss: {metrics['log_loss']:.4f}")

        # Save
        save_model(model, "random_forest", target)
        save_artifacts(imputer, scaler, split["feature_cols"], target)

    print("\n✅ All models trained and saved to models/")


if __name__ == "__main__":
    if "--streamlit" in sys.argv:
        os.system("streamlit run streamlit_app/app.py")
        sys.exit(0)

    if "--train" in sys.argv:
        train_and_evaluate()
        sys.exit(0)

    print("\nMLS 2026 Predictor v3.0")
    print("Usage:")
    print("  python main.py --train       # Train models")
    print("  python main.py --streamlit   # Launch dashboard")
