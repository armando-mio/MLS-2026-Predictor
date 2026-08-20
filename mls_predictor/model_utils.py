"""
Model training, evaluation, and prediction — simplified to RandomForest.

Follows the Dataquest project-walkthroughs workflow:
  - Single RandomForestClassifier
  - Temporal train/test split (2022-2024 train, 2025 test)
  - Simple predict_proba (no ensemble)
"""
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from mls_predictor.config import MODELS_DIR, TEST_SEASON

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Model Definition
# ═══════════════════════════════════════════════════════════════════════════

def create_model() -> RandomForestClassifier:
    """Create a regularized RandomForest classifier for soccer outcome prediction."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=8,
        min_samples_split=16,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Preprocessing (Imputation + Scaling)
# ═══════════════════════════════════════════════════════════════════════════

def prepare_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer, StandardScaler]:
    """Impute NaNs with median. Scaler fitted for reference but RF uses unscaled.

    Returns DataFrames (preserving column names) plus fitted imputer and scaler.
    """
    columns = X_train.columns.tolist()

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=columns, index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=columns, index=X_test.index
    )

    # Scaler kept for potential future use but RF doesn't need it
    scaler = StandardScaler()
    scaler.fit(X_train_imp)

    return X_train_imp, X_test_imp, imputer, scaler


# ═══════════════════════════════════════════════════════════════════════════
#  Time-Series Split
# ═══════════════════════════════════════════════════════════════════════════

def temporal_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    test_season: int = TEST_SEASON,
) -> dict:
    """
    Split data respecting temporal order.
    Train on all seasons before test_season (2022-2024).
    Test on test_season (2025).
    """
    train_df = df[df["Season"] < test_season].copy()
    test_df = df[df["Season"] == test_season].copy()

    # If test set is empty, fallback to chronological 80/20
    if len(test_df) == 0:
        logger.warning("No data for season %d, falling back to 80/20 split", test_season)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

    # Filter to only existing feature columns
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("Missing feature columns (will be skipped): %s", missing)

    result = {
        "X_train": train_df[available],
        "y_train": train_df[target_col],
        "X_test": test_df[available],
        "y_test": test_df[target_col],
        "feature_cols": available,
        "train_dates": train_df["Date"],
        "test_dates": test_df["Date"],
        "test_df": test_df,
    }

    logger.info("Split: Train=%d (seasons < %d), Test=%d (season %d) | Features=%d",
                len(train_df), test_season, len(test_df), test_season, len(available))
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    target_name: str = "1x2",
) -> RandomForestClassifier:
    """Train a RandomForest classifier on the given data."""
    model = create_model()
    logger.info("Training RandomForest for target '%s' on %d samples …",
                target_name, len(y_train))
    model.fit(X_train, y_train)
    logger.info("  ✓ RandomForest trained (n_estimators=%d, max_depth=%d)",
                model.n_estimators, model.max_depth)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    target_name: str = "1x2",
) -> dict:
    """Evaluate the model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    try:
        ll = log_loss(y_test, y_proba, labels=sorted(np.unique(y_test)))
    except Exception:
        ll = np.nan
    try:
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    except Exception:
        prec = np.nan

    metrics = {
        "model": "random_forest",
        "target": target_name,
        "accuracy": acc,
        "log_loss": ll,
        "precision": prec,
    }
    logger.info("  RF — Acc: %.4f | LogLoss: %.4f | Precision: %.4f",
                acc, ll, prec)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  Save / Load
# ═══════════════════════════════════════════════════════════════════════════

def save_model(model, name: str = "random_forest", target: str = "1x2") -> None:
    """Save a model to disk."""
    path = MODELS_DIR / f"{name}_{target}.joblib"
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)


def load_model(name: str = "random_forest", target: str = "1x2"):
    """Load a model from disk."""
    path = MODELS_DIR / f"{name}_{target}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def save_artifacts(
    imputer, scaler, feature_cols: list[str], target: str = "1x2"
) -> None:
    """Save preprocessing artifacts."""
    path = MODELS_DIR / f"preprocess_{target}.joblib"
    joblib.dump({"imputer": imputer, "scaler": scaler, "feature_cols": feature_cols}, path)
    logger.info("Preprocessing artifacts saved → %s", path)


def load_artifacts(target: str = "1x2") -> dict:
    """Load preprocessing artifacts."""
    path = MODELS_DIR / f"preprocess_{target}.joblib"
    return joblib.load(path)
