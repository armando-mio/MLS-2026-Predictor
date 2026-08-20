"""
SHAP explainability helpers.
Provides functions to compute and serialise SHAP values for single-match
and global feature importance analysis.

Tested with shap==0.52.0 — values are 3D ndarrays (n_samples, n_features, n_classes).
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_shap():
    """Import shap; warn if missing."""
    try:
        import shap
        return shap
    except ImportError:
        logger.warning("shap not installed. Install with: pip install shap")
        return None


def compute_shap_values(
    model,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
    model_type: str = "tree",
) -> dict | None:
    """
    Compute SHAP values for the given data.

    Returns
    -------
    dict with keys:
        - shap_values: np.ndarray — shape (n_samples, n_features, n_classes) for multi-class
                                     or (n_samples, n_features) for binary
        - expected_value: np.ndarray — shape (n_classes,)
        - feature_names: list[str]
    """
    shap = _get_shap()
    if shap is None:
        return None

    try:
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X)

        # Use shap_values() — returns ndarray directly in shap>=0.42
        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value

        # Normalize shap_values to ndarray
        if isinstance(shap_values, list):
            # Old-style: list of (n_samples, n_features) per class → stack
            shap_values = np.stack(shap_values, axis=-1)
        # Now guaranteed: (n_samples, n_features, n_classes) or (n_samples, n_features)

        # Normalize expected_value to 1D ndarray
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = np.atleast_1d(np.asarray(expected_value).ravel())
        else:
            expected_value = np.array([float(expected_value)])

        if feature_names is None and isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        elif feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]

        return {
            "shap_values": shap_values,
            "expected_value": expected_value,
            "feature_names": feature_names,
        }
    except Exception as e:
        logger.error("SHAP computation failed: %s", e)
        return None


def single_match_shap(
    model,
    X_single: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    class_names: list[str] | None = None,
) -> dict | None:
    """
    Compute SHAP values for a single match prediction.
    Returns a dict keyed by class name, each containing sorted feature contributions.
    """
    result = compute_shap_values(model, X_single, feature_names, model_type="tree")
    if result is None:
        return None

    sv = result["shap_values"]   # (1, n_features, n_classes) or (1, n_features)
    ev = result["expected_value"]  # (n_classes,) or (1,)

    if sv.ndim == 3:
        # Multi-class: sv shape is (1, n_features, n_classes)
        n_classes = sv.shape[2]
        explanations = {}
        for c in range(n_classes):
            cls_name = class_names[c] if class_names and c < len(class_names) else f"class_{c}"
            feature_contribs = []
            for i, fname in enumerate(feature_names):
                feature_contribs.append({
                    "feature": fname,
                    "shap_value": float(sv[0, i, c]),
                })
            feature_contribs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            explanations[cls_name] = {
                "base_value": float(ev[c]) if c < len(ev) else 0.0,
                "features": feature_contribs,
            }
        return explanations
    else:
        # Binary / single output: sv shape is (1, n_features)
        feature_contribs = []
        for i, fname in enumerate(feature_names):
            feature_contribs.append({
                "feature": fname,
                "shap_value": float(sv[0, i]),
            })
        feature_contribs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return {
            "base_value": float(ev[0]),
            "features": feature_contribs,
        }


def global_feature_importance(
    model,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame | None:
    """
    Compute mean |SHAP| for each feature across all samples.
    Returns a DataFrame sorted by importance.
    """
    result = compute_shap_values(model, X, feature_names, model_type="tree")
    if result is None:
        return None

    sv = result["shap_values"]

    if sv.ndim == 3:
        # (n_samples, n_features, n_classes) → mean across samples and classes
        mean_abs = np.mean(np.abs(sv), axis=(0, 2))
    elif sv.ndim == 2:
        # (n_samples, n_features) → mean across samples
        mean_abs = np.mean(np.abs(sv), axis=0)
    else:
        logger.error("Unexpected SHAP values ndim: %d", sv.ndim)
        return None

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return importance_df
