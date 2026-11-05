"""
src/ml/evaluation.py
─────────────────────
Comprehensive metric computation for classification and regression.
Supports binary and multiclass classification and regression.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    is_binary: bool = True,
) -> Dict[str, Any]:
    """
    Compute classification evaluation metrics.

    Returns a dict with:
        accuracy, balanced_accuracy, f1_weighted, f1_macro,
        precision_weighted, recall_weighted, confusion_matrix,
        roc_auc (if y_proba supplied),
        pr_auc (binary only, if y_proba supplied),
        class_names
    """
    metrics: Dict[str, Any] = {}

    def _try(name: str, fn, *args, **kw):
        try:
            val = fn(*args, **kw)
            metrics[name] = float(val) if not isinstance(val, (list, np.ndarray)) else val
        except Exception as e:
            logger.warning("%s failed: %s", name, e)
            metrics[name] = None

    _try("accuracy", accuracy_score, y_true, y_pred)
    _try("balanced_accuracy", balanced_accuracy_score, y_true, y_pred)
    _try("f1_weighted", f1_score, y_true, y_pred, average="weighted", zero_division=0)
    _try("f1_macro", f1_score, y_true, y_pred, average="macro", zero_division=0)
    _try("precision_weighted", precision_score, y_true, y_pred, average="weighted", zero_division=0)
    _try("recall_weighted", recall_score, y_true, y_pred, average="weighted", zero_division=0)

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        if class_names:
            metrics["class_names"] = class_names
    except Exception as e:
        logger.warning("confusion_matrix failed: %s", e)
        metrics["confusion_matrix"] = None

    # Probability-based metrics
    if y_proba is not None:
        # ROC-AUC
        try:
            if is_binary:
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
        except Exception as e:
            logger.warning("roc_auc_score failed: %s", e)
            metrics["roc_auc"] = None

        # PR-AUC (binary only — especially useful for imbalanced datasets)
        if is_binary:
            try:
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["pr_auc"] = float(average_precision_score(y_true, proba_pos))
            except Exception as e:
                logger.warning("pr_auc (average_precision_score) failed: %s", e)
                metrics["pr_auc"] = None

    return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute regression evaluation metrics.

    Returns a dict with: mae, rmse, r2, mape (when safe)
    """
    metrics: Dict[str, Any] = {}

    try:
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    except Exception as e:
        logger.warning("MAE failed: %s", e)
        metrics["mae"] = None

    try:
        mse = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = float(np.sqrt(mse))
    except Exception as e:
        logger.warning("RMSE failed: %s", e)
        metrics["rmse"] = None

    try:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    except Exception as e:
        logger.warning("R2 failed: %s", e)
        metrics["r2"] = None

    # MAPE — only when true values are non-zero
    try:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        mask = y_true_arr != 0
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100)
            metrics["mape_pct"] = round(mape, 4)
        else:
            metrics["mape_pct"] = None
    except Exception as e:
        logger.warning("MAPE failed: %s", e)
        metrics["mape_pct"] = None

    return metrics


def summarize_cv_results(cv_results: Dict[str, Any], scoring: str) -> Dict[str, float]:
    """
    Extract mean and std from GridSearchCV cv_results_ for the best parameters.

    Returns dict with mean_cv_score, std_cv_score, scoring.
    """
    # GridSearchCV uses 'mean_test_score' and 'std_test_score'
    mean_key = "mean_test_score"
    std_key = "std_test_score"

    # Also try the scoring-specific keys (older sklearn versions)
    if mean_key not in cv_results:
        scoring_safe = scoring.replace("-", "_")
        alt_mean = f"mean_test_{scoring_safe}"
        if alt_mean in cv_results:
            mean_key = alt_mean
            std_key = f"std_test_{scoring_safe}"
        else:
            # Last resort: any mean_test_ key
            mean_keys = [k for k in cv_results if k.startswith("mean_test_")]
            if mean_keys:
                mean_key = mean_keys[0]
                std_key = mean_key.replace("mean_", "std_")
            else:
                logger.warning("No CV score keys found in cv_results_")
                return {"scoring": scoring}

    try:
        means = np.array(cv_results[mean_key])
        best_idx = int(np.argmax(means))
        stds = cv_results.get(std_key, [0.0] * len(means))

        return {
            "mean_cv_score": float(means[best_idx]),
            "std_cv_score": float(stds[best_idx]),
            "scoring": scoring,
            "n_candidates": len(means),
        }
    except Exception as e:
        logger.warning("summarize_cv_results failed: %s", e)
        return {"scoring": scoring}


def format_metrics_table(
    metrics: Dict[str, Any],
    problem_type: str,
) -> pd.DataFrame:
    """
    Return a tidy DataFrame of metrics for display in Streamlit or report.
    Excludes confusion_matrix and class_names from the table.
    """
    exclude = {"confusion_matrix", "class_names"}
    rows = []
    for k, v in metrics.items():
        if k in exclude or v is None:
            continue
        if isinstance(v, float):
            rows.append({"Metric": k.replace("_", " ").title(), "Value": f"{v:.4f}"})
        elif isinstance(v, int):
            rows.append({"Metric": k.replace("_", " ").title(), "Value": str(v)})
    return pd.DataFrame(rows)
