"""
src/ml/diagnostics.py
──────────────────────
Structured provenance tracking and diagnostic fingerprints across the ML pipeline.
Provides immutable hashes for dataset profiles, hyperparameter configurations,
prediction arrays, and end-to-end provenance tracing.
"""

from dataclasses import dataclass, field
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.profiler import DatasetProfile
from src.rag.retriever import RetrievedDocument

logger = logging.getLogger(__name__)


def compute_config_hash(
    param_grid: Dict[str, Any],
    preprocessing_config: Dict[str, Any],
    cv_scoring: str,
) -> str:
    """
    Compute a deterministic 12-char SHA256 hash of the effective configuration.
    Sorts keys and values to ensure identical configurations always hash to the same value.
    """
    clean_grid = {}
    for k, v in sorted(param_grid.items()):
        if isinstance(v, (list, tuple)):
            clean_grid[k] = sorted(list(v), key=lambda x: (str(type(x)), str(x)))
        else:
            clean_grid[k] = v

    clean_prep = {k: v for k, v in sorted(preprocessing_config.items())}

    payload = {
        "cv_scoring": str(cv_scoring),
        "param_grid": clean_grid,
        "preprocessing": clean_prep,
    }
    dumped = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:12]


def compute_prediction_hash(y_pred: Any) -> str:
    """Compute a deterministic 12-char SHA256 hash of the held-out predictions."""
    if y_pred is None:
        return "none"
    arr = np.asarray(y_pred)
    if arr.size == 0:
        return "empty"
    return hashlib.sha256(arr.tobytes()).hexdigest()[:12]


def compute_prediction_agreement(y_pred_a: Any, y_pred_b: Any) -> Optional[float]:
    """Compute fraction of identical predictions between two arrays."""
    if y_pred_a is None or y_pred_b is None:
        return None
    arr_a = np.asarray(y_pred_a)
    arr_b = np.asarray(y_pred_b)
    if arr_a.shape != arr_b.shape or arr_a.size == 0:
        return None
    return float(np.mean(arr_a == arr_b))


@dataclass
class DatasetFingerprint:
    """Deterministic summary and fingerprint of a dataset profile."""
    n_rows: int
    n_cols: int
    problem_type: str
    target_column: str
    has_missing: bool
    missing_rate_overall: float
    is_imbalanced: bool
    imbalance_ratio: float
    high_cardinality_count: int
    fingerprint_hash: str = ""

    @classmethod
    def from_profile(cls, profile: DatasetProfile) -> "DatasetFingerprint":
        fp = cls(
            n_rows=profile.n_rows,
            n_cols=profile.n_cols,
            problem_type=profile.problem_type or "unknown",
            target_column=profile.target_column or "",
            has_missing=bool(profile.has_missing),
            missing_rate_overall=round(float(profile.missing_rate_overall or 0.0), 4),
            is_imbalanced=bool(profile.is_imbalanced),
            imbalance_ratio=round(float(profile.imbalance_ratio or 1.0), 4),
            high_cardinality_count=len(profile.high_cardinality_features or []),
        )
        data = {
            "rows": fp.n_rows,
            "cols": fp.n_cols,
            "type": fp.problem_type,
            "target": fp.target_column,
            "missing": fp.has_missing,
            "missing_rate": fp.missing_rate_overall,
            "imbalanced": fp.is_imbalanced,
            "imbalance_ratio": fp.imbalance_ratio,
            "high_card": fp.high_cardinality_count,
        }
        dumped = json.dumps(data, sort_keys=True)
        fp.fingerprint_hash = hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:12]
        return fp


@dataclass
class PipelineProvenanceTrace:
    """Full auditable provenance chain from retrieval to final evaluation."""
    strategy_name: str
    dataset_fingerprint: str
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    generated_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    applied_configuration_changes: List[str] = field(default_factory=list)
    config_hash: str = ""
    prediction_hash: str = ""
    effective_cv_scoring: str = ""
    best_cv_score: Optional[float] = None
    cv_score_std: Optional[float] = None
    best_parameters: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)

