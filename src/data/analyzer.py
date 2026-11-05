"""
src/data/analyzer.py
─────────────────────
Higher-level dataset analysis that wraps the profiler and adds
correlation analysis, EDA helpers, and summary statistics used
by the Streamlit interface.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.profiler import DatasetProfile, DatasetProfiler

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """
    Runs complete dataset analysis including profiling, correlation,
    and EDA summaries. Acts as the primary entry point for Phase 2.
    """

    def __init__(self) -> None:
        self._profiler = DatasetProfiler()
        self.profile: Optional[DatasetProfile] = None
        self.df: Optional[pd.DataFrame] = None

    def analyze(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None,
    ) -> DatasetProfile:
        """Run full analysis and return a DatasetProfile."""
        self.df = df
        self.profile = self._profiler.profile(df, target_column, problem_type)
        return self.profile

    # ── EDA helpers ───────────────────────────────────────────────────────────

    def get_correlation_matrix(
        self,
        method: str = "pearson",
        max_features: int = 25,
    ) -> Optional[pd.DataFrame]:
        """Return the correlation matrix of numerical columns (capped at max_features)."""
        if self.df is None:
            return None
        num_cols = self.profile.numerical_features if self.profile else []
        if self.profile and self.profile.target_column:
            target = self.profile.target_column
            if target in self.df.columns and pd.api.types.is_numeric_dtype(self.df[target]):
                num_cols = list(num_cols) + [target]
        num_cols = [c for c in num_cols if c in self.df.columns][:max_features]
        if len(num_cols) < 2:
            return None
        try:
            return self.df[num_cols].corr(method=method)
        except Exception as exc:
            logger.warning("Could not compute correlation matrix: %s", exc)
            return None

    def get_column_summary(self, col: str) -> Dict[str, Any]:
        """Return a per-column summary dictionary."""
        if self.df is None or col not in self.df.columns:
            return {}
        series = self.df[col]
        summary: Dict[str, Any] = {
            "column": col,
            "dtype": str(series.dtype),
            "n_missing": int(series.isnull().sum()),
            "pct_missing": round(series.isnull().mean() * 100, 2),
            "n_unique": int(series.nunique(dropna=True)),
            "unique_ratio": round(series.nunique(dropna=True) / max(len(series), 1), 4),
        }
        if pd.api.types.is_numeric_dtype(series.dtype):
            desc = series.describe()
            summary.update({
                "mean": float(desc["mean"]) if "mean" in desc else None,
                "std": float(desc["std"]) if "std" in desc else None,
                "min": float(desc["min"]) if "min" in desc else None,
                "25%": float(desc["25%"]) if "25%" in desc else None,
                "50%": float(desc["50%"]) if "50%" in desc else None,
                "75%": float(desc["75%"]) if "75%" in desc else None,
                "max": float(desc["max"]) if "max" in desc else None,
            })
        else:
            top = series.value_counts().head(5)
            summary["top_values"] = {str(k): int(v) for k, v in top.items()}
        return summary

    def get_all_column_summaries(self) -> List[Dict[str, Any]]:
        """Return summaries for all columns."""
        if self.df is None:
            return []
        return [self.get_column_summary(col) for col in self.df.columns]

    def get_target_distribution(self) -> Optional[Dict[str, Any]]:
        """Return target value distribution info."""
        if self.df is None or self.profile is None:
            return None
        target = self.profile.target_column
        if not target or target not in self.df.columns:
            return None
        series = self.df[target].dropna()
        if self.profile.problem_type and "classification" in self.profile.problem_type:
            vc = series.value_counts()
            return {
                "type": "categorical",
                "distribution": {str(k): int(v) for k, v in vc.items()},
                "n_classes": self.profile.n_classes,
                "is_imbalanced": self.profile.is_imbalanced,
                "imbalance_ratio": self.profile.imbalance_ratio,
            }
        else:
            return {
                "type": "continuous",
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "skewness": float(series.skew()),
            }

    def get_missing_value_report(self) -> pd.DataFrame:
        """Return a DataFrame summarizing missing values across all columns."""
        if self.df is None:
            return pd.DataFrame()
        missing = self.df.isnull().sum()
        pct = missing / len(self.df) * 100
        report = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": pct.values.round(2),
        })
        return report[report["Missing Count"] > 0].sort_values("Missing %", ascending=False)

    def detect_leakage_candidates(self, target: str) -> List[str]:
        """
        Heuristically detect features that might cause data leakage.
        Returns a list of suspicious column names with reasons.
        """
        if self.df is None:
            return []
        candidates = []
        target_lower = target.lower()
        for col in self.df.columns:
            if col == target:
                continue
            col_lower = col.lower()
            # Columns that share words with the target or contain 'target', 'label', 'outcome'
            leakage_keywords = ["target", "label", "outcome", "class", "result", "prediction"]
            if any(kw in col_lower for kw in leakage_keywords):
                candidates.append(col)
            elif any(kw in target_lower for kw in col_lower.split("_") if len(kw) > 3):
                candidates.append(col)
        return list(set(candidates))
