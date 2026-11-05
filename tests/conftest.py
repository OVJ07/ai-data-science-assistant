"""
tests/conftest.py
──────────────────
Shared fixtures for the test suite.
All tests use synthetic datasets — no real CSV files or network required.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Synthetic datasets ────────────────────────────────────────────────────────

@pytest.fixture
def binary_classification_df() -> pd.DataFrame:
    """Small balanced binary classification dataset."""
    rng = np.random.default_rng(42)
    n = 120
    return pd.DataFrame({
        "age": rng.integers(18, 70, n).astype(float),
        "income": rng.normal(50000, 15000, n),
        "education": rng.choice(["high_school", "bachelor", "master", "phd"], n),
        "employed": rng.choice(["yes", "no"], n),
        "target": rng.choice([0, 1], n),
    })


@pytest.fixture
def imbalanced_classification_df() -> pd.DataFrame:
    """Imbalanced binary classification dataset (ratio ~0.10)."""
    rng = np.random.default_rng(42)
    majority = 90
    minority = 10
    n = majority + minority
    return pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(5, 2, n),
        "category": rng.choice(["A", "B", "C"], n),
        "target": [0] * majority + [1] * minority,
    })


@pytest.fixture
def multiclass_df() -> pd.DataFrame:
    """Small multiclass classification dataset (3 classes)."""
    rng = np.random.default_rng(42)
    n = 90
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(1, 0.5, n),
        "cat": rng.choice(["red", "blue", "green"], n),
        "target": rng.choice([0, 1, 2], n),
    })


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """Regression dataset with continuous target."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(0, 1, n)
    return pd.DataFrame({
        "x1": x,
        "x2": rng.normal(2, 1, n),
        "category": rng.choice(["A", "B"], n),
        "target": 3 * x + rng.normal(0, 0.5, n),
    })


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """Dataset with missing values."""
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n).astype(float),
        "income": rng.normal(50000, 15000, n),
        "city": rng.choice(["NY", "LA", "Chicago"], n),
        "target": rng.choice([0, 1], n),
    })
    # Introduce missings
    mask_age = rng.choice([True, False], n, p=[0.1, 0.9])
    mask_city = rng.choice([True, False], n, p=[0.15, 0.85])
    df.loc[mask_age, "age"] = np.nan
    df.loc[mask_city, "city"] = np.nan
    return df


@pytest.fixture
def high_cardinality_df() -> pd.DataFrame:
    """Dataset with a high-cardinality categorical column."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "product_id": [f"PROD_{i:04d}" for i in range(n)],
        "price": rng.normal(100, 30, n),
        "category": rng.choice(["A", "B"], n),
        "target": rng.choice([0, 1], n),
    })


@pytest.fixture
def adult_income_sample_df() -> pd.DataFrame:
    """Adult Census-like rows with a pandas string-extension target.

    ``native.country`` deliberately has more than the high-cardinality
    threshold so the RAG top-K one-hot recommendation is exercised.
    """
    rng = np.random.default_rng(314)
    n = 500
    countries = [f"country_{i:02d}" for i in range(24)]
    age = rng.integers(18, 70, n)
    education_num = rng.integers(6, 16, n)
    hours = rng.integers(20, 61, n)
    high_income = (age + education_num * 2 + hours // 8 + rng.normal(0, 7, n)) > 108
    return pd.DataFrame({
        "age": age,
        "fnlwgt": rng.integers(15_000, 400_000, n),
        "education.num": education_num,
        "capital.gain": rng.choice([0, 0, 0, 0, 1500, 5000], n),
        "capital.loss": rng.choice([0, 0, 0, 200, 1200], n),
        "hours.per.week": hours,
        "workclass": pd.Series(rng.choice(["Private", "Self-emp", "Government"], n), dtype="str"),
        "education": pd.Series(rng.choice(["HS-grad", "Bachelors", "Masters"], n), dtype="str"),
        "marital.status": pd.Series(rng.choice(["Married", "Single", "Divorced"], n), dtype="str"),
        "occupation": pd.Series(rng.choice(["Tech", "Sales", "Service", "Admin"], n), dtype="str"),
        "relationship": pd.Series(rng.choice(["Husband", "Wife", "Not-in-family"], n), dtype="str"),
        "race": pd.Series(rng.choice(["White", "Black", "Asian-Pac-Islander"], n), dtype="str"),
        "sex": pd.Series(rng.choice(["Male", "Female"], n), dtype="str"),
        "native.country": pd.Series(rng.choice(countries, n), dtype="str"),
        "income": pd.Series(np.where(high_income, ">50K", "<=50K"), dtype="str"),
    })


@pytest.fixture
def knowledge_base_dir(tmp_path) -> Path:
    """Create a minimal temporary knowledge base for retrieval tests."""
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "missing_values.md").write_text(
        "# Missing Values\n\nUse median imputation for missing numerical values.\n"
        "Class imbalance requires class_weight='balanced'."
    )
    (kb / "class_imbalance.md").write_text(
        "# Class Imbalance\n\nClass imbalance causes models to predict the majority class.\n"
        "Use balanced accuracy as the evaluation metric."
    )
    (kb / "random_forest.md").write_text(
        "# Random Forest\n\nn_estimators and max_depth are key hyperparameters for random forest.\n"
        "class_weight balanced helps with imbalanced datasets."
    )
    return kb
