# Cross-Validation

## Overview
Cross-validation (CV) is a technique for estimating model generalization performance by training and evaluating the model on multiple data partitions. It provides a more reliable estimate than a single train/test split, especially for small datasets.

## Why Cross-Validation?
- Single train/test splits have high variance — results depend heavily on how the split happens to partition the data.
- CV uses all training data for both training and validation (in different folds), reducing this variance.
- CV should ONLY be applied to training data; the test set is held out entirely for final evaluation.

## K-Fold Cross-Validation
Divide training data into K equal parts (folds). Train on K-1 folds, evaluate on the remaining fold. Repeat K times.
- `K=5` or `K=10` is standard. `K=5` offers a good bias-variance tradeoff.
- Use `shuffle=True` with a fixed `random_state` for reproducibility.

```python
from sklearn.model_selection import KFold, cross_val_score
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2')
```

## Stratified K-Fold (Classification)
Preserves the class distribution in each fold. Always use for classification, especially with class imbalance.

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

## GridSearchCV
Exhaustive search over a hyperparameter grid using cross-validation to select the best configuration.

```python
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1,
    refit=True          # refit best model on full training set
)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```

## Preventing Data Leakage in CV
The entire preprocessing pipeline must be inside the GridSearchCV/cross_val_score call.
- **Correct**: `GridSearchCV(Pipeline([preprocessor, model]), ...)`
- **Incorrect**: Fit preprocessor on full data, then do CV on preprocessed data.

## Reporting CV Results
Report:
- Mean CV score across folds
- Standard deviation (measure of stability)
- Best hyperparameter values
- All CV scores (for transparency)

## Nested Cross-Validation
For unbiased model selection AND performance estimation:
- Outer CV: estimates generalization performance.
- Inner CV: selects hyperparameters.
- Expensive but statistically rigorous. Not required for this system but worth documenting.

## Limitations
- CV does not guarantee good performance on the true test set.
- High standard deviation in CV scores indicates instability.
- CV with small datasets may produce high variance estimates.
- Not a substitute for domain-specific validation (e.g., time-series data requires temporal CV).
