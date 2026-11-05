# Random Forest

## Overview
Random Forest is an ensemble method that builds many decision trees on bootstrap samples of the training data and averages their predictions. It reduces overfitting through variance reduction (bagging) and feature randomization.

## When to Use
- General-purpose classification and regression.
- When a single decision tree overfits.
- When feature importance estimates are needed.
- When robustness to outliers and noisy features is needed.
- When training time is not a primary constraint.

## Key Hyperparameters

### `n_estimators`
- Number of trees in the forest.
- More trees → better performance (up to a plateau) but slower training.
- Default: 100. Recommended search: [50, 100, 200, 300]

### `max_depth`
- Maximum depth of each tree.
- `None` (default): grow fully. Limiting depth reduces overfitting slightly.
- Recommended search: [None, 10, 20, 30]

### `min_samples_split`
- Minimum samples to split a node.
- Default: 2. Recommended search: [2, 5, 10]

### `min_samples_leaf`
- Minimum samples at a leaf.
- Default: 1. Recommended search: [1, 2, 4]

### `max_features`
- Features considered per split.
- Classification default: `'sqrt'`. Regression default: `1.0`.
- Recommended: `['sqrt', 'log2', 0.5]`

### `bootstrap`
- Whether to use bootstrap sampling.
- Default: `True`. Setting `False` disables bagging (out-of-bag scoring unavailable).

### `class_weight`
- `None`, `'balanced'`, or `'balanced_subsample'`.
- `'balanced'` weights classes globally; `'balanced_subsample'` weights per bootstrap sample.
- Use for imbalanced datasets.

### `n_jobs`
- Number of parallel jobs. `-1` uses all CPU cores.

## Recommended Classification Search Space
```python
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__class_weight': [None, 'balanced']
}
```

## Preprocessing Requirements
- **Scaling**: Not required.
- **Imputation**: Required (scikit-learn does not handle NaN).
- **Encoding**: OHE works; ordinal encoding also acceptable.

## Feature Importance
- Provided via `feature_importances_` attribute.
- Based on mean decrease in impurity across all trees.
- Can be biased toward high-cardinality features.
- Permutation importance (`sklearn.inspection.permutation_importance`) is more reliable.

## Out-of-Bag (OOB) Score
When `bootstrap=True`, each tree is evaluated on its out-of-bag samples, providing a free cross-validation-like estimate.

## Limitations
- Less interpretable than a single decision tree.
- Training and inference are slower than simpler models.
- Feature importances can be biased; use permutation importance for more reliable estimates.
- May not extrapolate well outside the training data range.
- Memory-intensive for large forests.
