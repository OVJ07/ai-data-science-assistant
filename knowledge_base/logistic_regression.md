# Logistic Regression

## Overview
Logistic Regression is a linear probabilistic classifier that models the log-odds of a binary (or multiclass with softmax) outcome as a linear combination of input features. Despite its name, it is a classification algorithm.

## When to Use
- Binary or multiclass classification.
- When interpretability is important (coefficients are directly interpretable).
- When the decision boundary is approximately linear.
- As a strong baseline before trying complex models.
- When training time is limited or dataset is large.

## Assumptions
- Linear relationship between features and log-odds.
- Features are approximately independent (no severe multicollinearity).
- No extreme outliers in continuous features.
- Features should be scaled (StandardScaler or MinMaxScaler).

## Key Hyperparameters

### `C` (Regularization Strength)
- Inverse of regularization strength: smaller `C` → stronger regularization.
- Range: [0.001, 0.01, 0.1, 1, 10, 100]
- Default: `C=1.0`
- Too small: underfitting; too large: overfitting.

### `penalty`
- `'l2'` (Ridge): default, shrinks all coefficients smoothly. Best for most cases.
- `'l1'` (Lasso): produces sparse solutions; performs feature selection. Requires `solver='liblinear'` or `solver='saga'`.
- `'elasticnet'`: combination; requires `solver='saga'` and `l1_ratio` parameter.
- `'none'`: no regularization; risk of overfitting on high-dimensional data.

### `solver`
- `'lbfgs'`: default, supports l2 and multiclass. Good for small/medium datasets.
- `'liblinear'`: supports l1 and l2; does not support multiclass natively.
- `'saga'`: supports l1, l2, elasticnet; good for large datasets.

### `max_iter`
- Maximum number of iterations for convergence. Increase if convergence warnings appear.
- Default: 100. Recommend: 1000 for most datasets.

### `class_weight`
- `None` or `'balanced'`. Set `'balanced'` for imbalanced datasets.

## Recommended Search Space
```python
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear'],
    'classifier__max_iter': [1000],
    'classifier__class_weight': [None, 'balanced']
}
```

## Preprocessing Requirements
- **Scaling is mandatory**: StandardScaler or MinMaxScaler.
- **Imputation**: required before scaling.
- **Encoding**: OHE for categorical features.

## Interpretation
- Positive coefficient → feature increases log-odds of positive class.
- Coefficient magnitude → relative importance (after scaling).
- Coefficients are NOT causal; they reflect associations in the training data.

## Limitations
- Poor performance when decision boundary is non-linear.
- Sensitive to outliers in continuous features.
- Collinear features inflate variance of coefficients.
- Not suitable for very high-dimensional sparse data without regularization.
