# Regression Models

## Overview
Regression predicts a continuous numerical output. This document covers Linear Regression, Decision Tree Regressor, and Random Forest Regressor within a scikit-learn pipeline.

## Problem Type Detection
Use regression when:
- The target variable is continuous (float/int with many unique values).
- Unique target values / total rows > 0.05 for integer targets.
- The problem is not a classification task.

## Linear Regression

### Overview
Models the target as a linear combination of input features. Fast, interpretable, and effective when the relationship is approximately linear.

### Key Hyperparameters
- No significant tunable hyperparameters for OLS LinearRegression.
- Consider Ridge or Lasso for regularization:
  - **Ridge (L2)**: `alpha` in [0.001, 0.01, 0.1, 1, 10, 100]
  - **Lasso (L1)**: `alpha` in [0.001, 0.01, 0.1, 1, 10]

### Preprocessing Requirements
- **Scaling**: Required (StandardScaler).
- **Imputation**: Required.
- **Encoding**: OHE for categorical features.

## Decision Tree Regressor

### Key Hyperparameters
```python
param_grid = {
    'regressor__max_depth': [3, 5, 7, 10, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 5],
    'regressor__criterion': ['squared_error', 'absolute_error']
}
```

### Preprocessing Requirements
- Imputation only (no scaling required).

## Random Forest Regressor

### Key Hyperparameters
```python
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2],
    'regressor__max_features': ['sqrt', 1.0]
}
```

## Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference; same units as target.
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more than MAE.
- **R² (Coefficient of Determination)**: Proportion of variance explained. 1.0 is perfect; 0.0 predicts the mean; negative values indicate worse than the mean.

## Cross-Validation for Regression
Use `KFold` cross-validation (not `StratifiedKFold`).
```python
from sklearn.model_selection import KFold, cross_val_score
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
```

## Limitations
- Linear regression assumes linearity, homoscedasticity, and normally distributed residuals.
- Decision tree and random forest regressors do not extrapolate beyond training data range.
- All models can overfit with insufficient data or too many features.
