# Decision Tree

## Overview
Decision Trees are non-parametric supervised learning algorithms that partition the feature space into rectangular regions by recursively splitting on the most informative features. They are highly interpretable and require minimal preprocessing.

## When to Use
- When interpretability is a priority (human-readable rules).
- When non-linear decision boundaries are needed.
- As a component in ensemble methods (Random Forest, Gradient Boosting).
- When feature scaling is unavailable or undesirable.
- For both classification and regression.

## Key Hyperparameters

### `max_depth`
- Maximum depth of the tree.
- Controls overfitting: deeper trees → more complex, risk of overfitting.
- `None` (default): grow until pure leaves or min_samples_leaf reached.
- Recommended search: [3, 5, 7, 10, None]

### `min_samples_split`
- Minimum samples required to split an internal node.
- Higher values prevent splitting small nodes → reduces overfitting.
- Default: 2. Recommended search: [2, 5, 10, 20]

### `min_samples_leaf`
- Minimum samples required at a leaf node.
- Prevents creating leaves with very few samples.
- Default: 1. Recommended search: [1, 2, 5, 10]

### `max_features`
- Number of features to consider at each split.
- `'sqrt'`: square root of total features (good default).
- `'log2'`: log base 2 of total features.
- `None` or `1.0`: use all features.

### `criterion`
- Splitting criterion.
- Classification: `'gini'` (default) or `'entropy'`.
- Regression: `'squared_error'` (default) or `'absolute_error'`.

### `class_weight`
- `None` or `'balanced'`. Set `'balanced'` for imbalanced datasets.

## Recommended Classification Search Space
```python
param_grid = {
    'classifier__max_depth': [3, 5, 7, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 5],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__class_weight': [None, 'balanced']
}
```

## Preprocessing Requirements
- **Scaling**: Not required (splits are threshold-based).
- **Imputation**: Required (scikit-learn decision trees cannot handle NaN).
- **Encoding**: OHE or ordinal encoding both work.

## Overfitting
Decision trees are prone to overfitting, especially on small datasets.
Mitigations:
- Limit `max_depth`.
- Increase `min_samples_leaf`.
- Use pruning (cost-complexity pruning with `ccp_alpha`).
- Use Random Forest instead.

## Interpretation
- Feature importance: based on total impurity reduction across splits.
- Path from root to leaf provides a human-readable decision rule.
- Feature importances do NOT indicate causation.

## Limitations
- Unstable: small changes in data can produce very different trees.
- Biased toward features with many unique values.
- Poor extrapolation outside the range of training data.
- Single trees are generally outperformed by ensemble methods.
