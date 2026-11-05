# Hyperparameter Tuning

## Overview
Hyperparameters are model configuration values set before training (as opposed to parameters learned during training). Selecting good hyperparameters can significantly improve model performance and generalization.

## Key Principles
1. **Never tune on the test set**: Use cross-validation on training data only.
2. **Preprocessing inside CV**: Fit all preprocessing steps within each CV fold.
3. **Start simple**: Begin with a small grid to identify promising regions.
4. **Reproducibility**: Use fixed random seeds.
5. **Measure uncertainty**: Report CV mean and standard deviation, not just the best score.

## Tuning Strategies

### Grid Search (GridSearchCV)
Exhaustively evaluates all combinations in a defined grid.
- **Pros**: Deterministic, reproducible, easy to interpret.
- **Cons**: Exponential in the number of hyperparameters.
- **Use when**: Grid is small (< ~1000 combinations).

```python
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)
```

### Randomized Search (RandomizedSearchCV)
Samples a fixed number of random combinations from distributions or lists.
- **Pros**: More efficient for large search spaces; can explore wider ranges.
- **Cons**: Non-deterministic (mitigated by `random_state`).
- **Use when**: Grid is large or hyperparameters interact in complex ways.

### Bayesian Optimization
Uses a probabilistic model to guide search toward promising regions.
- More sample-efficient than grid or random search.
- Available via `scikit-optimize`, `optuna`, `hyperopt`.
- Not required for this system; grid search is sufficient.

## Dataset-Aware Search Space Generation

The search space should reflect actual dataset characteristics:

| Dataset Property | Hyperparameter Implication |
|---|---|
| Class imbalance | Add `class_weight='balanced'` |
| Missing values | Include imputation strategy in search |
| High-cardinality features | Limit OHE, consider frequency encoding |
| Small dataset (< 100 samples) | Reduce tree depth to prevent overfitting |
| Large feature set | Reduce `max_features` candidates |
| Many classes (multiclass) | Use macro-averaged scoring |

## Validation Best Practices
- Use `StratifiedKFold` for classification (preserves class ratios per fold).
- Use `KFold` for regression.
- `refit=True` in GridSearchCV refits the best model on the full training set.
- Always evaluate the final refitted model on the held-out test set.

## Interpreting Results
- `best_score_`: Mean CV score of the best configuration.
- `best_params_`: Best hyperparameter combination.
- `cv_results_`: Full grid search results for all combinations.
- A large gap between train score and CV score indicates overfitting.
- High standard deviation in CV scores indicates instability.

## Common Mistakes
- Tuning directly on the test set (data leakage).
- Fitting preprocessing on the full dataset before CV.
- Choosing hyperparameters based on test set performance.
- Reporting only the best CV score without standard deviation.
- Using a single train/test split instead of CV for hyperparameter selection.

## RAG-Guided Tuning
This system uses dataset characteristics to generate targeted search spaces:
1. Analyze dataset properties (imbalance, missing rate, cardinality, etc.).
2. Retrieve relevant knowledge documents via TF-IDF similarity.
3. Generate a dataset-aware search space based on retrieved recommendations.
4. Compare against a conventional baseline search space.
5. Report actual CV results for both approaches.
