# Categorical Feature Encoding

## Overview
Machine learning models require numerical inputs. Categorical features must be encoded before training. The choice of encoding strategy significantly impacts model performance and can introduce data leakage if not handled carefully.

## Types of Categorical Variables
- **Nominal**: No inherent order (e.g., color, city, gender). Use One-Hot Encoding.
- **Ordinal**: Natural order exists (e.g., low/medium/high, education level). Use Ordinal Encoding.
- **High cardinality**: Many unique values (e.g., postal codes, user IDs). Requires special handling (see high_cardinality.md).

## Encoding Methods

### One-Hot Encoding (OHE)
- Creates a binary column for each unique category.
- Best for nominal features with low cardinality (<15–20 categories).
- Prevents models from assuming ordinal relationships.
- `handle_unknown='ignore'` prevents failures on unseen test categories.
- **Hyperparameter**: `drop='first'` reduces multicollinearity for linear models.
- **Warning**: Creates sparse, high-dimensional feature spaces for high-cardinality features.

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
```

### Ordinal Encoding
- Maps categories to integers (0, 1, 2, ...).
- Appropriate only when a meaningful order exists.
- Tree-based models can discover non-monotonic relationships even with ordinal encoding.
- Risk: linear models will interpret the integer difference as meaningful.

```python
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
```

### Binary Encoding
- Converts category integers to binary representation.
- More compact than OHE for medium-cardinality features.
- Available via the `category_encoders` library.

### Target Encoding
- Replaces each category with the mean target value.
- Powerful but prone to data leakage; must use cross-validation-aware encoding.
- Not recommended unless leakage is controlled.

## Pipeline Integration
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
```

## Preventing Data Leakage
- Always fit encoders on training data only.
- Use `ColumnTransformer` inside a scikit-learn `Pipeline` so encoders are fitted during cross-validation.
- Never fit on the full dataset before splitting.

## Hyperparameter Recommendations for Models with Encoded Categorical Features
- **Logistic Regression**: Scale features after encoding; use `max_iter=1000`.
- **Decision Tree**: No scaling required; one-hot or ordinal both work.
- **Random Forest**: No scaling required; one-hot encoding is fine for low cardinality.
- For high-cardinality features, consider target encoding (with care) or dimensionality reduction.
