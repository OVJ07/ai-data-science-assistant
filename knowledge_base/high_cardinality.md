# High-Cardinality Features

## Overview
High-cardinality features are categorical variables with many unique values (typically >15–20 unique values, or close to the number of rows). Examples include postal codes, product SKUs, user IDs, free-text fields, and country codes.

## Detection
A feature is considered high-cardinality when:
- Unique values > 20 (configurable threshold)
- Unique ratio (unique values / total rows) > 0.5

ID-like features are extreme cases where unique ratio ≈ 1.0. These should be dropped before training.

## Risks
- **Dimensionality explosion**: One-hot encoding a column with 500 categories creates 500 new features.
- **Overfitting**: High-cardinality features allow models to memorize training data.
- **Data leakage**: Target encoding must be done carefully within cross-validation folds.
- **Sparse representations**: Many OHE columns are all-zero, wasting memory and computation.

## Handling Strategies

### 1. Drop the Feature
When the feature is an ID-like column (unique ratio > 0.95), it carries no generalizable signal. Drop it.

### 2. Frequency Encoding
Replace each category with its frequency in the training set. Simple and effective for tree-based models.
```python
freq_map = X_train[col].value_counts() / len(X_train)
X_train[col] = X_train[col].map(freq_map).fillna(0)
```

### 3. Top-K Encoding
Keep only the top-K most frequent categories; group all others into an 'Other' bucket. Then apply OHE.
```python
top_k = X_train[col].value_counts().head(15).index
X_train[col] = X_train[col].apply(lambda x: x if x in top_k else '__other__')
```

### 4. Binary / Hashing Encoding
Use a fixed-size hash to map categories to binary features. Avoids dimensionality explosion but introduces hash collisions.

### 5. Target Encoding (with CV)
Replace each category with the mean target value (computed within each CV fold to prevent leakage). Powerful but complex.

### 6. Embedding Representations
For very high-cardinality text-like features, neural embeddings may be more appropriate (outside this system's scope).

## Hyperparameter Recommendations
When high-cardinality features are detected:
- Warn the user before applying OHE.
- Suggest dropping obvious ID columns.
- Recommend limiting OHE to top-K categories.
- For Random Forest: high-cardinality is less problematic due to feature sub-sampling.
- For Logistic Regression: high-cardinality with OHE increases memory usage significantly; consider dimensionality reduction.

## Limitations
- Frequency encoding loses information about individual categories.
- Top-K encoding requires choosing K, which is a hyperparameter.
- Target encoding introduces leakage if not properly cross-validated.
- No single strategy is universally optimal; empirical evaluation is required.
