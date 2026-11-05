# Missing Values in Machine Learning

## Overview
Missing values (also called null values or NaN) are a common challenge in real-world datasets. How you handle them significantly affects model performance and validity.

## Types of Missingness
- **MCAR (Missing Completely at Random)**: No systematic pattern; safe to impute.
- **MAR (Missing at Random)**: Missingness depends on observed variables; imputation is appropriate with care.
- **MNAR (Missing Not at Random)**: Missingness depends on the missing value itself; imputation may introduce bias.

## Detection
Always report: count of missing values per column, percentage missing, and pattern (are certain rows systematically missing across multiple columns?).

## Imputation Strategies

### Numerical Features
- **Mean imputation**: Fast, works well for symmetric distributions with low missing rate (<5%).
- **Median imputation**: More robust to outliers; preferred for skewed distributions.
- **KNN imputation**: Preserves relationships between features; computationally expensive for large datasets.
- **Iterative imputation**: Models each feature as a function of others; most accurate but slowest.

### Categorical Features
- **Mode imputation**: Fill with most frequent value; simple and effective.
- **Constant imputation**: Use a dedicated 'Unknown' or 'Missing' category; preserves missingness information.
- **Model-based imputation**: Use a classifier to predict missing categories.

## Hyperparameter Recommendations
When missing values are present:
- Include imputation as the first step in your preprocessing pipeline.
- For numerical features with <5% missing: use `SimpleImputer(strategy='mean')` or `strategy='median'`.
- For numerical features with >5% missing: prefer `SimpleImputer(strategy='median')` or `KNNImputer`.
- For categorical features: use `SimpleImputer(strategy='most_frequent')` or `strategy='constant', fill_value='Unknown'`.
- Always fit the imputer on training data only and transform both train and test sets.

## Pipeline Integration
```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline

# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
```

## Warnings
- Never impute the target variable.
- Imputation must be fit on training data only to prevent data leakage.
- High missing rates (>30%) in a feature may indicate the feature is unreliable or should be dropped.
- Create a binary indicator feature for missingness when MNAR is suspected.

## Impact on Model Performance
- Tree-based models (Random Forest, Decision Tree) can handle missing values with surrogate splits (though scikit-learn requires imputation).
- Linear models and SVMs require imputation.
- Poorly handled missing values inflate variance and reduce generalization.
