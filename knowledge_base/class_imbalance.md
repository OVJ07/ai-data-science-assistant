# Class Imbalance in Classification

## Overview
Class imbalance occurs when one or more classes in a classification problem have significantly fewer samples than others. This is common in fraud detection, medical diagnosis, anomaly detection, and churn prediction.

## Detection
- **Imbalance ratio**: minority class count / majority class count < 0.20 → imbalanced
- **Absolute minority count**: fewer than 100 samples in minority class is high risk
- **Multiclass**: check the smallest class relative to the total

## Risks of Ignoring Imbalance
- Models maximize accuracy by predicting the majority class.
- Minority class recall collapses to near zero.
- Accuracy metric becomes misleading (e.g., 95% accuracy when 95% of samples are class 0).
- Feature importance may be distorted toward majority-class patterns.

## Strategies

### 1. Class Weights (`class_weight='balanced'`)
Most scikit-learn classifiers support `class_weight='balanced'`, which inversely weights each class by its frequency.
- **Zero cost**: no resampling required; works within standard pipelines.
- **Recommended** as a first approach for moderate imbalance.
- Available in: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SVC.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
```

### 2. Oversampling (SMOTE)
Synthetic Minority Over-sampling TEchnique generates synthetic minority samples.
- Requires `imbalanced-learn` library.
- Apply ONLY inside CV folds to prevent data leakage.
- **Warning**: SMOTE should be inside the pipeline, not applied to the full training set before CV.

### 3. Undersampling
Randomly remove majority-class samples. Simple but discards data.

### 4. Threshold Adjustment
Lower the decision threshold below 0.5 to increase minority-class recall. Tune threshold on validation data.

## Evaluation Metrics for Imbalanced Data
Standard accuracy is misleading. Use:
- **Balanced Accuracy**: average recall per class.
- **F1-score (weighted or macro)**: harmonic mean of precision and recall.
- **ROC-AUC**: area under the ROC curve; threshold-independent.
- **Precision-Recall AUC**: especially informative when positive class is rare.
- **Confusion Matrix**: always display to reveal class-specific performance.

## Hyperparameter Recommendations
When class imbalance is detected (ratio < 0.20):
- Add `class_weight='balanced'` as a candidate hyperparameter.
- Use `scoring='balanced_accuracy'` or `scoring='f1_weighted'` in cross-validation.
- Use `StratifiedKFold` to preserve class distribution in folds.
- Report confusion matrix and per-class precision/recall.

## Stratified Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='balanced_accuracy')
```

## Limitations
- `class_weight='balanced'` does not create new samples; it only reweights the loss function.
- SMOTE requires careful implementation to avoid leakage.
- No single strategy guarantees improvement on all datasets; empirical evaluation is essential.
- Imbalance handling may trade recall for precision; understand the business cost of each error type.
