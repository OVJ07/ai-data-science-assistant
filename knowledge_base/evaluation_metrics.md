# Evaluation Metrics

## Overview
Choosing the right evaluation metric is critical for honest assessment of model performance. The appropriate metric depends on the problem type (classification vs regression) and the data characteristics (class balance, cost asymmetry).

## Classification Metrics

### Accuracy
- Proportion of correctly classified samples.
- `accuracy = (TP + TN) / (TP + TN + FP + FN)`
- **Problem**: Misleading for imbalanced datasets (a model predicting all-majority achieves high accuracy).
- Use when classes are balanced.

### Precision
- Of all positive predictions, how many are correct?
- `precision = TP / (TP + FP)`
- High precision → few false positives. Important when false positives are costly.

### Recall (Sensitivity)
- Of all actual positives, how many were detected?
- `recall = TP / (TP + FN)`
- High recall → few false negatives. Important when false negatives are costly (e.g., disease detection).

### F1-Score
- Harmonic mean of precision and recall.
- `F1 = 2 * (precision * recall) / (precision + recall)`
- Balanced metric; appropriate when both false positives and false negatives matter.
- **Macro F1**: average F1 per class (treats all classes equally).
- **Weighted F1**: weighted by class support (good for imbalanced classes).

### Balanced Accuracy
- Average recall per class.
- `balanced_accuracy = (recall_class_0 + recall_class_1 + ...) / n_classes`
- More informative than accuracy for imbalanced datasets.

### ROC-AUC
- Area under the Receiver Operating Characteristic curve.
- Threshold-independent; measures ranking quality.
- Range: [0.5, 1.0]. 0.5 = random; 1.0 = perfect.
- Available for binary classification; for multiclass use `roc_auc_score(multi_class='ovr')`.

### Confusion Matrix
- Table showing TP, TN, FP, FN for each class.
- Essential for understanding per-class performance.

## When to Use Which Metric
| Scenario | Recommended Metric |
|---|---|
| Balanced classes | Accuracy, F1-macro |
| Imbalanced classes | Balanced accuracy, F1-weighted, ROC-AUC |
| High FP cost | Precision-oriented |
| High FN cost | Recall-oriented |
| Threshold-free ranking | ROC-AUC |

## Regression Metrics

### MAE (Mean Absolute Error)
- `MAE = mean(|y_true - y_pred|)`
- Robust to outliers; same units as target.
- Symmetric; does not differentiate over- and under-prediction.

### RMSE (Root Mean Squared Error)
- `RMSE = sqrt(mean((y_true - y_pred)^2))`
- Penalizes large errors more than MAE.
- Same units as target; sensitive to outliers.

### R² (Coefficient of Determination)
- Proportion of variance in the target explained by the model.
- `R² = 1 - SS_res / SS_tot`
- 1.0: perfect; 0.0: predicts mean; negative: worse than predicting mean.
- Not comparable across datasets with different target scales.

## Cross-Validation Scoring
- Classification (balanced): `scoring='balanced_accuracy'`
- Classification (standard): `scoring='f1_weighted'`
- Regression: `scoring='neg_root_mean_squared_error'` or `scoring='r2'`

## Limitations
- No metric is universally optimal; choose based on problem context.
- Metrics computed on the test set are point estimates; confidence intervals require bootstrapping.
- Comparing models using a single metric can be misleading; use multiple metrics.
