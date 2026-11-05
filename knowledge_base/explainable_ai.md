# Explainable Artificial Intelligence (XAI)

## Overview
Explainable AI (XAI) refers to techniques that make machine learning model predictions understandable to humans. In this system, XAI is used to generate both global (model-wide) and local (prediction-specific) explanations that are grounded in actual model outputs.

## Why XAI Matters
- Builds trust: stakeholders can verify model behaviour.
- Debugging: reveals unexpected feature dependencies or data leakage.
- Compliance: regulations (e.g., GDPR) may require explanation of automated decisions.
- Scientific validity: distinguishes model associations from causal claims.

## Global Explanations
Global explanations summarize the model's behaviour across all training examples.
- **Feature Importance** (e.g., Random Forest `feature_importances_`): average impurity reduction per feature.
- **Permutation Importance**: measures performance drop when a feature's values are shuffled.
- **Global SHAP**: mean absolute SHAP values across many samples, ranked by importance.

## Local Explanations
Local explanations explain a single prediction for a specific input instance.
- **SHAP**: assigns a contribution value to each feature for a specific prediction.
- **LIME**: fits a local linear model around a single prediction.

## SHAP (SHapley Additive exPlanations)

### Theory
SHAP is based on cooperative game theory. The SHAP value for feature `i` is the average marginal contribution of that feature across all possible orderings of features.

Key property: SHAP values sum to the model's output minus the base rate (expected output):
`sum(SHAP values) = model_output - E[model_output]`

### Explainers
- **TreeExplainer**: Fast, exact SHAP for tree models (Decision Tree, Random Forest, XGBoost).
- **LinearExplainer**: Fast, exact SHAP for linear models (Logistic Regression, Linear Regression).
- **KernelExplainer**: Model-agnostic, slow, approximate. Use as fallback.

### Output
- `shap_values`: Array of shape `(n_samples, n_features)` (binary) or `(n_classes, n_samples, n_features)` (multiclass).
- Positive SHAP → feature pushed prediction toward the positive class/higher value.
- Negative SHAP → feature pushed prediction toward the negative class/lower value.
- Magnitude indicates relative importance.

### Code Example
```python
import shap

# For tree models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

# For linear models
explainer = shap.LinearExplainer(model, X_transformed)
shap_values = explainer.shap_values(X_transformed)
```

## LIME (Local Interpretable Model-agnostic Explanations)

### Theory
LIME perturbs an input instance and fits a locally-linear interpretable model to approximate the behaviour of the complex model in the neighbourhood of that instance.

### When to Use
- When TreeExplainer or LinearExplainer are not applicable.
- As a complementary explanation for verification.

### Code Example
```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)
exp = explainer.explain_instance(instance, model.predict_proba, num_features=10)
```

## Interpreting Contributions Correctly

### Critical Distinctions
1. **Association ≠ Causation**: A positive SHAP value means the feature increased the prediction in this model; it does NOT mean the feature causes the outcome.
2. **Local ≠ Global**: A feature may be locally important for one prediction but globally unimportant.
3. **Direction**: Positive SHAP → pushed toward higher output. Negative SHAP → pushed toward lower output.
4. **Magnitude**: Larger absolute SHAP → larger contribution to this prediction.
5. **Baseline**: SHAP values are relative to the model's expected output (base rate), not to zero.

### What Feature Contributions Mean
- Feature A has SHAP value +0.3: "This model's prediction for this instance is 0.3 higher than the base rate because of Feature A's value."
- This reflects the model's learned association, not ground truth causality.

## RAG Integration for XAI
In this system:
1. SHAP/LIME generates actual feature contribution values.
2. These values are passed to the RAG retriever to find relevant knowledge.
3. Retrieved documents and actual SHAP values are passed to the LLM prompt.
4. The LLM must only use the provided values — it cannot invent contributions.

## Limitations
- SHAP TreeExplainer is not applicable to all model types.
- KernelExplainer is computationally expensive for large datasets.
- LIME explanations are stochastic; results may vary across runs.
- SHAP values for correlated features can be ambiguous.
- Feature contributions explain model behaviour, not data-generating processes.
- Users must understand that high SHAP importance does not imply practical significance.
