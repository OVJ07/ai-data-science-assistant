"""
src/llm/prompts.py
───────────────────
Grounded LLM prompt templates.
All prompts enforce that actual XAI evidence must be used.
"""

from src.rag.explanation import GroundedExplanationContext


def build_local_explanation_prompt(ctx: GroundedExplanationContext) -> str:
    """
    Build a grounded LLM prompt for explaining an individual prediction.

    Rules enforced in the prompt:
    1. Use only the supplied feature contribution values.
    2. Never invent feature importance scores or model metrics.
    3. Distinguish model associations from causation.
    4. Explain uncertainty and limitations.
    5. Do not describe a feature as influencing a prediction unless the supplied explanation supports it.
    6. Clearly identify the observation and predicted outcome being explained.
    7. Distinguish local explanations from global feature importance.
    8. Treat retrieved knowledge as contextual evidence, not proof of causality.
    """
    shap_evidence = ctx.format_shap_evidence()
    retrieved_knowledge = ctx.format_retrieved_knowledge()

    return f"""You are an Explainable AI assistant. Your task is to explain a machine learning model's prediction for a specific observation.

STRICT RULES - you MUST follow all of these:
1. ONLY use the feature contribution values provided below. Do not invent, estimate, or extrapolate any values.
2. Never state that a feature "caused" the prediction - only that it "contributed to" or "was associated with" the prediction IN THIS MODEL.
3. If you are uncertain about an interpretation, say so explicitly.
4. Clearly distinguish between this local explanation (one observation) and global feature importance (model-wide).
5. Treat retrieved knowledge as background context, not proof of causality.
6. If SHAP values are missing, state that the explanation is unavailable and do not substitute invented values.

OBSERVATION DETAILS:
- Observation Index: #{ctx.observation_index}
- Model: {ctx.model_name}
- Problem Type: {ctx.problem_type}
- Target Column: {ctx.target_column}
- Predicted Outcome: {ctx.prediction}
- Predicted Class Label: {ctx.predicted_class_label or "N/A"}
- Prediction Probability: {f"{ctx.prediction_probability:.4f}" if ctx.prediction_probability is not None else "N/A"}
- SHAP Explained Output: class/value {ctx.explained_output_label or "N/A"} ({f"{ctx.explained_output_probability:.4f}" if ctx.explained_output_probability is not None else "N/A"})
- SHAP Output Space: {ctx.output_space or "not verified"}
- XAI Method: {ctx.xai_method}

{shap_evidence}

RETRIEVED KNOWLEDGE (contextual background - do NOT treat as causal proof):
{retrieved_knowledge}

{ctx.causation_disclaimer}

TASK:
Write a clear, factual explanation of why the model made this prediction for this specific observation.
- Start by stating the prediction and its probability.
- Identify the top features that moved the explicitly stated SHAP output up or down, using the supplied SHAP values.
- For each top feature, state: the feature name, the SHAP value, and the direction (increases/decreases prediction).
- Note that these are model associations, not causal relationships.
- Use the retrieved knowledge to provide relevant context about these features.
- End with a brief limitations statement about SHAP explanations and this model type.

Keep your explanation concise (200-350 words). Use plain language accessible to a non-expert audience.
"""


def build_global_explanation_prompt(
    global_importance: dict,
    model_name: str,
    problem_type: str,
    target_column: str,
    retrieved_knowledge: str,
    top_n: int = 10,
) -> str:
    """Build a grounded LLM prompt for global feature importance explanation."""
    sorted_features = sorted(global_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    feature_lines = "\n".join(
        f"  {i+1}. {feat}: mean |SHAP| = {val:.4f}"
        for i, (feat, val) in enumerate(sorted_features)
    )

    return f"""You are an Explainable AI assistant explaining global feature importance.

STRICT RULES:
1. Only use the mean |SHAP| values provided below.
2. Do not invent or modify any values.
3. Distinguish global importance from local (per-prediction) contributions.
4. Do not claim causation - these are model associations.

MODEL: {model_name}
PROBLEM TYPE: {problem_type}
TARGET: {target_column}

GLOBAL FEATURE IMPORTANCE (mean absolute SHAP values across training data):
{feature_lines}

RETRIEVED KNOWLEDGE (context only):
{retrieved_knowledge}

TASK:
Explain which features the model consistently relies on most across all predictions.
- List the top 5 most important features with their mean |SHAP| values.
- Explain what each value means (higher = more consistently influential in the model).
- Note that these reflect model behaviour, not causal relationships.
- Mention that individual predictions may be driven by different features.
Keep your explanation to 150-250 words.
"""


def build_fallback_local_explanation(ctx: GroundedExplanationContext) -> str:
    """
    Deterministic (no LLM) fallback explanation from actual SHAP values.
    Used when LLM is unavailable.
    """
    if not ctx.shap_values:
        prob_str = f"{ctx.prediction_probability:.4f}" if ctx.prediction_probability is not None else "N/A"
        return (
            f"Observation #{ctx.observation_index}: "
            f"The model predicted '{ctx.predicted_class_label or ctx.prediction}' "
            f"(probability: {prob_str}). "
            "SHAP explanations are not available for this model."
        )

    sorted_shap = sorted(ctx.shap_values.items(), key=lambda x: abs(x[1]), reverse=True)

    lines = [
        f"## Model Prediction Explanation (Observation #{ctx.observation_index})",
        "",
        f"**Model**: {ctx.model_name}",
        f"**Predicted Outcome**: {ctx.predicted_class_label or ctx.prediction}",
        f"**Prediction Probability**: {f'{ctx.prediction_probability:.4f}' if ctx.prediction_probability is not None else 'N/A'}",
        f"**XAI Method**: {ctx.xai_method}",
        f"**Base Value** (expected output): {f'{ctx.base_value:.4f}' if ctx.base_value is not None else 'N/A'}",
        "",
        "### Top Feature Contributions",
        "",
        "| Feature | SHAP Value | Direction |",
        "|---------|-----------|-----------|",
    ]

    for feat, val in sorted_shap[:10]:
        direction = "increases" if val > 0 else "decreases"
        lines.append(f"| {feat} | {val:+.4f} | {direction} |")

    lines += [
        "",
        "### Interpretation",
        "",
        f"Positive SHAP values moved the explained output ({ctx.explained_output_label or 'model output'}) higher; negative values moved it lower.",
        "The magnitude of the SHAP value indicates the strength of the feature's contribution to this prediction.",
        "",
        "### Important Limitations",
        "",
        f"- These contributions reflect statistical associations learned by **{ctx.model_name}**, not causal relationships.",
        "- A high SHAP value does NOT mean the feature caused the outcome.",
        "- This is a LOCAL explanation - it describes this specific prediction, not the model's overall behaviour.",
        "- SHAP values depend on the training data distribution and model architecture.",
    ]

    return "\n".join(lines)


def build_fallback_global_explanation(
    global_importance: dict,
    model_name: str,
    problem_type: str,
    target_column: str,
) -> str:
    """Deterministic fallback for global feature importance explanation."""
    sorted_feats = sorted(global_importance.items(), key=lambda x: abs(x[1]), reverse=True)

    lines = [
        "## Global Feature Importance",
        f"**Model**: {model_name} | **Target**: {target_column}",
        "",
        "The following features are most consistently influential across all predictions,",
        "ranked by mean absolute SHAP value (higher = more consistently important):",
        "",
        "| Rank | Feature | Mean |SHAP| |",
        "|------|---------|------|",
    ]

    for rank, (feat, val) in enumerate(sorted_feats[:15], 1):
        lines.append(f"| {rank} | {feat} | {val:.4f} |")

    lines += [
        "",
        "### Limitations",
        "- Feature importance reflects model associations, NOT causal relationships.",
        "- Mean |SHAP| values may be biased toward high-cardinality features.",
        "- Consider permutation importance for a more reliable estimate.",
        "- Individual predictions may be dominated by different features.",
    ]

    return "\n".join(lines)
