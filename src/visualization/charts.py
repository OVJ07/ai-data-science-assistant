"""
src/visualization/charts.py
────────────────────────────
Plotly-based chart functions for the Streamlit UI.
All charts use actual data — no hardcoded/example values.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Color scheme ──────────────────────────────────────────────────────────────
PRIMARY_COLOR = "#4F8EF7"
SECONDARY_COLOR = "#F7844F"
POSITIVE_COLOR = "#2ECC71"
NEGATIVE_COLOR = "#E74C3C"
NEUTRAL_COLOR = "#95A5A6"
BG_COLOR = "#1E1E2E"
PAPER_COLOR = "#1E1E2E"


def plot_target_distribution(
    series: pd.Series,
    problem_type: str,
    title: str = "Target Distribution",
) -> go.Figure:
    """Bar chart or histogram of target variable."""
    if "classification" in problem_type:
        vc = series.value_counts().reset_index()
        vc.columns = ["class", "count"]
        fig = px.bar(
            vc, x="class", y="count", color="class",
            title=title, text="count",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="outside")
    else:
        fig = px.histogram(
            series.dropna(), x=series.name or "value",
            title=title, nbins=30,
            color_discrete_sequence=[PRIMARY_COLOR],
        )
    _style(fig)
    return fig


def plot_missing_values(missing_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of missing values percentage."""
    if missing_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No missing values!", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        _style(fig)
        return fig

    fig = px.bar(
        missing_df.sort_values("Missing %"),
        x="Missing %", y="Column", orientation="h",
        title="Missing Values by Column",
        color="Missing %",
        color_continuous_scale="RdYlGn_r",
        text="Missing %",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    _style(fig)
    return fig


def plot_correlation_matrix(corr_df: pd.DataFrame) -> go.Figure:
    """Heatmap of the correlation matrix."""
    fig = px.imshow(
        corr_df,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Matrix",
        text_auto=".2f",
    )
    fig.update_layout(
        xaxis=dict(tickangle=-45),
        coloraxis_colorbar=dict(title="Pearson r"),
    )
    _style(fig)
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    numerical_cols: List[str],
    max_cols: int = 9,
) -> go.Figure:
    """Grid of histograms for numerical features."""
    cols = numerical_cols[:max_cols]
    n = len(cols)
    if n == 0:
        return go.Figure()
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols)
    for i, col in enumerate(cols):
        row, col_idx = divmod(i, ncols)
        trace = go.Histogram(x=df[col].dropna(), name=col, showlegend=False,
                             marker_color=PRIMARY_COLOR, opacity=0.75)
        fig.add_trace(trace, row=row + 1, col=col_idx + 1)

    fig.update_layout(title_text="Numerical Feature Distributions", height=280 * nrows)
    _style(fig)
    return fig


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: Optional[List[str]] = None,
) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    import numpy as np
    cm_arr = np.array(cm)
    labels = class_names or [str(i) for i in range(len(cm_arr))]
    fig = px.imshow(
        cm_arr, x=labels, y=labels,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
    )
    _style(fig)
    return fig


def plot_global_shap(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Global Feature Importance (Mean |SHAP|)",
) -> go.Figure:
    """Horizontal bar chart of global SHAP importance."""
    df = importance_df.head(top_n).copy()
    fig = px.bar(
        df.iloc[::-1],
        x="mean_abs_shap", y="feature", orientation="h",
        title=title,
        color="mean_abs_shap",
        color_continuous_scale="Blues",
        text="mean_abs_shap",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False)
    _style(fig)
    return fig


def plot_local_shap(
    shap_dict: Dict[str, float],
    prediction: Any,
    prediction_probability: Optional[float],
    observation_index: int,
    top_n: int = 12,
) -> go.Figure:
    """Waterfall-style bar chart for local SHAP values."""
    sorted_items = sorted(shap_dict.items(), key=lambda x: x[1])[-top_n:]
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]

    title = (
        f"Local SHAP — Observation #{observation_index} "
        f"(Predicted: {prediction}"
        + (f", P={prediction_probability:.3f}" if prediction_probability is not None else "")
        + ")"
    )

    fig = go.Figure(go.Bar(
        x=values, y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="SHAP Value (contribution to prediction)",
        shapes=[dict(type="line", x0=0, x1=0, y0=-0.5, y1=len(features) - 0.5,
                     line=dict(color="white", width=1, dash="dash"))],
    )
    _style(fig)
    return fig


def plot_cv_results(
    cv_results: Dict[str, Any],
    scoring: str,
    title: str = "Cross-Validation Results",
) -> go.Figure:
    """Box plot or scatter of CV scores across configurations."""
    mean_key = f"mean_test_score"
    std_key = f"std_test_score"

    if mean_key not in cv_results:
        return go.Figure()

    means = list(cv_results[mean_key])
    stds = list(cv_results.get(std_key, [0] * len(means)))

    fig = go.Figure(go.Scatter(
        y=means,
        error_y=dict(type="data", array=stds, visible=True),
        mode="markers",
        marker=dict(size=8, color=PRIMARY_COLOR),
        name="CV Score",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Configuration Index",
        yaxis_title=f"CV Score ({scoring})",
    )
    _style(fig)
    return fig


def plot_comparison_bar(
    baseline_metrics: Dict[str, float],
    rag_metrics: Dict[str, float],
    metric_keys: Optional[List[str]] = None,
) -> go.Figure:
    """Grouped bar chart comparing baseline vs RAG metrics."""
    keys = metric_keys or [
        k for k in baseline_metrics
        if isinstance(baseline_metrics.get(k), (int, float))
        and k not in ("confusion_matrix",)
    ]

    b_vals = [baseline_metrics.get(k) for k in keys]
    r_vals = [rag_metrics.get(k) for k in keys]

    # Filter out None
    valid = [(k, b, r) for k, b, r in zip(keys, b_vals, r_vals)
             if b is not None and r is not None]
    if not valid:
        return go.Figure()

    keys_v, b_vals_v, r_vals_v = zip(*valid)

    fig = go.Figure(data=[
        go.Bar(name="Baseline", x=list(keys_v), y=list(b_vals_v),
               marker_color=NEUTRAL_COLOR),
        go.Bar(name="RAG-Guided", x=list(keys_v), y=list(r_vals_v),
               marker_color=PRIMARY_COLOR),
    ])
    fig.update_layout(
        barmode="group",
        title="Baseline vs RAG-Guided Performance Comparison",
        yaxis_title="Score",
    )
    _style(fig)
    return fig


def plot_class_distribution(class_dist: Dict[str, int]) -> go.Figure:
    """Pie chart of class distribution."""
    labels = list(class_dist.keys())
    values = list(class_dist.values())
    fig = px.pie(
        names=labels, values=values,
        title="Class Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    _style(fig)
    return fig


def plot_retrieval_scores(docs) -> go.Figure:
    """Bar chart of retrieved document relevance scores."""
    if not docs:
        return go.Figure()
    titles = [d.title[:30] for d in docs]
    scores = [d.relevance_score for d in docs]
    fig = px.bar(
        x=scores, y=titles, orientation="h",
        title="Retrieved Document Relevance Scores",
        color=scores, color_continuous_scale="Blues",
        text=[f"{s:.3f}" for s in scores],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False)
    _style(fig)
    return fig


# ── Shared styling ────────────────────────────────────────────────────────────

def _style(fig: go.Figure, height: int = 400) -> None:
    """Apply consistent dark-theme styling."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=PAPER_COLOR,
        height=height,
        font=dict(family="Inter, sans-serif", size=12, color="white"),
        margin=dict(l=10, r=10, t=50, b=10),
        title_font=dict(size=15, color="white"),
    )
