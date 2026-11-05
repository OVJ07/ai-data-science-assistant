"""
app.py
───────
AI Data Science Assistant — Streamlit Interface
16-section workflow from data upload to RAG-grounded XAI explanations.
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ── Lazy imports (after path setup) ─────────────────────────────────────────
from config.settings import (
    RANDOM_STATE,
    SUPPORTED_EMBEDDING_MODELS,
    EMBEDDING_MODEL,
    OLLAMA_EMBED_BASE_URL,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    LLM_PROVIDER,
)
from src.data.loader import DataLoader
from src.data.analyzer import DatasetAnalyzer
from src.data.profiler import DatasetProfile
from src.ml.model_factory import ModelFactory
from src.ml.auto_selector import select_model_auto
from src.ml.baseline import BaselineStrategy
from src.ml.comparison import ModelComparator
from src.ml.trainer import ModelTrainer
from src.rag.query_generator import QueryGenerator
from src.rag.recommendation import RecommendationEngine, SearchSpaceConfig
from src.rag.retriever import KnowledgeRetriever
from src.rag.retriever_registry import (
    RetrieverRegistry,
    STRATEGY_TFIDF,
    STRATEGY_DENSE,
    STRATEGY_HYBRID,
)
from src.rag.benchmark import RetrievalBenchmark, compare_reports
from src.rag.recommendation_evaluator import RecommendationEvaluator
from src.ml.multi_retriever_comparison import MultiRetrieverComparator
from src.rag.explanation import ExplanationGrounder
from src.xai.explainer import XAIExplainer
from src.llm.provider import create_llm_service
from src.reporting.report_generator import ReportGenerator
from src.visualization import charts

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Science Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title { font-size: 2.4rem; font-weight: 800; color: #4F8EF7; margin-bottom: 0; }
  .subtitle { color: #a0a0b8; font-size: 1rem; margin-bottom: 2rem; }
  .section-card { background: #1e1e2e; border-left: 4px solid #4F8EF7; padding: 16px; border-radius: 6px; margin: 8px 0; }
  .metric-highlight { font-size: 2rem; font-weight: bold; color: #4F8EF7; }
  .badge-rag { background: #1a3a6e; color: #4F8EF7; padding: 2px 12px; border-radius: 12px; font-size: 0.8rem; }
  .badge-baseline { background: #2e2e2e; color: #a0a0b8; padding: 2px 12px; border-radius: 12px; font-size: 0.8rem; }
  .warn-banner { background: #3e2e00; border: 1px solid #F39C12; border-radius: 6px; padding: 12px 16px; margin: 8px 0; }
  .info-banner { background: #0a2e4e; border: 1px solid #4F8EF7; border-radius: 6px; padding: 12px 16px; margin: 8px 0; }
  .shap-positive { color: #2ECC71; font-weight: bold; }
  .shap-negative { color: #E74C3C; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Session state helpers ─────────────────────────────────────────────────────

def ss_init():
    """Initialize session state keys."""
    defaults = {
        "df": None,
        "profile": None,
        "analyzer": None,
        "target_column": None,
        "problem_type": None,
        "retriever": None,
        "retriever_built": False,
        "retrieval_strategy": STRATEGY_TFIDF,
        "embedding_model": EMBEDDING_MODEL,
        "retriever_registry": None,
        "retriever_status": None,
        "hybrid_diagnostics": [],
        "recommendation_eval_report": None,
        "benchmark_report_dict": None,
        "benchmark_markdown_comparison": None,
        "multi_retriever_comp_result": None,
        "queries": [],
        "query_gen_result": None,
        "retrieval_trace": [],
        "recommendations": [],
        "retrieved_docs": [],
        "search_space_config": None,
        "selected_model": None,
        "auto_selection_result": None,
        "training_result": None,
        "comparison_result": None,
        "xai_explainer": None,
        "xai_global_result": None,
        "xai_local_result": None,
        "grounded_context": None,
        "llm_explanation": None,
        "llm_service": None,
        "report_data": {},
        "dataset_name": "",
        "exclude_columns": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


ss_init()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## AI Data Science Assistant")
    st.markdown("---")
    st.markdown("""
**Sections**
1. Upload Dataset
2. Dataset Overview
3. Data Quality
4. Exploratory Analysis
5. Target and Problem Type
6. RAG Retrieval
7. Preprocessing Recommendations
8. Model Selection
9. Hyperparameter Search
10. Cross-Validation Results
11. Model Evaluation
12. Baseline vs RAG
13. Global Explainability
14. Individual Explanation
15. RAG-Grounded Explanation
16. Download Report
    """)
    st.markdown("---")
    st.caption("Built with Streamlit · SHAP · TF-IDF RAG")
    if st.session_state.get("llm_service"):
        llm = st.session_state["llm_service"]
        icon = "[LLM Active]" if llm.is_llm_available else "[Deterministic Mode]"
        st.caption(f"{icon}: {llm.provider_name}")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">AI Data Science Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">RAG-Guided Hyperparameter Optimization · SHAP Explainability · Grounded Natural Language Explanations</p>', unsafe_allow_html=True)
st.markdown("---")


# =============================================================================
# SECTION 1: DATASET UPLOAD
# =============================================================================
st.markdown("## 1. Dataset Upload")

uploaded_file = st.file_uploader(
    "Upload a CSV or TSV file",
    type=["csv", "tsv"],
    help="Upload your dataset. Columns with >95% unique values will be flagged as potential ID columns.",
)

if uploaded_file is not None:
    loader = DataLoader()
    try:
        with st.spinner(f"Loading {uploaded_file.name}..."):
            df = loader.load_from_file_object(uploaded_file, uploaded_file.name)
        st.session_state["dataset_name"] = uploaded_file.name

        if (
            st.session_state["df"] is None
            or not df.equals(st.session_state["df"])
        ):
            st.session_state["df"] = df
            for key in [
                "profile", "analyzer", "target_column", "problem_type",
                "queries", "recommendations", "retrieved_docs", "retrieval_trace", "search_space_config",
                "training_result", "comparison_result", "xai_explainer",
                "xai_global_result", "xai_local_result", "grounded_context",
                "llm_explanation", "exclude_columns",
            ]:
                st.session_state[key] = None if key not in [
                    "queries", "recommendations", "retrieved_docs", "retrieval_trace", "exclude_columns"
                ] else []

        st.success(f"Loaded: **{st.session_state['dataset_name']}** — {df.shape[0]} rows x {df.shape[1]} columns")

    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

df = st.session_state.get("df")
if df is None:
    st.info("Upload a CSV or TSV file above to begin.")
    st.stop()


# =============================================================================
# SECTION 2: DATASET OVERVIEW
# =============================================================================
st.markdown("---")
st.markdown("## 2. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{df.shape[0]:,}")
col2.metric("Columns", df.shape[1])
col3.metric("Duplicates", int(df.duplicated().sum()))
col4.metric("Memory (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")

with st.expander("Preview (first 10 rows)", expanded=False):
    st.dataframe(df.head(10), width="stretch")

with st.expander("Data Types", expanded=False):
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(d) for d in df.dtypes],
        "Non-Null Count": df.notna().sum().values,
        "Unique Values": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(dtype_df, width="stretch")


# =============================================================================
# SECTION 3: DATA QUALITY
# =============================================================================
st.markdown("---")
st.markdown("## 3. Data Quality")

analyzer = DatasetAnalyzer()
profile = analyzer.analyze(df)
st.session_state["analyzer"] = analyzer

missing_df = analyzer.get_missing_value_report()
if missing_df.empty:
    st.success("No missing values detected.")
else:
    st.warning(f"Missing values in {len(missing_df)} column(s).")
    st.plotly_chart(charts.plot_missing_values(missing_df), width="stretch")
    with st.expander("Missing Value Details"):
        st.dataframe(missing_df, width="stretch")

n_dup = int(df.duplicated().sum())
if n_dup > 0:
    st.warning(f"{n_dup} duplicate rows found.")


# =============================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS
# =============================================================================
st.markdown("---")
st.markdown("## 4. Exploratory Data Analysis")

num_cols = profile.numerical_features
cat_cols = profile.categorical_features

tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Column Summaries"])

with tab1:
    if num_cols:
        st.plotly_chart(
            charts.plot_feature_distributions(df, num_cols[:9]),
            width="stretch",
        )
    else:
        st.info("No numerical features found.")

    if cat_cols:
        st.markdown("#### Categorical Feature Value Counts")
        for col in cat_cols[:3]:
            vc = df[col].value_counts().head(10)
            st.bar_chart(vc)

with tab2:
    corr = analyzer.get_correlation_matrix()
    if corr is not None:
        st.plotly_chart(charts.plot_correlation_matrix(corr), width="stretch")
    else:
        st.info("Correlation matrix requires at least 2 numerical columns.")

with tab3:
    summaries = analyzer.get_all_column_summaries()
    if summaries:
        summary_df = pd.DataFrame(summaries)
        st.dataframe(summary_df, width="stretch")


# =============================================================================
# SECTION 5: TARGET AND PROBLEM TYPE
# =============================================================================
st.markdown("---")
st.markdown("## 5. Target and Problem Type")

col1, col2 = st.columns(2)
with col1:
    target_column = st.selectbox(
        "Select Target Column",
        options=["(none)"] + list(df.columns),
        index=0,
        help="Select the column you want to predict.",
    )
with col2:
    problem_type_override = st.selectbox(
        "Problem Type (auto-detect or override)",
        options=["Auto-detect", "binary_classification", "multiclass_classification", "regression"],
        index=0,
    )

if target_column == "(none)":
    st.info("Select a target column to continue.")
    st.stop()

pt = None if problem_type_override == "Auto-detect" else problem_type_override

profile = analyzer.analyze(df, target_column=target_column, problem_type=pt)
st.session_state["profile"] = profile
st.session_state["target_column"] = target_column
st.session_state["problem_type"] = profile.problem_type

col1, col2, col3, col4 = st.columns(4)
col1.metric("Problem Type", profile.problem_type or "Unknown")
col2.metric("Classes", profile.n_classes or "N/A")
col3.metric("Imbalanced", "Yes" if profile.is_imbalanced else "No")
col4.metric("Imbalance Ratio", f"{profile.imbalance_ratio:.3f}" if profile.imbalance_ratio else "N/A")

target_info = analyzer.get_target_distribution()
if target_info:
    fig = charts.plot_target_distribution(df[target_column].dropna(), profile.problem_type or "classification")
    st.plotly_chart(fig, width="stretch")

for w in profile.warnings:
    st.markdown(f'<div class="warn-banner">{w}</div>', unsafe_allow_html=True)

st.markdown("#### Column Exclusion")
exclude_suggestions = profile.id_like_features + profile.constant_features
default_exclude = list(set(exclude_suggestions))
exclude_cols = st.multiselect(
    "Exclude columns from training (auto-suggests ID-like and constant features):",
    options=[c for c in df.columns if c != target_column],
    default=[c for c in default_exclude if c in df.columns and c != target_column],
)
st.session_state["exclude_columns"] = exclude_cols


# =============================================================================
# SECTION 6: RAG RETRIEVAL, RETRIEVAL BENCHMARK, AND RECOMMENDATIONS
# =============================================================================
st.markdown("---")
st.markdown("## 6. RAG Retrieval & Benchmark")
st.caption(
    "Configure retrieval strategies (TF-IDF, Dense, Hybrid RRF), benchmark retrieval performance, "
    "and generate grounded preprocessing and search space recommendations."
)

tab_retrieval, tab_benchmark, tab_multi_comp = st.tabs([
    "Active RAG Retrieval",
    "Retrieval Strategy Benchmark",
    "Multi-Retriever Training Comparison",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: ACTIVE RAG RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────
with tab_retrieval:
    # ── Strategy and Model Selectors ──
    col_strat1, col_strat2, col_strat3 = st.columns([1.5, 1.5, 1])
    with col_strat1:
        selected_strategy = st.selectbox(
            "Retrieval Strategy",
            options=[STRATEGY_TFIDF, STRATEGY_DENSE, STRATEGY_HYBRID],
            index=[STRATEGY_TFIDF, STRATEGY_DENSE, STRATEGY_HYBRID].index(
                st.session_state.get("retrieval_strategy", STRATEGY_TFIDF)
            ),
            format_func=lambda s: {
                STRATEGY_TFIDF: "TF-IDF (Sparse Baseline)",
                STRATEGY_DENSE: "Dense Semantic (Ollama Embedding)",
                STRATEGY_HYBRID: "Hybrid (Reciprocal Rank Fusion)",
            }.get(s, s),
            help="TF-IDF uses lexical overlap. Dense uses local Ollama embeddings. Hybrid fuses both with RRF.",
        )
        st.session_state["retrieval_strategy"] = selected_strategy

    with col_strat2:
        embedding_options = list(SUPPORTED_EMBEDDING_MODELS)
        current_emb = st.session_state.get("embedding_model", EMBEDDING_MODEL)
        if current_emb not in embedding_options:
            embedding_options.append(current_emb)
        selected_emb_model = st.selectbox(
            "Ollama Embedding Model",
            options=embedding_options,
            index=embedding_options.index(current_emb) if current_emb in embedding_options else 0,
            disabled=(selected_strategy == STRATEGY_TFIDF),
            help="Local Ollama embedding model for dense and hybrid retrieval.",
        )
        st.session_state["embedding_model"] = selected_emb_model

    with col_strat3:
        force_rebuild = st.checkbox("Force Re-index", value=False, help="Re-embed and rebuild cache from scratch.")

    # ── Active Runtime Configuration Display ──
    with st.container(border=True):
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("**Active LLM**")
            st.caption(f"Provider: `{LLM_PROVIDER}`\n\nModel: `{OPENAI_MODEL}`\n\nEndpoint: `{OPENAI_BASE_URL}`")
        with col_c2:
            st.markdown("**Active Retriever**")
            st.caption(f"Strategy: `{selected_strategy.upper()}`\n\nThreshold: `min_score=0.05`\n\nCandidate cutoff: `k=5`")
        with col_c3:
            st.markdown("**Active Embedding Model**")
            if selected_strategy == STRATEGY_TFIDF:
                st.caption("Not used in TF-IDF mode")
            else:
                st.caption(f"Model: `{selected_emb_model}`\n\nEndpoint: `{OLLAMA_EMBED_BASE_URL}`\n\nCache: `.embedding_cache/`")

    with st.container(horizontal=True, horizontal_alignment="right"):
        use_llm_queries = st.checkbox(
            "Allow LLM query enhancement for this run",
            value=False,
            help="Automatic selection remains local-first. Remote providers require LLM_ALLOW_REMOTE_DATA=true.",
        )

    if st.button("Run RAG Retrieval", type="primary", width="stretch"):
        with st.spinner(f"Building {selected_strategy.upper()} index and retrieving documents..."):
            registry = RetrieverRegistry()
            status = registry.build(
                strategy=selected_strategy,
                embedding_model=selected_emb_model,
                force_rebuild=force_rebuild,
            )
            st.session_state["retriever_registry"] = registry
            st.session_state["retriever_status"] = status
            st.session_state["retriever"] = registry._active_retriever()
            st.session_state["retriever_built"] = status.is_built

            if status.error:
                st.error(f"Retriever build notice: {status.error}")
                if "Ollama" in status.error or selected_strategy in (STRATEGY_DENSE, STRATEGY_HYBRID):
                    st.info(
                        f"**Action required:** Ensure Ollama is running and the embedding model is installed:\n\n"
                        f"```bash\nollama pull {selected_emb_model}\nollama serve\n```\n"
                        f"The application has safely preserved TF-IDF fallback capability."
                    )

            qg = QueryGenerator()
            qg_result = qg.generate_queries(
                profile,
                model_name=None,
                use_llm=True if use_llm_queries else None,
            )
            st.session_state["query_gen_result"] = qg_result
            st.session_state["queries"] = qg_result.queries

            if status.is_built:
                docs, trace_or_diags = registry.retrieve_multiple_queries_with_provenance(
                    qg_result.queries, global_top_k=6
                )
                st.session_state["retrieved_docs"] = docs

                if selected_strategy == STRATEGY_HYBRID:
                    st.session_state["hybrid_diagnostics"] = trace_or_diags
                    st.session_state["retrieval_trace"] = []
                else:
                    st.session_state["retrieval_trace"] = trace_or_diags
                    st.session_state["hybrid_diagnostics"] = []

                rec_engine = RecommendationEngine(registry)
                recs, recommendation_docs = rec_engine.generate_recommendations(
                    profile, qg_result.queries, retrieved_docs=docs
                )
                st.session_state["recommendations"] = recs

                search_config = rec_engine.generate_search_space(
                    profile, "", recs, recommendation_docs, is_rag_guided=True
                )
                st.session_state["search_space_config"] = search_config

                # Evaluate recommendation quality
                rec_evaluator = RecommendationEvaluator()
                eval_report = rec_evaluator.evaluate_recommendations(
                    strategy_name=selected_strategy,
                    profile=profile,
                    retrieved_docs=docs,
                    recommendations=recs,
                    search_config=search_config,
                )
                st.session_state["recommendation_eval_report"] = eval_report
            else:
                st.warning("Could not execute retrieval because retriever index could not be built.")

    # ── Display Retrieval Results ──
    r_status = st.session_state.get("retriever_status")
    if r_status:
        st.markdown(
            f"**Index Status:** `{'Ready' if r_status.is_built else 'Failed'}` · "
            f"Documents: `{r_status.n_documents}` · Build time: `{r_status.build_time_seconds:.3f}s`"
            + (f" · Cache hit rate: `{r_status.cache_hit_rate:.1%}`" if r_status.cache_hit_rate is not None else "")
        )

    qg_result = st.session_state.get("query_gen_result")
    if qg_result:
        badge = qg_result.strategy.replace("_", " ").upper()
        st.markdown(
            f'**Query Strategy:** <span style="background:#1a3a6e;padding:2px 8px;border-radius:8px;">'
            f'{badge}</span> &nbsp; {qg_result.selection_reason}',
            unsafe_allow_html=True,
        )
        if qg_result.fallback_occurred:
            st.warning(f"Query generation fallback: {qg_result.fallback_reason}")
        with st.expander("RAG execution diagnostics", expanded=False):
            st.caption("Safe runtime metadata only; prompts and credentials are never displayed.")
            st.json({
                "query_strategy": qg_result.strategy,
                "retriever_strategy": st.session_state.get("retrieval_strategy"),
                "embedding_model": st.session_state.get("embedding_model"),
                "generation_duration_seconds": qg_result.generation_duration_seconds,
                "llm_request": qg_result.llm_diagnostics or "No LLM request was made.",
            })
            st.markdown("**Deterministic queries**")
            st.write(qg_result.deterministic_queries or "None")
            st.markdown("**LLM-generated queries (validated)**")
            st.write(qg_result.llm_queries or "None")

    queries = st.session_state.get("queries", [])
    retrieved_docs = st.session_state.get("retrieved_docs", [])

    if queries:
        with st.expander(f"Generated Retrieval Queries ({len(queries)} total)", expanded=False):
            for i, q in enumerate(queries, 1):
                st.markdown(f"**{i}.** {q}")

    if retrieved_docs:
        st.plotly_chart(charts.plot_retrieval_scores(retrieved_docs), width="stretch")
        st.markdown(f"**Retrieved {len(retrieved_docs)} relevant documents using {st.session_state.get('retrieval_strategy', '').upper()}:**")
        for doc in retrieved_docs:
            with st.expander(f"{doc.title} (relevance score: {doc.relevance_score:.4f}, doc_id: {doc.doc_id})"):
                st.markdown(f"**Source file:** `{doc.source_file}`")
                st.markdown(doc.snippet)

        # Hybrid RRF Diagnostics
        hybrid_diags = st.session_state.get("hybrid_diagnostics", [])
        if hybrid_diags:
            with st.expander("Hybrid RRF Fusion Diagnostics", expanded=False):
                st.caption("Reciprocal Rank Fusion components and rank positions:")
                diag_rows = []
                for d in hybrid_diags:
                    diag_rows.append({
                        "Document": d.title,
                        "Doc ID": d.doc_id,
                        "RRF Score": f"{d.rrf_score:.6f}",
                        "TF-IDF Rank": d.tfidf_rank if d.tfidf_rank is not None else "Not in top candidates",
                        "TF-IDF Score": f"{d.tfidf_score:.4f}" if d.tfidf_score is not None else "N/A",
                        "Dense Rank": d.dense_rank if d.dense_rank is not None else "Not in top candidates",
                        "Dense Score": f"{d.dense_score:.4f}" if d.dense_score is not None else "N/A",
                        "Contributing Sources": ", ".join(d.sources),
                    })
                st.dataframe(pd.DataFrame(diag_rows), width="stretch")

        trace = st.session_state.get("retrieval_trace", [])
        if trace:
            with st.expander("TF-IDF Retrieval Provenance", expanded=False):
                st.caption("Query matches behind the retrieved documents:")
                st.dataframe(pd.DataFrame([
                    {"Query": item.query, "Document": item.title, "Document ID": item.doc_id, "TF-IDF score": item.relevance_score}
                    for item in trace
                ]), width="stretch")

        # Recommendation Audit
        eval_report = st.session_state.get("recommendation_eval_report")
        if eval_report:
            with st.expander("Recommendation Quality Audit", expanded=False):
                col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                col_e1.metric("Grounded Support", f"{eval_report.support_rate:.1%}")
                col_e2.metric("Technical Validity", f"{eval_report.validity_rate:.1%}")
                col_e3.metric("Dataset Relevance", f"{eval_report.relevance_rate:.1%}")
                col_e4.metric("Pipeline Applied", f"{eval_report.application_rate:.1%}")
                audit_rows = [
                    {
                        "Technique": item.technique,
                        "Supporting Doc": item.supporting_doc,
                        "Supported by Retrieval": "Yes" if item.is_supported_by_retrieval else "No (Rule fallback)",
                        "Valid": "Yes" if item.is_technically_valid else "No",
                        "Relevant": "Yes" if item.is_dataset_relevant else "No",
                        "Pipeline Status": item.application_status,
                        "Notes": "; ".join(item.notes) if item.notes else "All checks passed",
                    }
                    for item in eval_report.evaluations
                ]
                st.dataframe(pd.DataFrame(audit_rows), width="stretch")
    elif st.session_state.get("retriever_built") is False:
        st.info("Run RAG retrieval above to view retrieved context and recommendations.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: RETRIEVAL STRATEGY BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
with tab_benchmark:
    st.markdown("### Reproducible RAG Retrieval Benchmark")
    st.caption(
        "Directly compares TF-IDF, Dense Semantic, and Hybrid RRF strategies across 12 fixed benchmark queries "
        "covering categorical encoding, class imbalance, imputation, scaling, hyperparameter tuning, metrics, and XAI. "
        "Evaluated on ground-truth document relevance without modifying model training."
    )

    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        bench_emb_model = st.selectbox(
            "Benchmark Embedding Model",
            options=list(SUPPORTED_EMBEDDING_MODELS),
            index=0,
            key="bench_emb_select",
        )
    with col_b2:
        run_all_btn = st.button("Run Full Retrieval Benchmark", width="stretch")

    if run_all_btn:
        with st.spinner("Running retrieval benchmark across all 3 strategies..."):
            benchmark = RetrievalBenchmark()
            reports = benchmark.run_all(embedding_model=bench_emb_model)
            st.session_state["benchmark_report_dict"] = reports
            markdown_comp = compare_reports(reports)
            st.session_state["benchmark_markdown_comparison"] = markdown_comp

    reports = st.session_state.get("benchmark_report_dict")
    if reports:
        st.markdown("#### Retrieval Performance Summary")
        md_table = st.session_state.get("benchmark_markdown_comparison", "")
        if md_table:
            st.markdown(md_table)

        # Metric bar chart comparison
        bench_chart_data = []
        for strat, rep in reports.items():
            if not rep.errors:
                bench_chart_data.append({
                    "Strategy": rep.strategy_name,
                    "Recall@5": rep.mean_recall_at_5,
                    "Precision@5": rep.mean_precision_at_5,
                    "MRR": rep.mrr,
                    "nDCG@5": rep.mean_ndcg_at_5,
                })
        if bench_chart_data:
            st.dataframe(pd.DataFrame(bench_chart_data), width="stretch")

        with st.expander("Per-Query Diagnostic Results", expanded=False):
            bench_strategy_inspect = st.selectbox(
                "Inspect Strategy Queries",
                options=list(reports.keys()),
                key="bench_strat_inspect",
            )
            inspect_report = reports[bench_strategy_inspect]
            query_rows = []
            for qr in inspect_report.per_query_results:
                query_rows.append({
                    "Query ID": qr.query_id,
                    "Query Topic": qr.topic,
                    "Query Text": qr.query_text,
                    "Relevant Docs": ", ".join(qr.relevant_doc_ids),
                    "Retrieved Docs": ", ".join(qr.retrieved_doc_ids),
                    "Recall@5": f"{qr.recall_at_5:.2f}",
                    "Precision@5": f"{qr.precision_at_5:.2f}",
                    "MRR": f"{qr.mrr:.2f}",
                    "nDCG@5": f"{qr.ndcg_at_5:.2f}",
                    "Latency (ms)": f"{qr.latency_ms:.1f}",
                })
            st.dataframe(pd.DataFrame(query_rows), width="stretch")
    else:
        st.info("Click 'Run Full Retrieval Benchmark' to compare TF-IDF, Dense, and Hybrid retrieval.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: MULTI-RETRIEVER TRAINING COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab_multi_comp:
    st.markdown("### Four-Way Model Training Comparison")
    st.caption(
        "Runs Baseline (no RAG), TF-IDF RAG, Dense RAG, and Hybrid RAG under identical splits and random seed (42). "
        "Verifies whether retrieval strategy differences actually impact final machine learning model performance."
    )

    if st.button("Run Four-Way Comparison", width="stretch"):
        with st.spinner("Training baseline, TF-IDF, Dense, and Hybrid pipelines on identical data..."):
            multi_comp = MultiRetrieverComparator()
            m_res = multi_comp.run_comparison(
                df=df,
                profile=profile,
                model_name=model_name if "model_name" in locals() and model_name else "Decision Tree",
                embedding_model=selected_emb_model,
                exclude_columns=st.session_state.get("exclude_columns", []),
            )
            st.session_state["multi_retriever_comp_result"] = m_res

    m_res = st.session_state.get("multi_retriever_comp_result")
    if m_res:
        st.markdown(f"**Model:** `{m_res.model_name}` · **Observations:** `{m_res.dataset_rows}` · **Random Seed:** `{m_res.random_state}`")
        if m_res.comparison_table:
            st.dataframe(pd.DataFrame(m_res.comparison_table), width="stretch")

        st.markdown("#### Measured Assessment")
        st.info(m_res.honest_summary)

        with st.expander("Per-Strategy Provenance & Retrieved Document Breakdown", expanded=False):
            st.json({
                "Baseline": {
                    "config_hash": m_res.baseline_run.config_hash,
                    "applied_configuration": m_res.baseline_run.applied_configuration,
                },
                "TF-IDF": {
                    "config_hash": m_res.tfidf_run.config_hash,
                    "retrieved_docs": [d.doc_id for d in m_res.tfidf_run.retrieved_documents],
                    "search_space_reasoning": m_res.tfidf_run.search_space_reasoning,
                    "pred_agreement_with_baseline": f"{m_res.tfidf_run.prediction_agreement_with_baseline:.1%}" if m_res.tfidf_run.prediction_agreement_with_baseline is not None else "N/A",
                },
                "Dense": {
                    "config_hash": m_res.dense_run.config_hash,
                    "retrieved_docs": [d.doc_id for d in m_res.dense_run.retrieved_documents],
                    "search_space_reasoning": m_res.dense_run.search_space_reasoning,
                    "pred_agreement_with_baseline": f"{m_res.dense_run.prediction_agreement_with_baseline:.1%}" if m_res.dense_run.prediction_agreement_with_baseline is not None else "N/A",
                },
                "Hybrid": {
                    "config_hash": m_res.hybrid_run.config_hash,
                    "retrieved_docs": [d.doc_id for d in m_res.hybrid_run.retrieved_documents],
                    "search_space_reasoning": m_res.hybrid_run.search_space_reasoning,
                    "pred_agreement_with_baseline": f"{m_res.hybrid_run.prediction_agreement_with_baseline:.1%}" if m_res.hybrid_run.prediction_agreement_with_baseline is not None else "N/A",
                },
            })
    else:
        st.info("Click 'Run Four-Way Comparison' to evaluate downstream effects on model training.")


# =============================================================================
# SECTION 7: PREPROCESSING RECOMMENDATIONS
# =============================================================================
st.markdown("---")
st.markdown("## 7. Preprocessing Recommendations")

recommendations = st.session_state.get("recommendations", [])
if recommendations:
    for rec in recommendations:
        with st.expander(f"{rec.technique} (priority={rec.priority}; evidence={rec.evidence_status})", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Trigger:** {rec.trigger}")
                st.markdown(f"**Reason:** {rec.reason}")
                st.markdown(f"**Evidence:** `{rec.evidence_status}` · knowledge-base source: `{rec.supporting_doc}`")
            with col2:
                if rec.config:
                    st.markdown("**Proposed Configuration:**")
                    st.json(rec.config)
            if rec.limitations:
                st.markdown(f'<div class="info-banner">Limitation: {rec.limitations}</div>', unsafe_allow_html=True)
else:
    st.info("Run RAG retrieval (Section 6) to generate recommendations.")


# =============================================================================
# SECTION 8: MODEL SELECTION (MANUAL OR AUTOMATIC)
# =============================================================================
st.markdown("---")
st.markdown("## 8. Select the prediction model")
st.caption(
    "Choose a model yourself, or use Auto-select to compare eligible estimators with cross-validation on training data only. "
    "This choice controls the model trained in Sections 9–16."
)

_avail_models = (
    ModelFactory.get_classification_names()
    if "classification" in (profile.problem_type or "")
    else ModelFactory.get_regression_names()
)

col_m1, col_m2 = st.columns([2, 1])
with col_m1:
    _manual_model = st.selectbox(
        "Prediction model",
        options=_avail_models,
        index=_avail_models.index(st.session_state.get("selected_model")) if st.session_state.get("selected_model") in _avail_models else 0,
    )
with col_m2:
    if st.button("Auto-Select Model", width="stretch",
                 help="Runs CV on training data to pick the best model automatically."):
        with st.spinner("Running cross-validation to select the best model..."):
            try:
                auto_res = select_model_auto(
                    df=df, profile=profile,
                    exclude_columns=st.session_state.get("exclude_columns", []),
                )
                st.session_state["auto_selection_result"] = auto_res
                st.session_state["selected_model"] = auto_res.selected_model
                st.success(f"Auto-selected: **{auto_res.selected_model}** ({auto_res.selection_reason})")
            except Exception as exc:
                st.error(f"Auto-selection failed: {exc}")

auto_res = st.session_state.get("auto_selection_result")
if auto_res:
    with st.expander("Auto-Selection Details", expanded=False):
        st.markdown(f"**Selected:** {auto_res.selected_model}")
        st.markdown(f"**Reason:** {auto_res.selection_reason}")
        if auto_res.fallback_used:
            st.warning(f"Fallback used: {auto_res.fallback_reason}")
        rows = []
        for c in auto_res.candidates:
            rows.append({
                "Model": c.model_name,
                "CV Score": f"{c.cv_score:.4f}" if c.cv_score is not None else "N/A",
                "CV Std": f"+/-{c.cv_std:.4f}" if c.cv_std is not None else "N/A",
                "Time (s)": f"{c.training_time_s:.1f}",
                "Excluded": c.excluded_reason or "",
                "Error": c.error or "",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")

# Use auto-selected model if available, else manual selection
if st.session_state.get("auto_selection_result") and st.session_state["auto_selection_result"].selected_model:
    model_name = st.session_state["auto_selection_result"].selected_model
else:
    model_name = _manual_model
    st.session_state["selected_model"] = model_name

st.markdown(f"**Active model for training:** `{model_name}`")

# Update search space config if model changed
search_config = st.session_state.get("search_space_config")
if search_config and search_config.model_name != model_name:
    # Regenerate search space for the newly selected model
    _retriever = st.session_state.get("retriever")
    _recs = st.session_state.get("recommendations", [])
    _docs = st.session_state.get("retrieved_docs", [])
    if _retriever and _retriever.is_built:
        _rec_engine = RecommendationEngine(_retriever)
        search_config = _rec_engine.generate_search_space(
            profile, model_name, _recs, _docs, is_rag_guided=True
        )
        st.session_state["search_space_config"] = search_config

if search_config and search_config.model_name == model_name:
    st.markdown(f"**CV Scoring:** `{search_config.cv_scoring}`")
    if search_config.reasoning:
        st.markdown("**RAG-guided search space reasoning:**")
        for r in search_config.reasoning:
            st.markdown(f"- {r}")
    if search_config.application_notes:
        st.markdown("**Recommendation integration:**")
        for note in search_config.application_notes:
            st.markdown(f"- {note}")
    if search_config.recommendation_outcomes:
        with st.expander("Recommendation application audit", expanded=False):
            st.dataframe(pd.DataFrame(search_config.recommendation_outcomes), width="stretch")
    with st.expander("RAG-Guided Hyperparameter Search Space"):
        st.json(search_config.param_grid)
else:
    st.info("Run RAG retrieval (Section 6) to generate the hyperparameter search space.")


# =============================================================================
# SECTION 9: HYPERPARAMETER SEARCH
# =============================================================================
st.markdown("---")
st.markdown("## 9. Hyperparameter Search")

_recommended_num = search_config.preprocessing_config.get("numerical_imputer_strategy", "median") if search_config else "median"
_recommended_cat = search_config.preprocessing_config.get("categorical_imputer_strategy", "most_frequent") if search_config else "most_frequent"
num_imputer = st.selectbox("Numerical Imputation Strategy", ["median", "mean"], index=["median", "mean"].index(_recommended_num))
cat_imputer = st.selectbox("Categorical Imputation Strategy", ["most_frequent", "constant"], index=["most_frequent", "constant"].index(_recommended_cat))
scale_num = st.checkbox("Scale Numerical Features (StandardScaler)", value=True)

if st.button("Run RAG-Guided Training", type="primary", width="stretch"):
    if not search_config:
        st.error("Run RAG retrieval first (Section 6) to generate search space.")
    else:
        with st.spinner(f"Training {model_name} with RAG-guided hyperparameter search..."):
            trainer = ModelTrainer()
            result = trainer.train(
                df=df,
                profile=profile,
                model_name=model_name,
                param_grid=search_config.param_grid,
                cv_scoring=search_config.cv_scoring,
                use_stratified=search_config.use_stratified,
                numerical_imputer_strategy=num_imputer,
                categorical_imputer_strategy=cat_imputer,
                scale_numerical=scale_num,
                exclude_columns=exclude_cols,
                is_rag_guided=True,
                max_ohe_categories=search_config.preprocessing_config.get("max_ohe_categories"),
            )
            st.session_state["training_result"] = result

        if result.is_successful():
            st.success(f"Training complete in {result.training_time_seconds:.2f}s")
        else:
            st.error(f"Training failed: {result.error}")

tr_res = st.session_state.get("training_result")
if tr_res and tr_res.is_successful():
    with st.expander("Training Provenance & Effective Configuration Diagnostics", expanded=False):
        st.caption("Detailed provenance tracing from retrieval to fitted estimator parameters.")
        active_ret = st.session_state.get("retriever_strategy", "TF-IDF")
        st.json({
            "active_retriever": active_ret,
            "retrieved_documents_grounding": search_config.retrieved_docs if search_config else [],
            "rag_reasoning": search_config.reasoning if search_config else [],
            "effective_preprocessing": tr_res.preprocessing_config,
            "evaluated_param_grid": search_config.param_grid if search_config else {},
            "best_hyperparameters": tr_res.best_params,
            "cv_scoring": tr_res.cv_scoring,
            "cv_mean_score": round(tr_res.best_cv_score, 4) if tr_res.best_cv_score is not None else None,
            "cv_std": round(tr_res.cv_score_std, 4) if tr_res.cv_score_std is not None else None,
            "recommendation_outcomes": search_config.recommendation_outcomes if search_config else [],
        })


# =============================================================================
# SECTION 10: CROSS-VALIDATION RESULTS
# =============================================================================
st.markdown("---")
st.markdown("## 10. Cross-Validation Results")

training_result = st.session_state.get("training_result")
if training_result and training_result.is_successful():
    col1, col2, col3 = st.columns(3)
    col1.metric(
        f"Best CV Score ({training_result.cv_scoring})",
        f"{training_result.best_cv_score:.4f}" if training_result.best_cv_score is not None else "N/A"
    )
    col2.metric(
        "CV Std",
        f"+/-{training_result.cv_score_std:.4f}" if training_result.cv_score_std is not None else "N/A"
    )
    col3.metric("Configurations Tried", training_result.n_param_combinations)

    if training_result.best_params:
        st.markdown("**Best Hyperparameters:**")
        st.json(training_result.best_params)

    if training_result.cv_results:
        fig = charts.plot_cv_results(training_result.cv_results, training_result.cv_scoring)
        st.plotly_chart(fig, width="stretch")
else:
    st.info("Run hyperparameter search (Section 9) to see CV results.")


# =============================================================================
# SECTION 11: MODEL EVALUATION
# =============================================================================
st.markdown("---")
st.markdown("## 11. Model Evaluation")

if training_result and training_result.is_successful():
    metrics = training_result.test_metrics
    numeric_metrics = {
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float)) and v is not None and k != "confusion_matrix"
    }

    cols = st.columns(min(len(numeric_metrics), 4))
    for i, (key, val) in enumerate(numeric_metrics.items()):
        cols[i % 4].metric(key.replace("_", " ").title(), f"{val:.4f}")

    cm = metrics.get("confusion_matrix")
    class_names = metrics.get("class_names")
    if cm:
        st.plotly_chart(
            charts.plot_confusion_matrix(cm, class_names),
            width="stretch",
        )
else:
    st.info("Train a model (Section 9) to see evaluation results.")


# =============================================================================
# SECTION 12: BASELINE vs RAG COMPARISON
# =============================================================================
st.markdown("---")
st.markdown("## 12. Baseline vs RAG-Guided Comparison")

if st.button("Run Baseline vs RAG Comparison", width="stretch"):
    if not search_config:
        st.error("Run RAG retrieval first.")
    else:
        with st.spinner("Running baseline and RAG-guided training for fair comparison..."):
            comparator = ModelComparator()
            comp_result = comparator.compare(
                df=df,
                profile=profile,
                model_name=model_name,
                rag_search_config=search_config,
                exclude_columns=exclude_cols,
                numerical_imputer_strategy=num_imputer,
                categorical_imputer_strategy=cat_imputer,
            )
            st.session_state["comparison_result"] = comp_result

comp_result = st.session_state.get("comparison_result")
if comp_result:
    b = comp_result.baseline_result
    r = comp_result.rag_result

    if b and r and b.is_successful() and r.is_successful():
        if not comp_result.configurations_differ:
            st.warning(
                "No RAG-specific training change was applicable. Both runs use the same search space, scoring, "
                "and preprocessing, so identical predictions and metrics are expected—not a RAG performance result."
            )
        elif comp_result.rag_guidance_notes:
            st.caption("RAG-specific changes: " + " ".join(comp_result.rag_guidance_notes))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<span class="badge-baseline">BASELINE</span>', unsafe_allow_html=True)
            st.markdown(f"**Grid size:** {comp_result.baseline_grid_size} configs")
            st.markdown(f"**CV scoring:** `{b.cv_scoring}`")
            st.markdown(f"**CV mean ± std:** `{b.best_cv_score:.4f} ± {b.cv_score_std:.4f}`" if b.best_cv_score is not None and b.cv_score_std is not None else "**CV mean ± std:** unavailable")
            st.markdown(f"**Best parameters:** `{b.best_params or 'default estimator parameters'}`")
        with col2:
            st.markdown('<span class="badge-rag">RAG-GUIDED</span>', unsafe_allow_html=True)
            st.markdown(f"**Grid size:** {comp_result.rag_grid_size} configs")
            st.markdown(f"**CV scoring:** `{r.cv_scoring}`")
            st.markdown(f"**CV mean ± std:** `{r.best_cv_score:.4f} ± {r.cv_score_std:.4f}`" if r.best_cv_score is not None and r.cv_score_std is not None else "**CV mean ± std:** unavailable")
            st.markdown(f"**Best parameters:** `{r.best_params or 'default estimator parameters'}`")

        # ── Side-by-side metric comparison aligned row by row ─────────────────
        st.markdown("---")
        st.markdown("#### Test Set Metric Comparison")

        b_m = {k: v for k, v in b.test_metrics.items() if isinstance(v, (int, float)) and v is not None and k != "confusion_matrix"}
        r_m = {k: v for k, v in r.test_metrics.items() if isinstance(v, (int, float)) and v is not None and k != "confusion_matrix"}
        all_metric_keys = list(dict.fromkeys(list(b_m.keys()) + list(r_m.keys())))

        # Tabular overview for quick side-by-side scanning
        comp_table_rows = []
        for k in all_metric_keys:
            b_val = b_m.get(k)
            r_val = r_m.get(k)
            delta = (r_val - b_val) if (r_val is not None and b_val is not None) else None
            pct = (delta / abs(b_val) * 100) if (delta is not None and b_val is not None and abs(b_val) > 1e-9) else None
            comp_table_rows.append({
                "Metric": k.replace("_", " ").title(),
                "Baseline": f"{b_val:.4f}" if b_val is not None else "N/A",
                "RAG-Guided": f"{r_val:.4f}" if r_val is not None else "N/A",
                "Difference (RAG - Base)": f"{delta:+.4f}" if delta is not None else "N/A",
                "% Change": f"{pct:+.2f}%" if pct is not None else "N/A",
            })
        if comp_table_rows:
            st.dataframe(pd.DataFrame(comp_table_rows), width="stretch", hide_index=True)

        st.caption("Detailed visual metrics (aligned side-by-side):")
        hdr_col1, hdr_col2 = st.columns(2)
        with hdr_col1:
            st.markdown("**Baseline (Reference)**")
        with hdr_col2:
            st.markdown("**RAG-Guided (with Delta)**")

        for k in all_metric_keys:
            row_col1, row_col2 = st.columns(2)
            metric_title = k.replace("_", " ").title()
            b_val = b_m.get(k)
            r_val = r_m.get(k)
            delta = (r_val - b_val) if (r_val is not None and b_val is not None) else None

            with row_col1:
                st.metric(
                    label=metric_title,
                    value=f"{b_val:.4f}" if b_val is not None else "N/A",
                )
            with row_col2:
                st.metric(
                    label=metric_title,
                    value=f"{r_val:.4f}" if r_val is not None else "N/A",
                    delta=f"{delta:+.4f}" if delta is not None else None,
                )

        numeric_keys = [
            k for k in b.test_metrics
            if isinstance(b.test_metrics.get(k), (int, float)) and k != "confusion_matrix"
        ]
        st.plotly_chart(
            charts.plot_comparison_bar(b.test_metrics, r.test_metrics, numeric_keys),
            width="stretch",
        )

        st.markdown("#### Honest Assessment")
        st.info(comp_result.honest_assessment)
        with st.expander("Comparison diagnostics", expanded=False):
            st.caption("Both pipelines use the same held-out rows. These are the effective, fitted run inputs—not UI labels.")
            st.json({
                "dataset_fingerprint": comp_result.dataset_fingerprint,
                "baseline": {
                    "config_hash": comp_result.baseline_config_hash,
                    "prediction_hash": comp_result.baseline_prediction_hash,
                    "configuration": comp_result.baseline_configuration,
                    "effective_preprocessing": b.preprocessing_config,
                    "best_parameters": b.best_params,
                    "effective_estimator_parameters": b.effective_estimator_params,
                },
                "rag_guided": {
                    "config_hash": comp_result.rag_config_hash,
                    "prediction_hash": comp_result.rag_prediction_hash,
                    "configuration": comp_result.rag_configuration,
                    "effective_preprocessing": r.preprocessing_config,
                    "best_parameters": r.best_params,
                    "effective_estimator_parameters": r.effective_estimator_params,
                    "recommendation_outcomes": search_config.recommendation_outcomes if search_config else [],
                },
                "same_held_out_observations": comp_result.test_indices_match,
                "effective_configurations_differ": comp_result.configurations_differ,
                "prediction_agreement": f"{comp_result.prediction_agreement:.1%}" if comp_result.prediction_agreement is not None else "N/A",
            })
        if comp_result.test_indices_match:
            if comp_result.prediction_agreement is not None:
                st.caption(
                    f"Same held-out observations: yes. Prediction agreement: {comp_result.prediction_agreement:.1%}. "
                    f"Identical arrays: {'yes' if comp_result.predictions_identical else 'no'}."
                )
            elif comp_result.max_prediction_difference is not None:
                st.caption(
                    f"Same held-out observations: yes. Maximum absolute prediction difference: "
                    f"{comp_result.max_prediction_difference:.6g}. Identical arrays: "
                    f"{'yes' if comp_result.predictions_identical else 'no'}."
                )
        else:
            st.error("Comparison integrity check failed: held-out observations differ.")
else:
    st.info("Click 'Run Baseline vs RAG Comparison' to compare approaches.")


# =============================================================================
# SECTION 13: GLOBAL EXPLAINABILITY
# =============================================================================
st.markdown("---")
st.markdown("## 13. Global Explainability (SHAP)")

if training_result and training_result.is_successful():
    if st.button("Compute Global SHAP Importance", width="stretch"):
        with st.spinner("Computing SHAP values..."):
            try:
                explainer = XAIExplainer()
                explainer.fit(
                    pipeline=training_result.pipeline,
                    X_train=training_result.X_train,
                    feature_names=training_result.feature_names,
                    problem_type=profile.problem_type or "binary_classification",
                    label_encoder=training_result.label_encoder,
                )
                global_result = explainer.explain_global(X_sample=training_result.X_train)
                st.session_state["xai_explainer"] = explainer
                st.session_state["xai_global_result"] = global_result
            except Exception as exc:
                st.error(f"SHAP error: {exc}")
                st.session_state["xai_global_result"] = None

xai_global = st.session_state.get("xai_global_result")
if xai_global:
    if xai_global.error:
        st.warning(f"SHAP: {xai_global.error}")
    if xai_global.global_importance:
        st.markdown(f"**Method:** `{xai_global.shap_method}`")
        st.plotly_chart(
            charts.plot_global_shap(xai_global.global_importance_df),
            width="stretch",
        )
        with st.expander("Full Feature Importance Table"):
            st.dataframe(xai_global.global_importance_df, width="stretch")
elif training_result:
    st.info("Click 'Compute Global SHAP Importance' above.")
else:
    st.info("Train a model (Section 9) first.")


# =============================================================================
# SECTION 14: INDIVIDUAL PREDICTION EXPLANATION
# =============================================================================
st.markdown("---")
st.markdown("## 14. Individual Prediction Explanation")

explainer = st.session_state.get("xai_explainer")
if explainer and training_result and training_result.is_successful():
    test_df = training_result.X_test
    n_test = len(test_df)

    obs_idx = st.slider(
        "Select observation from test set:",
        min_value=0,
        max_value=max(0, n_test - 1),
        value=0,
        help="Choose a row from the held-out test set to explain.",
    )

    if st.button("Explain This Prediction", width="stretch"):
        with st.spinner("Computing local SHAP explanation..."):
            try:
                row = test_df.iloc[[obs_idx]]
                local_result = explainer.explain_local(row, observation_index=obs_idx)
                st.session_state["xai_local_result"] = local_result
            except Exception as exc:
                st.error(f"Local explanation failed: {exc}")
                local_result = None

    xai_local = st.session_state.get("xai_local_result")
    if xai_local and xai_local.is_successful():
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", str(xai_local.predicted_class_label or xai_local.prediction))
        col2.metric("Probability", f"{xai_local.prediction_probability:.4f}" if xai_local.prediction_probability else "N/A")
        col3.metric("Base Value", f"{xai_local.base_value:.4f}" if xai_local.base_value else "N/A")
        if xai_local.explained_output_label:
            st.caption(
                f"SHAP explains class/value `{xai_local.explained_output_label}` in "
                f"{xai_local.output_space or 'an unverified output space'}; "
                f"additivity: {'verified' if xai_local.additivity_verified else 'not verified'}"
                + (f" (error {xai_local.additivity_error:.3g})" if xai_local.additivity_error is not None else "")
            )

        if xai_local.local_shap:
            st.plotly_chart(
                charts.plot_local_shap(
                    xai_local.local_shap,
                    xai_local.prediction,
                    xai_local.prediction_probability,
                    obs_idx,
                ),
                width="stretch",
            )

            with st.expander("SHAP Values Table"):
                shap_df = pd.DataFrame({
                    "Feature": list(xai_local.local_shap.keys()),
                    "SHAP Value": list(xai_local.local_shap.values()),
                    "Direction": ["increases" if v > 0 else "decreases" for v in xai_local.local_shap.values()],
                }).sort_values("SHAP Value", key=abs, ascending=False)
                st.dataframe(shap_df, width="stretch")

        if xai_local.local_lime:
            with st.expander("LIME Explanation (supplementary)"):
                lime_df = pd.DataFrame({
                    "Feature": list(xai_local.local_lime.keys()),
                    "LIME Weight": list(xai_local.local_lime.values()),
                }).sort_values("LIME Weight", key=abs, ascending=False)
                st.dataframe(lime_df, width="stretch")

elif training_result:
    st.info("Compute global SHAP (Section 13) first.")
else:
    st.info("Train a model (Section 9) first.")


# =============================================================================
# SECTION 15: RAG-GROUNDED LLM EXPLANATION
# =============================================================================
st.markdown("---")
st.markdown("## 15. RAG-Grounded LLM Explanation")

xai_local = st.session_state.get("xai_local_result")
xai_global = st.session_state.get("xai_global_result")
retriever = st.session_state.get("retriever")

if xai_local and xai_local.is_successful():
    if st.button("Generate Grounded Explanation", width="stretch"):
        with st.spinner("Generating grounded explanation..."):
            try:
                if not st.session_state.get("llm_service"):
                    st.session_state["llm_service"] = create_llm_service()
                llm_service = st.session_state["llm_service"]

                qg = QueryGenerator()
                xai_queries = qg.generate_xai_queries(
                    profile,
                    top_features=list(xai_local.local_shap.keys())[:5] if xai_local.local_shap else [],
                    model_name=model_name,
                )

                grounder = ExplanationGrounder(retriever or KnowledgeRetriever())
                if retriever and retriever.is_built:
                    ctx = grounder.build_context(
                        shap_local=xai_local.local_shap,
                        shap_global=xai_global.global_importance if xai_global else None,
                        lime_local=xai_local.local_lime,
                        observation_index=xai_local.observation_index,
                        prediction=xai_local.prediction,
                        prediction_probability=xai_local.prediction_probability,
                        predicted_class_label=xai_local.predicted_class_label,
                        explained_output_label=xai_local.explained_output_label,
                        explained_output_probability=xai_local.explained_output_probability,
                        output_space=xai_local.output_space,
                        additivity_error=xai_local.additivity_error,
                        additivity_verified=xai_local.additivity_verified,
                        base_value=xai_local.base_value,
                        model_name=model_name,
                        problem_type=profile.problem_type or "",
                        target_column=target_column,
                        feature_names=training_result.feature_names if training_result else [],
                        xai_queries=xai_queries,
                    )
                else:
                    from src.rag.explanation import GroundedExplanationContext
                    ctx = GroundedExplanationContext(
                        observation_index=xai_local.observation_index,
                        prediction=xai_local.prediction,
                        prediction_probability=xai_local.prediction_probability,
                        predicted_class_label=xai_local.predicted_class_label,
                        explained_output_label=xai_local.explained_output_label,
                        explained_output_probability=xai_local.explained_output_probability,
                        output_space=xai_local.output_space,
                        additivity_error=xai_local.additivity_error,
                        additivity_verified=xai_local.additivity_verified,
                        base_value=xai_local.base_value,
                        shap_values=xai_local.local_shap or {},
                        lime_values=xai_local.local_lime or {},
                        global_importance=xai_global.global_importance if xai_global else {},
                        model_name=model_name,
                        problem_type=profile.problem_type or "",
                        target_column=target_column,
                        feature_names=training_result.feature_names if training_result else [],
                        xai_method=xai_local.shap_method,
                    )

                st.session_state["grounded_context"] = ctx
                explanation = llm_service.explain_local(ctx)
                st.session_state["llm_explanation"] = explanation

            except Exception as exc:
                st.error(f"Explanation generation failed: {exc}")
                traceback.print_exc()

    llm_explanation = st.session_state.get("llm_explanation")
    if llm_explanation:
        llm_service = st.session_state.get("llm_service")
        provider_name = llm_service.provider_name if llm_service else "Unknown"
        st.markdown(f"**Provider:** `{provider_name}`")

        ctx = st.session_state.get("grounded_context")
        if ctx and ctx.retrieved_docs:
            docs = ctx.retrieved_docs[0] if isinstance(ctx.retrieved_docs, tuple) else ctx.retrieved_docs
            if docs:
                with st.expander(f"Retrieved Knowledge Used ({len(docs)} docs)"):
                    for doc in docs:
                        title = getattr(doc, "title", str(doc))
                        score = getattr(doc, "relevance_score", 0.0)
                        snippet = getattr(doc, "snippet", "")
                        st.markdown(f"**{title}** (score: {score:.4f})")
                        if snippet:
                            st.markdown(snippet[:300])
                        st.markdown("---")

        st.markdown("### Grounded Explanation")
        st.markdown(llm_explanation)

        if ctx and ctx.shap_values:
            with st.expander("Verify: SHAP Evidence Passed to LLM"):
                st.markdown("The following actual SHAP values were provided to the explanation layer:")
                ev_df = pd.DataFrame({
                    "Feature": list(ctx.shap_values.keys()),
                    "Actual SHAP Value": list(ctx.shap_values.values()),
                }).sort_values("Actual SHAP Value", key=abs, ascending=False)
                st.dataframe(ev_df, width="stretch")
                st.caption("These values were computed from the actual fitted model, not generated by the LLM.")
else:
    st.info("Generate a local explanation (Section 14) first.")


# =============================================================================
# SECTION 16: DOWNLOAD REPORT
# =============================================================================
st.markdown("---")
st.markdown("## 16. Download Report")

can_generate_report = training_result and training_result.is_successful()

if can_generate_report:
    if st.button("Generate HTML Report", width="stretch"):
        with st.spinner("Generating report..."):
            report_data = {
                "dataset_name": st.session_state.get("dataset_name", "dataset"),
                "profile": profile,
                "query_gen_result": st.session_state.get("query_gen_result"),
                "retrieved_docs": retrieved_docs,
                "recommendations": recommendations,
                "training_result": training_result,
                "comparison_result": st.session_state.get("comparison_result"),
                "xai_result": xai_global,
                "llm_explanation": st.session_state.get("llm_explanation"),
                "llm_provider": st.session_state.get("llm_service", {}) and st.session_state["llm_service"].provider_name,
            }
            rg = ReportGenerator()
            html_content = rg.generate(report_data)
            st.session_state["report_html"] = html_content

    report_html = st.session_state.get("report_html")
    if report_html:
        import base64
        filename = f"ai_ds_report_{st.session_state.get('dataset_name', 'report').replace('.csv', '')}.html"
        st.download_button(
            label="Download HTML Report",
            data=report_html,
            file_name=filename,
            mime="text/html",
            width="stretch",
        )
        st.success("Report ready for download.")
else:
    st.info("Complete model training (Section 9) to generate a report.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("AI Data Science Assistant · RAG + SHAP + Streamlit · All results from actual execution")
