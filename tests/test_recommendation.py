"""tests/test_recommendation.py — Tests for the recommendation engine."""

import pytest
from src.data.profiler import DatasetProfiler
from src.rag.retriever import KnowledgeRetriever
from src.rag.recommendation import RecommendationEngine, Recommendation, SearchSpaceConfig
from src.rag.query_generator import QueryGenerator, QueryGenerationResult


class TestQueryGenerator:
    def test_generates_queries_binary(self, binary_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        qg = QueryGenerator()
        result = qg.generate_queries(profile, model_name="Random Forest", use_llm=False)
        assert isinstance(result, QueryGenerationResult)
        assert result.total_queries > 0
        assert all(isinstance(q, str) for q in result.queries)
        assert result.strategy == "deterministic"

    def test_generates_queries_imbalanced(self, imbalanced_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(imbalanced_classification_df, target_column="target")
        qg = QueryGenerator()
        result = qg.generate_queries(profile, use_llm=False)
        assert any("imbalance" in q.lower() or "balanced" in q.lower() for q in result.queries)

    def test_generates_queries_missing(self, df_with_missing):
        profiler = DatasetProfiler()
        profile = profiler.profile(df_with_missing, target_column="target")
        qg = QueryGenerator()
        result = qg.generate_queries(profile, use_llm=False)
        assert any("missing" in q.lower() or "imputation" in q.lower() for q in result.queries)

    def test_generates_queries_regression(self, regression_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(regression_df, target_column="target")
        qg = QueryGenerator()
        result = qg.generate_queries(profile, use_llm=False)
        assert any("regression" in q.lower() or "rmse" in q.lower() for q in result.queries)

    def test_deterministic_strategy_no_llm(self, binary_classification_df):
        """When use_llm=False, strategy must be 'deterministic'."""
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        qg = QueryGenerator()
        result = qg.generate_queries(profile, use_llm=False)
        assert result.strategy == "deterministic"
        assert result.fallback_occurred is False
        assert result.total_queries > 0

    def test_auto_strategy_uses_deterministic_when_no_candidate(self, binary_classification_df, monkeypatch):
        """Automatic mode must remain useful without an installed/configured LLM."""
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        monkeypatch.setattr(
            QueryGenerator,
            "_should_attempt_llm",
            staticmethod(lambda _requested: (False, {"reason": "No configured LLM provider is available; using deterministic queries."})),
        )
        result = QueryGenerator().generate_queries(profile)
        assert result.strategy == "deterministic"
        assert "No configured LLM" in result.selection_reason

    def test_invalid_llm_output_falls_back_to_deterministic(self, binary_classification_df, monkeypatch):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        monkeypatch.setattr(QueryGenerator, "_should_attempt_llm", staticmethod(lambda _requested: (True, {})))
        monkeypatch.setattr(
            QueryGenerator,
            "_try_llm_enhancement",
            lambda _self, _profile, _model: ([], {"error": "LLM returned no valid queries"}),
        )
        result = QueryGenerator().generate_queries(profile)
        assert result.strategy == "llm_fallback"
        assert result.fallback_occurred
        assert result.queries

    def test_successful_llm_queries_are_kept_separate_and_used(self, binary_classification_df, monkeypatch):
        profile = DatasetProfiler().profile(binary_classification_df, target_column="target")
        monkeypatch.setattr(QueryGenerator, "_should_attempt_llm", staticmethod(lambda _requested: (True, {})))
        monkeypatch.setattr(
            QueryGenerator,
            "_try_llm_enhancement",
            lambda _self, _profile, _model: (
                ["distinctive valid retrieval query about calibration metrics"],
                {"provider": "Test provider", "model": "test-model", "diagnostics": {"success": True, "parse_success": True}},
            ),
        )
        result = QueryGenerator().generate_queries(profile, use_llm=True)
        assert result.strategy == "llm_enhanced"
        assert result.llm_queries == ["distinctive valid retrieval query about calibration metrics"]
        assert result.llm_queries[0] in result.queries
        assert result.llm_diagnostics["parse_success"]

    def test_queries_are_deduplicated(self, binary_classification_df):
        """All queries should be unique."""
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        qg = QueryGenerator()
        result = qg.generate_queries(profile, use_llm=False)
        assert len(result.queries) == len(set(result.queries))

    def test_xai_queries_are_list_of_strings(self, binary_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        qg = QueryGenerator()
        xai_q = qg.generate_xai_queries(profile, top_features=["age", "income"])
        assert isinstance(xai_q, list)
        assert len(xai_q) > 0
        assert all(isinstance(q, str) for q in xai_q)


class TestRecommendationEngine:
    def test_recommendations_for_imbalanced(self, imbalanced_classification_df, knowledge_base_dir):
        profiler = DatasetProfiler()
        profile = profiler.profile(imbalanced_classification_df, target_column="target")
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        engine = RecommendationEngine(retriever)
        qg = QueryGenerator()
        qg_result = qg.generate_queries(profile, use_llm=False)
        recs, docs = engine.generate_recommendations(profile, qg_result.queries)
        assert len(recs) > 0
        # Should recommend class_weight for imbalanced data
        techniques = [r.technique for r in recs]
        assert any("class_weight" in t.lower() or "balanced" in t.lower() for t in techniques)

    def test_recommendations_for_missing_values(self, df_with_missing, knowledge_base_dir):
        profiler = DatasetProfiler()
        profile = profiler.profile(df_with_missing, target_column="target")
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        engine = RecommendationEngine(retriever)
        qg = QueryGenerator()
        qg_result = qg.generate_queries(profile, use_llm=False)
        recs, docs = engine.generate_recommendations(profile, qg_result.queries)
        assert len(recs) > 0
        techniques = [r.technique.lower() for r in recs]
        assert any("imputation" in t for t in techniques)

    def test_recommendation_has_required_fields(self, binary_classification_df, knowledge_base_dir):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        engine = RecommendationEngine(retriever)
        qg = QueryGenerator()
        qg_result = qg.generate_queries(profile, use_llm=False)
        recs, _ = engine.generate_recommendations(profile, qg_result.queries)
        for rec in recs:
            assert isinstance(rec, Recommendation)
            assert rec.technique
            assert rec.trigger
            assert rec.reason

    def test_search_space_generation_rag(self, imbalanced_classification_df, knowledge_base_dir):
        profiler = DatasetProfiler()
        profile = profiler.profile(imbalanced_classification_df, target_column="target")
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        engine = RecommendationEngine(retriever)
        qg = QueryGenerator()
        qg_result = qg.generate_queries(profile, model_name="Random Forest", use_llm=False)
        recs, docs = engine.generate_recommendations(profile, qg_result.queries)
        config = engine.generate_search_space(
            profile, "Random Forest", recs, docs, is_rag_guided=True
        )
        assert isinstance(config, SearchSpaceConfig)
        # RAG should add class_weight for imbalanced
        has_cw = any("class_weight" in k for k in config.param_grid)
        assert has_cw, f"Expected class_weight in {config.param_grid}"

    def test_consumed_preprocessing_recommendations_are_recorded(self, df_with_missing, knowledge_base_dir):
        profile = DatasetProfiler().profile(df_with_missing, target_column="target")
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        engine = RecommendationEngine(retriever)
        recs, docs = engine.generate_recommendations(profile, QueryGenerator().generate_queries(profile, use_llm=False).queries)
        config = engine.generate_search_space(profile, "Logistic Regression", recs, docs, is_rag_guided=True)
        assert config.preprocessing_config["numerical_imputer_strategy"] in {"mean", "median"}
        assert config.application_notes

    def test_recommendation_marks_retrieved_evidence(self, df_with_missing, knowledge_base_dir):
        profile = DatasetProfiler().profile(df_with_missing, target_column="target")
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        docs = retriever.retrieve_multiple_queries(["missing values imputation"], global_top_k=3)
        recs, returned_docs = RecommendationEngine(retriever).generate_recommendations(
            profile, ["missing values imputation"], retrieved_docs=docs
        )
        assert returned_docs is docs
        missing_recs = [rec for rec in recs if rec.supporting_doc == "missing_values"]
        assert missing_recs
        assert all(rec.evidence_status in {"retrieved", "rule-based"} for rec in missing_recs)

    def test_search_space_generation_baseline(self, binary_classification_df, knowledge_base_dir):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        engine = RecommendationEngine(retriever)
        config = engine.generate_search_space(
            profile, "Logistic Regression", [], [], is_rag_guided=False
        )
        # Baseline should NOT have class_weight by default
        has_cw = any("class_weight" in k for k in config.param_grid)
        assert not has_cw

    def test_different_datasets_different_spaces(
        self, imbalanced_classification_df, binary_classification_df, knowledge_base_dir
    ):
        profiler = DatasetProfiler()
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        engine = RecommendationEngine(retriever)
        qg = QueryGenerator()

        profile_imb = profiler.profile(imbalanced_classification_df, target_column="target")
        qg_result_imb = qg.generate_queries(profile_imb, model_name="Logistic Regression", use_llm=False)
        recs_imb, docs_imb = engine.generate_recommendations(profile_imb, qg_result_imb.queries)
        config_imb = engine.generate_search_space(
            profile_imb, "Logistic Regression", recs_imb, docs_imb, is_rag_guided=True
        )

        profile_bal = profiler.profile(binary_classification_df, target_column="target")
        qg_result_bal = qg.generate_queries(profile_bal, model_name="Logistic Regression", use_llm=False)
        recs_bal, docs_bal = engine.generate_recommendations(profile_bal, qg_result_bal.queries)
        config_bal = engine.generate_search_space(
            profile_bal, "Logistic Regression", recs_bal, docs_bal, is_rag_guided=True
        )

        # Imbalanced dataset should have class_weight candidate
        assert any("class_weight" in k for k in config_imb.param_grid), \
            "Expected class_weight in imbalanced config"

    def test_search_space_is_rag_guided_flag(self, binary_classification_df, knowledge_base_dir):
        """SearchSpaceConfig.is_rag_guided must reflect the parameter passed."""
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        engine = RecommendationEngine(retriever)
        config_rag = engine.generate_search_space(profile, "Logistic Regression", [], [], is_rag_guided=True)
        config_base = engine.generate_search_space(profile, "Logistic Regression", [], [], is_rag_guided=False)
        assert config_rag.is_rag_guided is True
        assert config_base.is_rag_guided is False
