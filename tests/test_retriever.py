"""tests/test_retriever.py — Tests for TF-IDF RAG retrieval system."""

import pytest
from pathlib import Path

from src.rag.retriever import KnowledgeRetriever, RetrievedDocument


class TestKnowledgeRetriever:
    def test_build_index(self, knowledge_base_dir):
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        assert retriever.is_built
        assert retriever.n_documents == 3

    def test_retrieve_returns_results(self, knowledge_base_dir):
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        docs = retriever.retrieve("class imbalance missing values")
        assert len(docs) > 0
        assert all(isinstance(d, RetrievedDocument) for d in docs)

    def test_retrieve_sorted_by_score(self, knowledge_base_dir):
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        docs = retriever.retrieve("random forest hyperparameters")
        scores = [d.relevance_score for d in docs]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_empty_query(self, knowledge_base_dir):
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        docs = retriever.retrieve("")
        assert docs == []

    def test_retrieve_not_built(self):
        retriever = KnowledgeRetriever()
        docs = retriever.retrieve("test query")
        assert docs == []

    def test_min_score_filter(self, knowledge_base_dir):
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        # Use a very high threshold — should return nothing
        docs = retriever.retrieve("random forest", min_score=0.99)
        assert docs == []

    def test_top_k_respected(self, knowledge_base_dir):
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        docs = retriever.retrieve("model", top_k=1)
        assert len(docs) <= 1

    def test_retrieve_multiple_queries(self, knowledge_base_dir):
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        queries = ["missing values imputation", "class imbalance balanced accuracy"]
        docs = retriever.retrieve_multiple_queries(queries, global_top_k=3)
        assert len(docs) <= 3
        # All docs should be unique
        doc_ids = [d.doc_id for d in docs]
        assert len(doc_ids) == len(set(doc_ids))

    def test_retrieval_provenance_links_documents_to_queries(self, knowledge_base_dir):
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        docs, trace = retriever.retrieve_multiple_queries_with_provenance(
            ["missing values imputation", "class imbalance balanced accuracy"], global_top_k=3
        )
        retained = {doc.doc_id for doc in docs}
        assert trace
        assert all(item.doc_id in retained for item in trace)
        assert all(0 <= item.relevance_score <= 1 for item in trace)

    def test_empty_kb_dir(self, tmp_path):
        empty_kb = tmp_path / "empty_kb"
        empty_kb.mkdir()
        retriever = KnowledgeRetriever(kb_dir=empty_kb)
        retriever.build_index()
        assert not retriever.is_built

    def test_nonexistent_kb_dir(self, tmp_path):
        fake_dir = tmp_path / "nonexistent"
        retriever = KnowledgeRetriever(kb_dir=fake_dir)
        retriever.build_index()
        assert not retriever.is_built

    def test_document_has_all_fields(self, knowledge_base_dir):
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        docs = retriever.retrieve("missing values")
        assert len(docs) > 0
        doc = docs[0]
        assert doc.doc_id
        assert doc.title
        assert doc.content
        assert doc.source_file
        assert isinstance(doc.relevance_score, float)
        assert doc.snippet

    def test_relevance_scores_between_0_and_1(self, knowledge_base_dir):
        retriever = KnowledgeRetriever(kb_dir=knowledge_base_dir)
        retriever.build_index()
        docs = retriever.retrieve("balanced accuracy class weight")
        for doc in docs:
            assert 0.0 <= doc.relevance_score <= 1.0
