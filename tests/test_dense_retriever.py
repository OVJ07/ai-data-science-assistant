"""
tests/test_dense_retriever.py
──────────────────────────────
Unit tests for OllamaDenseRetriever.
Ollama responses are fully mocked — no running LLM required.
"""

import hashlib
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_embedding(dim: int = 768) -> list:
    """Return a deterministic non-zero float list usable as a fake embedding."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, dim).tolist()


def _make_response(embedding: list):
    """Build a fake requests.Response for the Ollama /api/embeddings endpoint."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"embedding": embedding}
    resp.raise_for_status.return_value = None
    return resp


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def kb_dir(tmp_path) -> Path:
    """Minimal knowledge base with three docs."""
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "missing_values.md").write_text(
        "# Missing Values\nUse median imputation for missing numerical values.\n"
    )
    (kb / "class_imbalance.md").write_text(
        "# Class Imbalance\nClass imbalance causes models to predict the majority class.\n"
        "Use balanced accuracy as the evaluation metric."
    )
    (kb / "random_forest.md").write_text(
        "# Random Forest\nn_estimators and max_depth are key hyperparameters for random forest.\n"
        "class_weight balanced helps with imbalanced datasets."
    )
    return kb


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def retriever(kb_dir, cache_dir):
    """Return an OllamaDenseRetriever (not yet built)."""
    from src.rag.dense_retriever import OllamaDenseRetriever
    return OllamaDenseRetriever(
        kb_dir=kb_dir,
        embedding_model="nomic-embed-text:latest",
        base_url="http://localhost:11434",
        cache_dir=cache_dir,
    )


# ── Tests: index construction ─────────────────────────────────────────────────

class TestBuildIndex:
    def test_build_index_calls_embed_for_each_doc(self, retriever):
        emb = _fake_embedding()
        with patch("requests.post", return_value=_make_response(emb)) as mock_post:
            info = retriever.build_index()
        # 3 documents → 3 embedding calls
        assert mock_post.call_count == 3
        assert info["success"] is True
        assert info["n_documents"] == 3

    def test_build_index_is_built_after_success(self, retriever):
        emb = _fake_embedding()
        with patch("requests.post", return_value=_make_response(emb)):
            retriever.build_index()
        assert retriever.is_built

    def test_build_index_not_built_when_ollama_down(self, retriever):
        from src.rag.dense_retriever import OllamaUnavailableError
        import requests
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError("refused")):
            info = retriever.build_index()
        assert info["success"] is False
        assert not retriever.is_built
        assert "error" in info

    def test_build_index_not_built_when_model_missing(self, retriever):
        resp = MagicMock()
        resp.status_code = 404
        resp.raise_for_status.side_effect = Exception("404 Not Found")
        resp.json.return_value = {"error": "model 'nomic-embed-text:latest' not found"}
        with patch("requests.post", return_value=resp):
            info = retriever.build_index()
        # Should fail gracefully
        assert info["success"] is False

    def test_build_index_caches_embeddings(self, retriever):
        emb = _fake_embedding()
        with patch("requests.post", return_value=_make_response(emb)):
            retriever.build_index()
        # Rebuild — should use cache, so no new HTTP calls
        with patch("requests.post", return_value=_make_response(emb)) as mock_post:
            info = retriever.build_index()
        assert mock_post.call_count == 0  # all from cache
        assert info["cache_hits"] == 3
        assert info["cache_misses"] == 0

    def test_build_index_empty_kb(self, tmp_path, cache_dir):
        from src.rag.dense_retriever import OllamaDenseRetriever
        empty_kb = tmp_path / "empty_kb"
        empty_kb.mkdir()
        r = OllamaDenseRetriever(kb_dir=empty_kb, cache_dir=cache_dir)
        info = r.build_index()
        assert not r.is_built

    def test_build_index_nonexistent_kb(self, tmp_path, cache_dir):
        from src.rag.dense_retriever import OllamaDenseRetriever
        r = OllamaDenseRetriever(kb_dir=tmp_path / "no_such_dir", cache_dir=cache_dir)
        info = r.build_index()
        assert not r.is_built

    def test_embedding_dim_stored(self, retriever):
        emb = _fake_embedding(dim=768)
        with patch("requests.post", return_value=_make_response(emb)):
            retriever.build_index()
        assert retriever.embedding_dim == 768

    def test_dimension_mismatch_handled(self, retriever):
        """If different documents return different embedding sizes, build should fail gracefully."""
        dims = [768, 512, 768]  # doc 2 has wrong dim
        call_count = [0]
        def side_effect(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return _make_response(_fake_embedding(dims[idx]))
        with patch("requests.post", side_effect=side_effect):
            info = retriever.build_index()
        # Dimension mismatch should be caught — result may succeed or fail
        # but must never crash with an unhandled exception
        assert "success" in info


# ── Tests: retrieval ──────────────────────────────────────────────────────────

class TestRetrieve:
    def test_retrieve_not_built_returns_empty(self, retriever):
        results = retriever.retrieve("class imbalance missing values")
        assert results == []

    def test_retrieve_empty_query_returns_empty(self, retriever):
        emb = _fake_embedding()
        with patch("requests.post", return_value=_make_response(emb)):
            retriever.build_index()
        results = retriever.retrieve("")
        assert results == []

    def test_retrieve_returns_documents(self, retriever):
        emb = _fake_embedding()
        with patch("requests.post", return_value=_make_response(emb)):
            retriever.build_index()
            docs = retriever.retrieve("missing values imputation", top_k=3)
        # With same embedding for all docs + query, scores should be ~1.0
        assert len(docs) >= 1

    def test_retrieve_sorted_by_score(self, retriever):
        # Give each document a slightly different embedding
        rng = np.random.default_rng(0)
        doc_embs = [rng.normal(0, 1, 128).tolist() for _ in range(3)]
        query_emb = rng.normal(0, 1, 128).tolist()
        call_count = [0]
        def side_effect(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx < 3:
                return _make_response(doc_embs[idx])
            return _make_response(query_emb)
        with patch("requests.post", side_effect=side_effect):
            retriever.build_index()
            docs = retriever.retrieve("test query", top_k=3, min_score=0.0)
        scores = [d.relevance_score for d in docs]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_top_k_respected(self, retriever):
        emb = _fake_embedding()
        with patch("requests.post", return_value=_make_response(emb)):
            retriever.build_index()
            docs = retriever.retrieve("class imbalance", top_k=1)
        assert len(docs) <= 1

    def test_retrieve_result_fields(self, retriever):
        emb = _fake_embedding()
        with patch("requests.post", return_value=_make_response(emb)):
            retriever.build_index()
            docs = retriever.retrieve("random forest", top_k=3, min_score=0.0)
        assert len(docs) > 0
        doc = docs[0]
        assert doc.doc_id
        assert doc.title
        assert doc.content
        assert doc.source_file
        assert isinstance(doc.relevance_score, float)
        assert doc.snippet

    def test_retrieve_scores_in_valid_range(self, retriever):
        emb = _fake_embedding()
        with patch("requests.post", return_value=_make_response(emb)):
            retriever.build_index()
            docs = retriever.retrieve("missing values", top_k=3, min_score=0.0)
        for doc in docs:
            assert -1.0 <= doc.relevance_score <= 1.0

    def test_retrieve_ollama_down_at_query_time(self, retriever):
        import requests
        emb = _fake_embedding()
        # Build succeeds
        with patch("requests.post", return_value=_make_response(emb)):
            retriever.build_index()
        # Query fails because Ollama went down
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
            results = retriever.retrieve("class imbalance")
        assert results == []


# ── Tests: multiple queries ───────────────────────────────────────────────────

class TestRetrieveMultipleQueries:
    def test_no_duplicate_doc_ids(self, retriever):
        emb = _fake_embedding()
        with patch("requests.post", return_value=_make_response(emb)):
            retriever.build_index()
            docs = retriever.retrieve_multiple_queries(
                ["missing values", "class imbalance balanced accuracy"],
                global_top_k=3,
                min_score=0.0,
            )
        ids = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids))

    def test_global_top_k_respected(self, retriever):
        emb = _fake_embedding()
        with patch("requests.post", return_value=_make_response(emb)):
            retriever.build_index()
            docs = retriever.retrieve_multiple_queries(
                ["missing values", "class imbalance", "random forest"],
                global_top_k=2,
                min_score=0.0,
            )
        assert len(docs) <= 2


# ── Tests: cache ──────────────────────────────────────────────────────────────

class TestEmbeddingCache:
    def test_cache_miss_on_first_build(self, retriever):
        emb = _fake_embedding()
        with patch("requests.post", return_value=_make_response(emb)):
            info = retriever.build_index()
        assert info["cache_misses"] == 3
        assert info["cache_hits"] == 0

    def test_cache_hit_on_rebuild(self, retriever):
        emb = _fake_embedding()
        with patch("requests.post", return_value=_make_response(emb)):
            retriever.build_index()
        with patch("requests.post", return_value=_make_response(emb)) as mock_post:
            info = retriever.build_index()
        assert info["cache_hits"] == 3
        assert mock_post.call_count == 0

    def test_cache_invalidated_when_model_changes(self, kb_dir, cache_dir):
        from src.rag.dense_retriever import OllamaDenseRetriever
        emb = _fake_embedding()
        r1 = OllamaDenseRetriever(
            kb_dir=kb_dir, embedding_model="nomic-embed-text:latest", cache_dir=cache_dir
        )
        with patch("requests.post", return_value=_make_response(emb)):
            r1.build_index()

        # Different model → different cache namespace → cache misses
        r2 = OllamaDenseRetriever(
            kb_dir=kb_dir, embedding_model="qwen3-embedding:0.6b", cache_dir=cache_dir
        )
        with patch("requests.post", return_value=_make_response(emb)) as mock_post:
            info = r2.build_index()
        assert info["cache_misses"] == 3  # different model = no cache reuse
        assert mock_post.call_count == 3

    def test_cache_stats_property(self, retriever):
        emb = _fake_embedding()
        with patch("requests.post", return_value=_make_response(emb)):
            retriever.build_index()
        stats = retriever.cache_stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


# ── Tests: error handling ─────────────────────────────────────────────────────

class TestErrorHandling:
    def test_malformed_response_does_not_crash(self, retriever):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"not_embedding": [1, 2, 3]}  # malformed
        resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=resp):
            info = retriever.build_index()
        # Should fail gracefully
        assert "success" in info

    def test_empty_text_handled(self, retriever, kb_dir):
        """Empty doc content does not cause a crash during index build."""
        (kb_dir / "empty_doc.md").write_text("   ")  # whitespace only
        emb = _fake_embedding()
        with patch("requests.post", return_value=_make_response(emb)):
            info = retriever.build_index()
        # empty_doc.md should be skipped silently
        assert "success" in info

    def test_ollama_unavailable_error_type(self):
        from src.rag.dense_retriever import OllamaUnavailableError
        assert issubclass(OllamaUnavailableError, RuntimeError)

    def test_model_not_installed_error_type(self):
        from src.rag.dense_retriever import ModelNotInstalledError
        assert issubclass(ModelNotInstalledError, RuntimeError)

