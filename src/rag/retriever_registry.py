"""
src/rag/retriever_registry.py
──────────────────────────────
Common interface for selecting and managing the active retrieval strategy.

The three strategies are:
  - "tfidf"   : TF-IDF cosine similarity (KnowledgeRetriever)
  - "dense"   : Ollama embedding-based cosine similarity (OllamaDenseRetriever)
  - "hybrid"  : Reciprocal Rank Fusion of TF-IDF + dense (HybridRetriever)

Changing the retriever strategy does NOT affect:
  - The LLM model or provider
  - The knowledge base content
  - The dataset, target column, or train/test split
  - Any training settings

This module is responsible only for building and holding retriever instances.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import (
    EMBEDDING_CACHE_DIR,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_DIR,
    MAX_RETRIEVAL_DOCS,
    MIN_RELEVANCE_SCORE,
    SUPPORTED_EMBEDDING_MODELS,
)
from src.rag.retriever import KnowledgeRetriever, RetrievedDocument, RetrievalTrace

logger = logging.getLogger(__name__)

STRATEGY_TFIDF = "tfidf"
STRATEGY_DENSE = "dense"
STRATEGY_HYBRID = "hybrid"

ALL_STRATEGIES = [STRATEGY_TFIDF, STRATEGY_DENSE, STRATEGY_HYBRID]


@dataclass
class RetrieverStatus:
    """Reported status of the active retriever configuration."""
    strategy: str
    is_built: bool
    n_documents: int
    embedding_model: Optional[str] = None
    build_time_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: Optional[float] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class RetrieverRegistry:
    """
    Holds and builds retriever instances for TF-IDF, dense, and hybrid strategies.

    Usage
    -----
    registry = RetrieverRegistry()
    registry.build(strategy="tfidf")
    docs = registry.retrieve("class imbalance", top_k=5)
    status = registry.status()
    """

    def __init__(
        self,
        kb_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.kb_dir = Path(kb_dir) if kb_dir else KNOWLEDGE_BASE_DIR
        self.cache_dir = Path(cache_dir) if cache_dir else EMBEDDING_CACHE_DIR

        self._strategy: str = STRATEGY_TFIDF
        self._embedding_model: str = EMBEDDING_MODEL

        self._tfidf: Optional[KnowledgeRetriever] = None
        self._dense = None   # OllamaDenseRetriever (optional import)
        self._hybrid = None  # HybridRetriever (optional import)

        self._last_status: Optional[RetrieverStatus] = None
        self._build_info: Dict[str, Any] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(
        self,
        strategy: str = STRATEGY_TFIDF,
        embedding_model: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> RetrieverStatus:
        """
        Build the requested retrieval strategy.

        Parameters
        ----------
        strategy : "tfidf" | "dense" | "hybrid"
        embedding_model : embedding model name (dense/hybrid only)
        force_rebuild : force re-index even if already built

        Returns
        -------
        RetrieverStatus describing what was built and any errors
        """
        if strategy not in ALL_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {ALL_STRATEGIES}.")

        self._strategy = strategy
        if embedding_model:
            self._embedding_model = embedding_model

        started = time.perf_counter()
        status = RetrieverStatus(strategy=strategy, is_built=False, n_documents=0)

        try:
            if strategy == STRATEGY_TFIDF:
                status = self._build_tfidf(force_rebuild, started)
            elif strategy == STRATEGY_DENSE:
                status = self._build_dense(force_rebuild, started)
            elif strategy == STRATEGY_HYBRID:
                status = self._build_hybrid(force_rebuild, started)
        except Exception as exc:
            logger.error("RetrieverRegistry.build failed: %s", exc, exc_info=True)
            status.error = str(exc)
            status.is_built = False

        self._last_status = status
        return status

    def retrieve(
        self,
        query: str,
        top_k: int = MAX_RETRIEVAL_DOCS,
        min_score: float = MIN_RELEVANCE_SCORE,
    ) -> List[RetrievedDocument]:
        """Retrieve documents using the active strategy."""
        retriever = self._active_retriever()
        if retriever is None:
            logger.warning("No retriever built. Call build() first.")
            return []
        if self._strategy == STRATEGY_HYBRID:
            docs, _ = retriever.retrieve(query, top_k=top_k)
            return docs
        return retriever.retrieve(query, top_k=top_k, min_score=min_score)

    def retrieve_multiple_queries(
        self,
        queries: List[str],
        top_k_per_query: int = 3,
        global_top_k: int = MAX_RETRIEVAL_DOCS,
        min_score: float = MIN_RELEVANCE_SCORE,
    ) -> List[RetrievedDocument]:
        """Retrieve documents for multiple queries, deduplicated."""
        docs, _ = self.retrieve_multiple_queries_with_provenance(
            queries, top_k_per_query=top_k_per_query,
            global_top_k=global_top_k, min_score=min_score,
        )
        return docs

    def retrieve_multiple_queries_with_provenance(
        self,
        queries: List[str],
        top_k_per_query: int = 3,
        global_top_k: int = MAX_RETRIEVAL_DOCS,
        min_score: float = MIN_RELEVANCE_SCORE,
    ) -> Tuple[List[RetrievedDocument], List[Any]]:
        """
        Retrieve documents for multiple queries with provenance info.

        Returns
        -------
        (docs, trace) — trace is a list of RetrievalTrace for TF-IDF,
        or HybridRetrievalDiagnostics for hybrid, or [] for dense.
        """
        if self._strategy == STRATEGY_TFIDF and self._tfidf:
            return self._tfidf.retrieve_multiple_queries_with_provenance(
                queries,
                top_k_per_query=top_k_per_query,
                global_top_k=global_top_k,
                min_score=min_score,
            )
        elif self._strategy == STRATEGY_DENSE and self._dense:
            docs = self._dense.retrieve_multiple_queries(
                queries,
                top_k_per_query=top_k_per_query,
                global_top_k=global_top_k,
                min_score=min_score,
            )
            return docs, []
        elif self._strategy == STRATEGY_HYBRID and self._hybrid:
            return self._hybrid.retrieve_multiple_queries(
                queries,
                top_k_per_query=top_k_per_query,
                global_top_k=global_top_k,
            )
        return [], []

    def status(self) -> Optional[RetrieverStatus]:
        """Return the most recent build status."""
        return self._last_status

    @property
    def is_built(self) -> bool:
        retriever = self._active_retriever()
        if retriever is None:
            return False
        return getattr(retriever, "is_built", False)

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    @property
    def n_documents(self) -> int:
        retriever = self._active_retriever()
        return getattr(retriever, "n_documents", 0) if retriever else 0

    # ── Private build methods ─────────────────────────────────────────────────

    def _build_tfidf(self, force_rebuild: bool, started: float) -> RetrieverStatus:
        if self._tfidf is None or force_rebuild:
            self._tfidf = KnowledgeRetriever(kb_dir=self.kb_dir)
            self._tfidf.build_index()
        status = RetrieverStatus(
            strategy=STRATEGY_TFIDF,
            is_built=self._tfidf.is_built,
            n_documents=self._tfidf.n_documents,
            build_time_seconds=round(time.perf_counter() - started, 3),
        )
        if not self._tfidf.is_built:
            status.error = "TF-IDF index build failed. Check knowledge base directory."
        return status

    def _build_dense(self, force_rebuild: bool, started: float) -> RetrieverStatus:
        try:
            from src.rag.dense_retriever import (
                OllamaDenseRetriever,
                OllamaUnavailableError,
                ModelNotInstalledError,
            )
        except ImportError as exc:
            return RetrieverStatus(
                strategy=STRATEGY_DENSE,
                is_built=False,
                n_documents=0,
                error=f"Dense retriever not available: {exc}",
            )

        need_rebuild = (
            force_rebuild
            or self._dense is None
            or getattr(self._dense, "embedding_model", None) != self._embedding_model
            or not getattr(self._dense, "is_built", False)
        )
        if need_rebuild:
            self._dense = OllamaDenseRetriever(
                kb_dir=self.kb_dir,
                embedding_model=self._embedding_model,
                cache_dir=self.cache_dir,
            )
            build_info = self._dense.build_index()
        else:
            build_info = {"success": True, "cache_hits": 0, "cache_misses": 0}

        status = RetrieverStatus(
            strategy=STRATEGY_DENSE,
            is_built=self._dense.is_built,
            n_documents=self._dense.n_documents,
            embedding_model=self._embedding_model,
            build_time_seconds=round(time.perf_counter() - started, 3),
            cache_hits=build_info.get("cache_hits", 0),
            cache_misses=build_info.get("cache_misses", 0),
        )
        if status.cache_hits + status.cache_misses > 0:
            total = status.cache_hits + status.cache_misses
            status.cache_hit_rate = round(status.cache_hits / total, 3)

        if not self._dense.is_built:
            status.error = build_info.get(
                "error", "Dense index build failed. Ollama may be unavailable or model not installed."
            )
        return status

    def _build_hybrid(self, force_rebuild: bool, started: float) -> RetrieverStatus:
        # Always ensure TF-IDF is built first
        if self._tfidf is None or not self._tfidf.is_built:
            self._tfidf = KnowledgeRetriever(kb_dir=self.kb_dir)
            self._tfidf.build_index()

        # Then ensure dense is built
        dense_status = self._build_dense(force_rebuild, started)

        if not self._tfidf.is_built or not dense_status.is_built:
            return RetrieverStatus(
                strategy=STRATEGY_HYBRID,
                is_built=False,
                n_documents=0,
                embedding_model=self._embedding_model,
                error=dense_status.error or "Could not build hybrid: TF-IDF or dense index failed.",
                build_time_seconds=round(time.perf_counter() - started, 3),
                cache_hits=dense_status.cache_hits,
                cache_misses=dense_status.cache_misses,
                cache_hit_rate=dense_status.cache_hit_rate,
            )

        try:
            from src.rag.hybrid_retriever import HybridRetriever
            self._hybrid = HybridRetriever(
                tfidf_retriever=self._tfidf,
                dense_retriever=self._dense,
            )
        except ImportError as exc:
            return RetrieverStatus(
                strategy=STRATEGY_HYBRID,
                is_built=False,
                n_documents=0,
                error=f"Hybrid retriever not available: {exc}",
            )

        return RetrieverStatus(
            strategy=STRATEGY_HYBRID,
            is_built=self._hybrid.is_built,
            n_documents=self._tfidf.n_documents,
            embedding_model=self._embedding_model,
            build_time_seconds=round(time.perf_counter() - started, 3),
            cache_hits=dense_status.cache_hits,
            cache_misses=dense_status.cache_misses,
            cache_hit_rate=dense_status.cache_hit_rate,
        )

    def _active_retriever(self):
        """Return the currently active retriever instance."""
        if self._strategy == STRATEGY_TFIDF:
            return self._tfidf
        elif self._strategy == STRATEGY_DENSE:
            return self._dense
        elif self._strategy == STRATEGY_HYBRID:
            return self._hybrid
        return None

