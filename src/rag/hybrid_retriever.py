"""
src/rag/hybrid_retriever.py
────────────────────────────
Hybrid retrieval combining TF-IDF and dense embedding results
using Reciprocal Rank Fusion (RRF).

RRF score for document d = sum_over_retrievers( 1 / (k + rank(d)) )
where k is the RRF constant (default 60) and rank starts at 1.

Documents are deduplicated by doc_id before fusion.
Results include per-document source IDs, ranks, and component scores
for full diagnostic transparency.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.settings import (
    HYBRID_DENSE_CANDIDATES,
    HYBRID_TFIDF_CANDIDATES,
    MAX_RETRIEVAL_DOCS,
    MIN_RELEVANCE_SCORE,
    RRF_K,
)
from src.rag.retriever import KnowledgeRetriever, RetrievedDocument
from src.rag.dense_retriever import OllamaDenseRetriever

logger = logging.getLogger(__name__)


@dataclass
class HybridRetrievalDiagnostics:
    """Per-document diagnostics for hybrid RRF retrieval."""

    doc_id: str
    title: str
    rrf_score: float
    tfidf_rank: Optional[int]      # 1-based rank in TF-IDF results, None if not retrieved
    dense_rank: Optional[int]      # 1-based rank in dense results, None if not retrieved
    tfidf_score: Optional[float]
    dense_score: Optional[float]
    sources: List[str] = field(default_factory=list)  # ["tfidf"] | ["dense"] | both


class HybridRetriever:
    """
    Hybrid retriever that fuses TF-IDF and dense embedding results via
    Reciprocal Rank Fusion (RRF).

    RRF is a rank-based fusion method that does **not** mix raw component
    scores.  Each retriever contributes :math:`1 / (k + rank)` to the fused
    score, where *rank* is 1-based and *k* is the RRF constant (default 60).
    Raw component scores are preserved in :class:`HybridRetrievalDiagnostics`
    for inspection but are never blended arithmetically.

    Parameters
    ----------
    tfidf_retriever : KnowledgeRetriever
        A fully built TF-IDF retriever.
    dense_retriever : OllamaDenseRetriever
        A fully built dense retriever.  If not built (Ollama down), the hybrid
        falls back to TF-IDF-only results with a warning.
    rrf_k : int
        RRF smoothing constant.
    tfidf_candidates : int
        Number of candidates to pull from the TF-IDF retriever before fusion.
    dense_candidates : int
        Number of candidates to pull from the dense retriever before fusion.
    """

    def __init__(
        self,
        tfidf_retriever: KnowledgeRetriever,
        dense_retriever: OllamaDenseRetriever,
        rrf_k: int = RRF_K,
        tfidf_candidates: int = HYBRID_TFIDF_CANDIDATES,
        dense_candidates: int = HYBRID_DENSE_CANDIDATES,
    ) -> None:
        self._tfidf = tfidf_retriever
        self._dense = dense_retriever
        self._rrf_k = rrf_k
        self._tfidf_candidates = tfidf_candidates
        self._dense_candidates = dense_candidates

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = MAX_RETRIEVAL_DOCS,
        min_score: float = 0.0,
    ) -> Tuple[List[RetrievedDocument], List[HybridRetrievalDiagnostics]]:
        """
        Retrieve documents for *query* using Reciprocal Rank Fusion of TF-IDF
        and dense results.

        Algorithm
        ---------
        1. Collect up to ``tfidf_candidates`` documents from the TF-IDF
           retriever (with ``min_score=0`` so we always get candidates).
        2. Collect up to ``dense_candidates`` documents from the dense
           retriever (with ``min_score=0``).
        3. For each retriever's ranked list add ``1 / (k + rank)`` to that
           document's cumulative RRF score.
        4. Sort by RRF score descending, apply ``min_score`` filter, return
           top ``top_k``.

        Parameters
        ----------
        query : str
            Natural-language query.
        top_k : int
            Maximum number of documents to return.
        min_score : float
            Minimum RRF score filter (applied after fusion).

        Returns
        -------
        Tuple[List[RetrievedDocument], List[HybridRetrievalDiagnostics]]
            Fused documents and their per-document diagnostics.
        """
        if not query.strip():
            logger.warning("Empty query supplied to HybridRetriever.")
            return [], []

        # ── Step 1: Collect TF-IDF candidates ────────────────────────────────
        tfidf_docs: List[RetrievedDocument] = []
        if self._tfidf.is_built:
            tfidf_docs = self._tfidf.retrieve(
                query, top_k=self._tfidf_candidates, min_score=0.0
            )
        else:
            logger.warning("TF-IDF retriever is not built; skipping TF-IDF candidates.")

        # ── Step 2: Collect dense candidates ─────────────────────────────────
        dense_docs: List[RetrievedDocument] = []
        if self._dense.is_built:
            dense_docs = self._dense.retrieve(
                query, top_k=self._dense_candidates, min_score=0.0
            )
        else:
            logger.warning(
                "Dense retriever is not built (Ollama may be unavailable); "
                "falling back to TF-IDF-only fusion."
            )

        # ── Step 3: Build RRF score map ───────────────────────────────────────
        # rrf_scores[doc_id] -> cumulative RRF score
        rrf_scores: Dict[str, float] = {}
        # tfidf_rank_map[doc_id] -> (1-based rank, raw score)
        tfidf_rank_map: Dict[str, Tuple[int, float]] = {}
        dense_rank_map: Dict[str, Tuple[int, float]] = {}
        # Keep a representative RetrievedDocument per doc_id
        doc_store: Dict[str, RetrievedDocument] = {}

        for rank, doc in enumerate(tfidf_docs, start=1):
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0.0) + 1.0 / (self._rrf_k + rank)
            tfidf_rank_map[doc.doc_id] = (rank, doc.relevance_score)
            doc_store[doc.doc_id] = doc

        for rank, doc in enumerate(dense_docs, start=1):
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0.0) + 1.0 / (self._rrf_k + rank)
            dense_rank_map[doc.doc_id] = (rank, doc.relevance_score)
            if doc.doc_id not in doc_store:
                doc_store[doc.doc_id] = doc

        # ── Step 4: Sort by RRF score and apply filter ───────────────────────
        sorted_doc_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)

        results: List[RetrievedDocument] = []
        diagnostics: List[HybridRetrievalDiagnostics] = []

        for doc_id in sorted_doc_ids:
            if len(results) >= top_k:
                break
            rrf_score = round(rrf_scores[doc_id], 6)
            if rrf_score < min_score:
                continue

            ref_doc = doc_store[doc_id]

            tfidf_rank_val = tfidf_rank_map.get(doc_id)
            dense_rank_val = dense_rank_map.get(doc_id)

            sources: List[str] = []
            if tfidf_rank_val is not None:
                sources.append("tfidf")
            if dense_rank_val is not None:
                sources.append("dense")

            diag = HybridRetrievalDiagnostics(
                doc_id=doc_id,
                title=ref_doc.title,
                rrf_score=rrf_score,
                tfidf_rank=tfidf_rank_val[0] if tfidf_rank_val else None,
                dense_rank=dense_rank_val[0] if dense_rank_val else None,
                tfidf_score=tfidf_rank_val[1] if tfidf_rank_val else None,
                dense_score=dense_rank_val[1] if dense_rank_val else None,
                sources=sources,
            )
            diagnostics.append(diag)

            results.append(
                RetrievedDocument(
                    doc_id=doc_id,
                    title=ref_doc.title,
                    content=ref_doc.content,
                    relevance_score=rrf_score,
                    source_file=ref_doc.source_file,
                    snippet=ref_doc.snippet,
                )
            )

        logger.debug(
            "Hybrid RRF: %d documents for query '%s…' (tfidf=%d, dense=%d)",
            len(results),
            query[:60],
            len(tfidf_docs),
            len(dense_docs),
        )
        return results, diagnostics

    def retrieve_multiple_queries(
        self,
        queries: List[str],
        top_k_per_query: int = 3,
        global_top_k: int = MAX_RETRIEVAL_DOCS,
    ) -> Tuple[List[RetrievedDocument], List[HybridRetrievalDiagnostics]]:
        """
        Retrieve documents for multiple queries, deduplicate by RRF score, and
        return the top-k globally most relevant unique documents.

        When the same document appears for multiple queries, the **highest RRF
        score** across all queries is retained and the diagnostics for that
        winning query are kept.

        Parameters
        ----------
        queries : List[str]
            One or more query strings.
        top_k_per_query : int
            Maximum documents returned per individual query.
        global_top_k : int
            Maximum unique documents to return overall.

        Returns
        -------
        Tuple[List[RetrievedDocument], List[HybridRetrievalDiagnostics]]
        """
        seen_docs: Dict[str, RetrievedDocument] = {}
        seen_diags: Dict[str, HybridRetrievalDiagnostics] = {}

        for query in queries:
            docs, diags = self.retrieve(query, top_k=top_k_per_query, min_score=0.0)
            for doc, diag in zip(docs, diags):
                if (
                    doc.doc_id not in seen_docs
                    or doc.relevance_score > seen_docs[doc.doc_id].relevance_score
                ):
                    seen_docs[doc.doc_id] = doc
                    seen_diags[doc.doc_id] = diag

        sorted_docs = sorted(
            seen_docs.values(), key=lambda d: d.relevance_score, reverse=True
        )[:global_top_k]

        final_diags = [seen_diags[d.doc_id] for d in sorted_docs]
        return sorted_docs, final_diags

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_built(self) -> bool:
        """True only if **both** the TF-IDF and dense retrievers are built."""
        return self._tfidf.is_built and self._dense.is_built

    @property
    def rrf_k(self) -> int:
        """RRF smoothing constant *k*."""
        return self._rrf_k

    @property
    def tfidf_candidates(self) -> int:
        """Number of TF-IDF candidates fetched before fusion."""
        return self._tfidf_candidates

    @property
    def dense_candidates(self) -> int:
        """Number of dense candidates fetched before fusion."""
        return self._dense_candidates
