"""
src/rag/retriever.py
─────────────────────
Local TF-IDF retrieval system over Markdown knowledge-base documents.
No external vector databases or paid APIs required.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import (
    KNOWLEDGE_BASE_DIR,
    MAX_RETRIEVAL_DOCS,
    MIN_RELEVANCE_SCORE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A retrieved knowledge-base document with its metadata and score."""
    doc_id: str
    title: str
    content: str
    relevance_score: float
    source_file: str
    snippet: str = ""

    def __post_init__(self):
        if not self.snippet:
            self.snippet = self._extract_snippet(self.content)

    @staticmethod
    def _extract_snippet(content: str, max_chars: int = 300) -> str:
        """Extract a readable snippet from the document content."""
        # Remove markdown headers and excess whitespace
        clean = re.sub(r"^#+\s+", "", content, flags=re.MULTILINE)
        clean = re.sub(r"\n{2,}", "\n", clean).strip()
        if len(clean) <= max_chars:
            return clean
        return clean[:max_chars].rsplit(" ", 1)[0] + "…"


@dataclass(frozen=True)
class RetrievalTrace:
    """One query-to-document match used by multi-query TF-IDF retrieval."""
    query: str
    doc_id: str
    title: str
    relevance_score: float


class KnowledgeRetriever:
    """
    TF-IDF based retriever over a directory of Markdown files.

    Usage
    -----
    retriever = KnowledgeRetriever()
    retriever.build_index()          # or pass custom kb_dir
    docs = retriever.retrieve("class imbalance logistic regression", top_k=3)
    """

    def __init__(self, kb_dir: Optional[Path] = None) -> None:
        self.kb_dir: Path = Path(kb_dir) if kb_dir else KNOWLEDGE_BASE_DIR
        self._documents: Dict[str, str] = {}        # doc_id -> raw text
        self._titles: Dict[str, str] = {}           # doc_id -> title
        self._source_files: Dict[str, str] = {}     # doc_id -> filename
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._doc_ids: List[str] = []
        self._is_built: bool = False

    # ── Index construction ────────────────────────────────────────────────────

    def build_index(self, kb_dir: Optional[Path] = None) -> None:
        """Load all Markdown files and build the TF-IDF index."""
        if kb_dir:
            self.kb_dir = Path(kb_dir)

        self._documents.clear()
        self._titles.clear()
        self._source_files.clear()
        self._doc_ids.clear()

        if not self.kb_dir.exists():
            logger.warning("Knowledge base directory not found: %s", self.kb_dir)
            self._is_built = False
            return

        for md_file in sorted(self.kb_dir.glob("*.md")):
            doc_id = md_file.stem
            try:
                text = md_file.read_text(encoding="utf-8")
                if not text.strip():
                    logger.debug("Skipping empty file: %s", md_file.name)
                    continue
                self._documents[doc_id] = text
                self._titles[doc_id] = self._extract_title(text, doc_id)
                self._source_files[doc_id] = md_file.name
            except Exception as exc:
                logger.warning("Could not load %s: %s", md_file.name, exc)

        if not self._documents:
            logger.warning("No documents loaded from %s", self.kb_dir)
            self._is_built = False
            return

        self._doc_ids = list(self._documents.keys())
        corpus = [self._clean_text(self._documents[d]) for d in self._doc_ids]

        self._vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
        self._is_built = True
        logger.info(
            "TF-IDF index built: %d documents, %d features",
            len(self._doc_ids),
            self._tfidf_matrix.shape[1],
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = MAX_RETRIEVAL_DOCS,
        min_score: float = MIN_RELEVANCE_SCORE,
    ) -> List[RetrievedDocument]:
        """
        Retrieve the most relevant documents for a query.

        Parameters
        ----------
        query : str
        top_k : int
        min_score : float  — documents below this score are excluded

        Returns
        -------
        List[RetrievedDocument] sorted by relevance (descending)
        """
        if not self._is_built:
            logger.warning("Retriever not built. Call build_index() first.")
            return []

        if not query.strip():
            logger.warning("Empty query supplied to retriever.")
            return []

        query_vec = self._vectorizer.transform([self._clean_text(query)])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        ranked_indices = np.argsort(scores)[::-1]
        results: List[RetrievedDocument] = []

        for idx in ranked_indices:
            score = float(scores[idx])
            if score < min_score:
                break
            if len(results) >= top_k:
                break
            doc_id = self._doc_ids[idx]
            results.append(
                RetrievedDocument(
                    doc_id=doc_id,
                    title=self._titles[doc_id],
                    content=self._documents[doc_id],
                    relevance_score=round(score, 4),
                    source_file=self._source_files[doc_id],
                )
            )

        logger.debug(
            "Retrieved %d documents for query: '%s…'",
            len(results),
            query[:60],
        )
        return results

    def retrieve_multiple_queries(
        self,
        queries: List[str],
        top_k_per_query: int = 3,
        global_top_k: int = MAX_RETRIEVAL_DOCS,
        min_score: float = MIN_RELEVANCE_SCORE,
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents for multiple queries, deduplicate, and return
        the top-k most relevant unique documents.
        """
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
    ) -> Tuple[List[RetrievedDocument], List[RetrievalTrace]]:
        """Retrieve unique documents and retain the query matches behind them.

        Scores are TF-IDF cosine similarities over the local knowledge base:
        useful for ranking within this index, not calibrated probabilities.
        """
        seen: Dict[str, RetrievedDocument] = {}
        trace: List[RetrievalTrace] = []
        for query in queries:
            for doc in self.retrieve(query, top_k=top_k_per_query, min_score=min_score):
                trace.append(RetrievalTrace(query, doc.doc_id, doc.title, doc.relevance_score))
                if doc.doc_id not in seen or doc.relevance_score > seen[doc.doc_id].relevance_score:
                    seen[doc.doc_id] = doc

        sorted_docs = sorted(seen.values(), key=lambda d: d.relevance_score, reverse=True)
        retained_ids = {doc.doc_id for doc in sorted_docs[:global_top_k]}
        return sorted_docs[:global_top_k], [item for item in trace if item.doc_id in retained_ids]

    # ── Inspection helpers ────────────────────────────────────────────────────

    def get_document(self, doc_id: str) -> Optional[str]:
        """Return the full text of a document by ID."""
        return self._documents.get(doc_id)

    def list_documents(self) -> List[str]:
        """Return list of all loaded document IDs."""
        return list(self._doc_ids)

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def n_documents(self) -> int:
        return len(self._doc_ids)

    # ── Private utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip Markdown syntax for cleaner TF-IDF representation while preserving technical terms."""
        text = re.sub(r"```[\s\S]*?```", " ", text)       # code blocks
        text = re.sub(r"`([^`]+)`", r" \1 ", text)        # inline code: preserve the technical terms
        text = re.sub(r"#{1,6}\s+", " ", text)            # headers
        text = re.sub(r"[*_]{1,3}(.+?)[*_]{1,3}", r"\1", text)  # bold/italic
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)   # links
        text = re.sub(r"[|`>]", " ", text)               # table, blockquote chars
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        """Extract the first H1 heading as the document title."""
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return fallback.replace("_", " ").title()
