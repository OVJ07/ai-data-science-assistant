"""
src/rag/dense_retriever.py
──────────────────────────
Dense (embedding-based) retrieval using Ollama's local embedding API.
Embeddings are cached to disk keyed by (model_name, doc_id, content_hash)
to avoid regenerating on every Streamlit rerun.

Never labels results as embedding results unless they genuinely are.
Ollama unavailability, missing models, and malformed responses are
handled gracefully and reported explicitly.
"""

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    EMBEDDING_CACHE_DIR,
    EMBEDDING_MODEL,
    MAX_RETRIEVAL_DOCS,
    MIN_RELEVANCE_SCORE,
    OLLAMA_EMBED_BASE_URL,
    KNOWLEDGE_BASE_DIR,
)
from src.rag.retriever import RetrievedDocument

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = [
    "qwen3-embedding:4b",
    "nomic-embed-text:latest",
    "qwen3-embedding:0.6b",
]


# ── Custom exceptions ─────────────────────────────────────────────────────────

class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama server is not reachable."""


class ModelNotInstalledError(RuntimeError):
    """Raised when the embedding model is not installed in Ollama."""


# ── Dense retriever ───────────────────────────────────────────────────────────

class OllamaDenseRetriever:
    """
    Dense (embedding-based) retriever backed by a locally-running Ollama server.

    Documents are embedded once (on ``build_index``) and embeddings are cached
    to disk so Streamlit reruns do not repeat the expensive API calls. At query
    time the query is embedded on-the-fly and cosine similarity is used to rank
    documents.

    Parameters
    ----------
    kb_dir : Path, optional
        Directory containing Markdown knowledge-base files.  Defaults to
        ``KNOWLEDGE_BASE_DIR`` from settings.
    embedding_model : str
        Ollama model name to use for embeddings.
    base_url : str
        Base URL of the Ollama HTTP API (e.g. ``http://localhost:11434``).
    cache_dir : Path, optional
        Directory where ``.npy`` embedding cache files are stored.  Defaults to
        ``EMBEDDING_CACHE_DIR`` from settings.

    Raises
    ------
    OllamaUnavailableError
        If the Ollama server cannot be reached during ``build_index`` or
        ``retrieve``.
    ModelNotInstalledError
        If the specified embedding model is not installed in Ollama.
    """

    def __init__(
        self,
        kb_dir: Optional[Path] = None,
        embedding_model: str = EMBEDDING_MODEL,
        base_url: str = OLLAMA_EMBED_BASE_URL,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.kb_dir: Path = Path(kb_dir) if kb_dir else KNOWLEDGE_BASE_DIR
        self.embedding_model: str = embedding_model
        self.base_url: str = base_url.rstrip("/")
        self.cache_dir: Path = Path(cache_dir) if cache_dir else EMBEDDING_CACHE_DIR

        # Document storage
        self._documents: Dict[str, str] = {}
        self._titles: Dict[str, str] = {}
        self._source_files: Dict[str, str] = {}
        self._doc_ids: List[str] = []

        # Embedding matrix — shape [n_docs, dim] after build
        self._doc_embeddings: Optional[np.ndarray] = None
        self._embedding_dim: Optional[int] = None

        # State tracking
        self._is_built: bool = False
        self._build_time_seconds: float = 0.0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    # ── Index construction ────────────────────────────────────────────────────

    def build_index(
        self,
        kb_dir: Optional[Path] = None,
        force_rebuild: bool = False,
    ) -> dict:
        """
        Load Markdown files and build the dense embedding index.

        For each document the method first checks whether a cached embedding
        exists on disk.  Only documents without a valid cache entry are sent to
        Ollama.  The resulting embeddings are stacked into a float32 ndarray
        for fast cosine similarity computation.

        Parameters
        ----------
        kb_dir : Path, optional
            Override the knowledge-base directory for this build.
        force_rebuild : bool
            When ``True`` all cached embeddings are ignored and regenerated.

        Returns
        -------
        dict
            Keys: ``success`` (bool), ``n_documents`` (int), ``cache_hits``
            (int), ``cache_misses`` (int), ``build_time_seconds`` (float),
            ``embedding_dim`` (int or None), ``error`` (str or None).
        """
        t0 = time.monotonic()
        if kb_dir:
            self.kb_dir = Path(kb_dir)

        # Reset state
        self._documents.clear()
        self._titles.clear()
        self._source_files.clear()
        self._doc_ids.clear()
        self._doc_embeddings = None
        self._embedding_dim = None
        self._is_built = False
        self._cache_hits = 0
        self._cache_misses = 0

        if not self.kb_dir.exists():
            logger.warning("Knowledge base directory not found: %s", self.kb_dir)
            return self._build_result(False, 0, error=f"Directory not found: {self.kb_dir}")

        # Load documents
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
            return self._build_result(False, 0, error="No documents loaded")

        self._doc_ids = list(self._documents.keys())
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        embeddings: List[np.ndarray] = []
        dims_seen: set = set()

        for doc_id in self._doc_ids:
            cleaned = self._clean_text(self._documents[doc_id])
            cache_key = self._cache_key(doc_id, cleaned)
            cache_path = self._cache_path(cache_key)

            embedding: Optional[np.ndarray] = None

            if not force_rebuild and cache_path.exists():
                try:
                    embedding = np.load(str(cache_path))
                    self._cache_hits += 1
                    logger.debug("Cache hit for doc '%s'", doc_id)
                except Exception as exc:
                    logger.warning("Failed to load cached embedding for '%s': %s", doc_id, exc)
                    embedding = None

            if embedding is None:
                try:
                    embedding = self._embed_text(cleaned)
                except (OllamaUnavailableError, ModelNotInstalledError) as exc:
                    elapsed = time.monotonic() - t0
                    self._build_time_seconds = elapsed
                    return self._build_result(
                        False,
                        len(embeddings),
                        error=str(exc),
                    )
                self._cache_misses += 1
                if embedding is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(cache_path), embedding)
                    logger.debug("Cached embedding for doc '%s'", doc_id)

            if embedding is None:
                logger.warning("Skipping doc '%s': embedding returned None", doc_id)
                continue

            dims_seen.add(embedding.shape[0])
            embeddings.append(embedding)

        if not embeddings:
            return self._build_result(False, 0, error="All document embeddings failed")

        if len(dims_seen) > 1:
            return self._build_result(
                False,
                len(embeddings),
                error=f"Inconsistent embedding dimensions: {dims_seen}",
            )

        self._doc_embeddings = np.stack(embeddings, axis=0).astype(np.float32)
        self._embedding_dim = self._doc_embeddings.shape[1]
        self._is_built = True
        self._build_time_seconds = time.monotonic() - t0

        logger.info(
            "Dense index built: %d documents, dim=%d, cache_hits=%d, cache_misses=%d, %.2fs",
            len(self._doc_ids),
            self._embedding_dim,
            self._cache_hits,
            self._cache_misses,
            self._build_time_seconds,
        )
        return self._build_result(True, len(self._doc_ids))

    def _build_result(
        self,
        success: bool,
        n_documents: int,
        error: Optional[str] = None,
    ) -> dict:
        """Construct the standard build_index return dict."""
        return {
            "success": success,
            "n_documents": n_documents,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "build_time_seconds": round(self._build_time_seconds, 3),
            "embedding_dim": self._embedding_dim,
            "error": error,
        }

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Call the Ollama ``/api/embeddings`` endpoint and return a float32 array.

        Parameters
        ----------
        text : str
            The text to embed.  An empty string returns ``None`` immediately.

        Returns
        -------
        np.ndarray of shape (dim,) and dtype float32, or None on malformed
        responses.

        Raises
        ------
        OllamaUnavailableError
            When the Ollama server is not reachable (connection refused, etc.).
        ModelNotInstalledError
            When Ollama returns HTTP 404 or an error body mentioning the model.
        """
        import requests  # local import — only needed when Ollama is in use

        if not text.strip():
            return None

        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.embedding_model, "prompt": text}

        try:
            resp = requests.post(url, json=payload, timeout=60)
        except requests.exceptions.ConnectionError as exc:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.base_url}: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaUnavailableError(
                f"Request to Ollama failed: {exc}"
            ) from exc

        # HTTP-level error handling
        if resp.status_code == 404:
            raise ModelNotInstalledError(
                f"Model '{self.embedding_model}' not found in Ollama (HTTP 404). "
                "Run: ollama pull " + self.embedding_model
            )

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("Malformed JSON from Ollama embeddings endpoint: %s", exc)
            return None

        # Detect model-not-installed in error responses
        if "error" in data:
            err_msg: str = str(data["error"]).lower()
            if "model" in err_msg or self.embedding_model.lower() in err_msg:
                raise ModelNotInstalledError(
                    f"Ollama reports model error: {data['error']}"
                )
            logger.warning("Ollama embeddings error: %s", data["error"])
            return None

        if "embedding" not in data:
            logger.warning("Ollama response missing 'embedding' key: %s", list(data.keys()))
            return None

        raw = data["embedding"]
        if not isinstance(raw, list) or len(raw) == 0:
            logger.warning("Ollama returned empty or invalid embedding list")
            return None

        return np.array(raw, dtype=np.float32)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = MAX_RETRIEVAL_DOCS,
        min_score: float = MIN_RELEVANCE_SCORE,
    ) -> List[RetrievedDocument]:
        """
        Retrieve the most relevant documents for *query* using cosine similarity
        over dense embeddings.

        Parameters
        ----------
        query : str
            Natural-language query string.
        top_k : int
            Maximum number of documents to return.
        min_score : float
            Documents with cosine similarity below this threshold are excluded.

        Returns
        -------
        List[RetrievedDocument] sorted by relevance descending.
        """
        if not self._is_built:
            logger.warning("DenseRetriever not built. Call build_index() first.")
            return []

        if not query.strip():
            logger.warning("Empty query supplied to dense retriever.")
            return []

        try:
            query_vec = self._embed_text(self._clean_text(query))
        except (OllamaUnavailableError, ModelNotInstalledError) as exc:
            logger.warning("Dense retrieval skipped — Ollama error: %s", exc)
            return []

        if query_vec is None:
            logger.warning("Query embedding returned None; skipping dense retrieval.")
            return []

        if self._embedding_dim is not None and len(query_vec) != self._embedding_dim:
            logger.error(
                "Query embedding dimension (%d) does not match index dimension (%d). Skipping dense retrieval.",
                len(query_vec),
                self._embedding_dim,
            )
            return []

        scores = self._cosine_similarity(query_vec, self._doc_embeddings)
        ranked_indices = np.argsort(scores)[::-1]

        results: List[RetrievedDocument] = []
        for idx in ranked_indices:
            if len(results) >= top_k:
                break
            score = float(scores[idx])
            if score < min_score:
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
            "Dense retrieval: %d documents for query '%s…'",
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
        Retrieve documents for multiple queries, deduplicate, and return the
        top-k most relevant unique documents.

        When the same document is retrieved by multiple queries the highest
        cosine similarity score is kept.

        Parameters
        ----------
        queries : List[str]
            One or more query strings.
        top_k_per_query : int
            Candidates retrieved per individual query.
        global_top_k : int
            Maximum unique documents to return overall.
        min_score : float
            Per-document minimum similarity threshold.

        Returns
        -------
        List[RetrievedDocument] sorted by relevance descending.
        """
        seen: Dict[str, RetrievedDocument] = {}
        for query in queries:
            for doc in self.retrieve(query, top_k=top_k_per_query, min_score=min_score):
                if (
                    doc.doc_id not in seen
                    or doc.relevance_score > seen[doc.doc_id].relevance_score
                ):
                    seen[doc.doc_id] = doc

        sorted_docs = sorted(seen.values(), key=lambda d: d.relevance_score, reverse=True)
        return sorted_docs[:global_top_k]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_built(self) -> bool:
        """True if ``build_index`` completed successfully."""
        return self._is_built

    @property
    def n_documents(self) -> int:
        """Number of indexed documents."""
        return len(self._doc_ids)

    @property
    def embedding_dim(self) -> Optional[int]:
        """Dimensionality of the embedding vectors, or None if not built."""
        return self._embedding_dim

    @property
    def cache_stats(self) -> dict:
        """
        Cache performance statistics from the last ``build_index`` call.

        Returns
        -------
        dict with keys ``hits`` (int), ``misses`` (int), ``hit_rate`` (float).
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": round(hit_rate, 4),
        }

    # ── Private utilities ─────────────────────────────────────────────────────

    def _cache_key(self, doc_id: str, content: str) -> str:
        """
        Compute a deterministic SHA-256 cache key for a document embedding.

        The key encodes the model name, document ID, and content so that any
        change to any of those three factors invalidates the cache entry.
        """
        raw = f"{self.embedding_model}|{doc_id}|{content}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        """
        Return the filesystem path for a cache entry.

        Cache files live under ``{cache_dir}/{model_short}/{cache_key}.npy``
        where *model_short* is the model name sanitised to safe characters and
        truncated to 32 characters.
        """
        model_short = re.sub(r"[^a-zA-Z0-9_-]", "_", self.embedding_model)[:32]
        return self.cache_dir / model_short / f"{cache_key}.npy"

    @staticmethod
    def _cosine_similarity(
        query_vec: np.ndarray,
        doc_vecs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between a single query vector and a matrix of
        document vectors.

        A small epsilon (1e-10) is added to the denominator to avoid
        division-by-zero for zero vectors.

        Parameters
        ----------
        query_vec : np.ndarray, shape (dim,)
        doc_vecs  : np.ndarray, shape (n_docs, dim)

        Returns
        -------
        np.ndarray of shape (n_docs,) with similarity scores in [-1, 1].
        """
        return np.dot(query_vec, doc_vecs.T) / (
            np.linalg.norm(query_vec) * np.linalg.norm(doc_vecs, axis=1) + 1e-10
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip Markdown syntax for cleaner embedding representation."""
        text = re.sub(r"```[\s\S]*?```", " ", text)       # code blocks
        text = re.sub(r"`[^`]+`", " ", text)               # inline code
        text = re.sub(r"#{1,6}\s+", " ", text)             # headers
        text = re.sub(r"[*_]{1,3}(.+?)[*_]{1,3}", r"\1", text)   # bold/italic
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)     # links
        text = re.sub(r"[|`>]", " ", text)                 # table, blockquote chars
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
