"""
config/settings.py
──────────────────
Central configuration for the AI Data Science Assistant.
All tunable constants live here so they can be overridden via .env.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env if present ────────────────────────────────────────────────────
load_dotenv()

# ── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
DATA_DIR = BASE_DIR / "data"

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", 42))

# ── Train / test split ───────────────────────────────────────────────────────
TEST_SIZE: float = float(os.getenv("TEST_SIZE", 0.2))

# ── RAG / Retrieval ──────────────────────────────────────────────────────────
MAX_RETRIEVAL_DOCS: int = int(os.getenv("MAX_RETRIEVAL_DOCS", 5))
MIN_RELEVANCE_SCORE: float = 0.05          # documents below this are dropped
TFIDF_MAX_FEATURES: int = 5_000
TFIDF_NGRAM_RANGE: tuple = (1, 2)

# ── LLM Provider ─────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "none").lower()
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 8_192))
# Query generation runs only after an explicit Section 6 request.  Reasoning
# models can consume the general completion budget before emitting JSON, so it
# has a separately configurable ceiling.
LLM_QUERY_MAX_TOKENS: int = int(os.getenv("LLM_QUERY_MAX_TOKENS", 4_096))
LLM_TEMPERATURE: float = 0.2
LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", 120))   # seconds
# Query generation never sends dataset-derived metadata to a remote endpoint
# unless the deployment owner explicitly opts in.  Local OpenAI-compatible
# endpoints (for example Ollama/vLLM) remain eligible without this flag.
LLM_ALLOW_REMOTE_DATA: bool = os.getenv("LLM_ALLOW_REMOTE_DATA", "false").lower() in {"1", "true", "yes"}
LLM_QUERY_GENERATION_ENABLED: bool = os.getenv("LLM_QUERY_GENERATION_ENABLED", "true").lower() in {"1", "true", "yes"}

# ── ML ────────────────────────────────────────────────────────────────────────
CV_FOLDS: int = 5
HIGH_CARDINALITY_THRESHOLD: int = 20      # unique values → high-cardinality
ID_LIKE_UNIQUE_RATIO: float = 0.95        # unique / total → likely an ID column
CLASS_IMBALANCE_RATIO: float = 0.20       # minority / majority below this → imbalanced
MIN_SAMPLES_FOR_ML: int = 30
MAX_ONE_HOT_CATEGORIES: int = 15          # warn if OHE would exceed this

# ── XAI ──────────────────────────────────────────────────────────────────────
SHAP_MAX_DISPLAY: int = 15               # top-N features in SHAP plots
LIME_NUM_FEATURES: int = 10
LIME_NUM_SAMPLES: int = 500

# ── Reporting ────────────────────────────────────────────────────────────────
REPORT_TEMPLATE_DIR = BASE_DIR / "src" / "reporting" / "templates"

# ── Dense / Hybrid retrieval ──────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")
EMBEDDING_CACHE_DIR: Path = BASE_DIR / ".embedding_cache"
OLLAMA_EMBED_BASE_URL: str = os.getenv("OLLAMA_EMBED_BASE_URL", "http://localhost:11434")
# Supported embedding models (comma-separated in .env if overriding)
SUPPORTED_EMBEDDING_MODELS: list = [
    "qwen3-embedding:4b",
    "nomic-embed-text:latest",
    "qwen3-embedding:0.6b",
]
# Reciprocal Rank Fusion constant (k in 1/(k+rank))
RRF_K: int = int(os.getenv("RRF_K", 60))
# Number of candidates to fetch from each retriever before RRF fusion
HYBRID_TFIDF_CANDIDATES: int = int(os.getenv("HYBRID_TFIDF_CANDIDATES", 10))
HYBRID_DENSE_CANDIDATES: int = int(os.getenv("HYBRID_DENSE_CANDIDATES", 10))

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
