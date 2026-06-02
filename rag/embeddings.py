"""
rag/embeddings.py
─────────────────
Pluggable embedding factory.

Controlled by EMBEDDING_PROVIDER env var:
  - openai          → OpenAIEmbeddings (text-embedding-3-small, 1536 dims) — requires OPENAI_API_KEY
  - huggingface     → HuggingFaceEmbeddings (all-MiniLM-L6-v2, 384 dims)  — free, runs locally (needs PyTorch)
  - huggingface_api → HuggingFace Inference API (all-MiniLM-L6-v2, 384 dims) — free, runs remotely (NO PyTorch!)

Auto-detection priority:
  1. Explicit EMBEDDING_PROVIDER env var
  2. 'openai' if OPENAI_API_KEY is set
  3. 'huggingface' if torch is importable (local dev)
  4. 'huggingface_api' as final fallback (cloud deploy without PyTorch)

IMPORTANT: The Pinecone index dimension MUST match the embedding dimension.
  - openai          → dim=1536
  - huggingface     → dim=384
  - huggingface_api → dim=384  (same model, same dims — compatible with huggingface)
  If you switch between openai and huggingface*, run setup_pinecone.py again.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Suppress HuggingFace Hub unauthenticated warning
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Dimension map per provider
EMBEDDING_DIMS = {
    "openai": 1536,
    "huggingface": 384,
    "huggingface_api": 384,
}

_HF_API_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{_HF_API_MODEL}"


def _torch_available() -> bool:
    """Check if PyTorch is importable without actually loading it."""
    try:
        import importlib.util
        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False


def get_embedding_provider() -> str:
    """Resolve which embedding provider to use."""
    provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
    if provider in EMBEDDING_DIMS:
        return provider
    # Auto-detect
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    # Use local HuggingFace if torch is available, else fall back to API
    if _torch_available():
        return "huggingface"
    logger.info("PyTorch not available — using HuggingFace Inference API for embeddings.")
    return "huggingface_api"


# ─── Lightweight HuggingFace API Embeddings (no PyTorch needed) ───────────────
class _HuggingFaceAPIEmbeddings:
    """
    Minimal LangChain-compatible embeddings class that calls the free
    HuggingFace Inference API.  Uses the same model (all-MiniLM-L6-v2)
    as the local provider, so Pinecone vectors are fully compatible.

    Requires only the `requests` library (already a Flask dependency).
    """

    def __init__(self, api_url: str = _HF_API_URL, api_token: str | None = None):
        self._api_url = api_url
        self._headers = {"Content-Type": "application/json"}
        token = api_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        import requests

        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True},
        }
        resp = requests.post(self._api_url, json=payload, headers=self._headers, timeout=60)
        if resp.status_code == 503:
            # Model is loading — retry once
            import time
            time.sleep(5)
            resp = requests.post(self._api_url, json=payload, headers=self._headers, timeout=60)

        resp.raise_for_status()
        data = resp.json()

        # API returns list of lists of floats
        if isinstance(data, list) and len(data) > 0:
            # Might be nested: [[float, ...], ...] or [[[float, ...]]]
            if isinstance(data[0], list) and isinstance(data[0][0], float):
                return data  # Already flat: [[float, ...], [float, ...]]
            elif isinstance(data[0], list) and isinstance(data[0][0], list):
                # Token-level embeddings — mean-pool to get sentence embedding
                import statistics
                return [
                    [statistics.mean(token_vals) for token_vals in zip(*token_embeddings)]
                    for token_embeddings in data
                ]
        raise ValueError(f"Unexpected HF API response format: {type(data)}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        # HF Inference API has a limit; batch in chunks of 32
        all_embeddings = []
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            all_embeddings.extend(self._call_api(batch))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._call_api([text])[0]


def get_embeddings():
    """Return a LangChain-compatible embeddings object for the configured provider."""
    provider = get_embedding_provider()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            logger.info(
                "Loading HuggingFace embedding model 'all-MiniLM-L6-v2' locally "
                "(this may take a moment on first run)..."
            )
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("HuggingFace embedding model loaded successfully.")
            return embeddings
        except Exception as e:
            logger.warning(
                f"Local HuggingFace embeddings failed ({e}). "
                "Falling back to HuggingFace Inference API..."
            )
            # Fall through to API provider
            return _HuggingFaceAPIEmbeddings()

    elif provider == "huggingface_api":
        logger.info("Using HuggingFace Inference API for embeddings (no PyTorch required).")
        return _HuggingFaceAPIEmbeddings()

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: {provider!r}. "
            "Choose from: openai, huggingface, huggingface_api"
        )


def get_embedding_dimension() -> int:
    """Return the vector dimension for the current embedding provider."""
    return EMBEDDING_DIMS[get_embedding_provider()]
