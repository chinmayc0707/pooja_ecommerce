"""
rag/embeddings.py
─────────────────
Pluggable embedding factory.

Controlled by two env vars:
  EMBEDDING_PROVIDER  — which backend to use
  EMBED_MODEL         — override the default model name for any provider

Supported providers:
  - pinecone      → Pinecone Inference API (multilingual-e5-large, 1024 dims)
                    No extra key needed — reuses PINECONE_API_KEY.
  - openai        → OpenAI (text-embedding-3-small, 1536 dims)
  - google        → Google GenAI (models/text-embedding-004, 768 dims)
  - huggingface   → HuggingFace local (all-MiniLM-L6-v2, 384 dims) — NOT
                    recommended on memory-constrained servers (needs PyTorch).

Auto-detection priority (when EMBEDDING_PROVIDER is not set):
  pinecone (PINECONE_API_KEY) → openai (OPENAI_API_KEY) → google (GOOGLE_API_KEY) → huggingface

IMPORTANT: The Pinecone index dimension MUST match the embedding dimension.
  If you switch providers, run  python setup_pinecone.py  again (it will
  auto-detect the mismatch and recreate the index).
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ─── Default model names per provider ─────────────────────────────────────────
DEFAULT_MODELS = {
    "pinecone": "multilingual-e5-large",
    "openai": "text-embedding-3-small",
    "google": "models/text-embedding-004",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
}

# ─── Dimension map per provider/model ────────────────────────────────────────
EMBEDDING_DIMS = {
    "pinecone": 1024,
    "openai": 1536,
    "google": 768,
    "huggingface": 384,
}

# ─── The constant: resolved once at import time ──────────────────────────────
EMBED_MODEL: str = ""  # set lazily by get_embedding_provider()


def get_embedding_provider() -> str:
    """Resolve which embedding provider to use."""
    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if provider in DEFAULT_MODELS:
        return provider

    # Auto-detect — prefer cloud providers that don't blow up memory
    if os.getenv("PINECONE_API_KEY"):
        return "pinecone"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GOOGLE_API_KEY"):
        return "google"
    return "huggingface"


def _resolve_model(provider: str) -> str:
    """Return the model name: user override via EMBED_MODEL env, or default."""
    global EMBED_MODEL
    user_model = os.getenv("EMBED_MODEL", "").strip()
    EMBED_MODEL = user_model or DEFAULT_MODELS.get(provider, "")
    return EMBED_MODEL


def get_embeddings():
    """Return a LangChain embeddings object for the configured provider."""
    provider = get_embedding_provider()
    model = _resolve_model(provider)
    logger.info("Embedding provider: %s | model: %s", provider, model)

    if provider == "pinecone":
        from langchain_pinecone import PineconeEmbeddings

        return PineconeEmbeddings(
            model=model,
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        )

    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    elif provider == "huggingface":
        # Suppress HuggingFace Hub unauthenticated warning
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            logger.info(
                "Loading HuggingFace embedding model '%s' "
                "(this may take a moment on first run)...",
                model,
            )
            embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("HuggingFace embedding model loaded successfully.")
            return embeddings
        except Exception as e:
            raise RuntimeError(
                f"Failed to load HuggingFace embedding model: {e}. "
                "This often means the server ran out of memory or couldn't "
                "download the model. Consider switching to a cloud provider: "
                "set EMBEDDING_PROVIDER=pinecone in your environment."
            ) from e

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: {provider!r}. "
            "Choose from: pinecone, openai, google, huggingface"
        )


def get_embedding_dimension() -> int:
    """Return the vector dimension for the current embedding provider."""
    provider = get_embedding_provider()
    return EMBEDDING_DIMS.get(provider, 1024)
