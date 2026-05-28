"""
rag/embeddings.py
─────────────────
Pluggable embedding factory.

Controlled by EMBEDDING_PROVIDER env var:
  - openai      → OpenAIEmbeddings (text-embedding-3-small, 1536 dims) — requires OPENAI_API_KEY
  - huggingface → HuggingFaceEmbeddings (all-MiniLM-L6-v2, 384 dims)  — free, runs locally

Auto-detection: if EMBEDDING_PROVIDER is not set, uses 'openai' when OPENAI_API_KEY is present,
otherwise falls back to 'huggingface'.

IMPORTANT: The Pinecone index dimension MUST match the embedding dimension.
  - openai      → create index with dim=1536
  - huggingface → create index with dim=384
  If you switch providers, run setup_pinecone.py again (it will recreate the index).
"""

import os

# Suppress HuggingFace Hub unauthenticated warning — model runs locally after first download
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Dimension map per provider
EMBEDDING_DIMS = {
    "openai": 1536,
    "huggingface": 384,
}


def get_embedding_provider() -> str:
    """Resolve which embedding provider to use."""
    provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
    if provider in EMBEDDING_DIMS:
        return provider
    # Auto-detect: fall back to huggingface if no OpenAI key
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "huggingface"


def get_embeddings():
    """Return a LangChain embeddings object for the configured provider."""
    provider = get_embedding_provider()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: {provider!r}. Choose from: openai, huggingface"
        )


def get_embedding_dimension() -> int:
    """Return the vector dimension for the current embedding provider."""
    return EMBEDDING_DIMS[get_embedding_provider()]
