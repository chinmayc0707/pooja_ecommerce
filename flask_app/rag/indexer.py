"""
rag/indexer.py
──────────────
Indexes all products from the SQLite database into Pinecone.

Each product becomes one vector with rich text content for semantic search,
and metadata (product_id, name, price, category, url) returned as citations.

Usage:
    python rag/indexer.py          # run directly
    from rag.indexer import index_all_products   # call from app
"""

import os
import sys

# Allow running directly from flask_app root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from rag.embeddings import get_embeddings, get_embedding_dimension
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


def _get_embeddings():
    return get_embeddings()


def _get_or_create_index(pc: "Pinecone", index_name: str):
    """Create the Pinecone index, or recreate it if the dimension doesn't match."""
    import time
    dimension = get_embedding_dimension()
    existing_names = [idx.name for idx in pc.list_indexes()]

    if index_name in existing_names:
        # Check existing index dimension against what our embedding provider needs
        existing_index = pc.Index(index_name)
        try:
            stats = existing_index.describe_index_stats()
            existing_dim = stats.get("dimension", None)
        except Exception:
            existing_dim = None

        if existing_dim and existing_dim != dimension:
            print(f"[indexer] ⚠  Dimension mismatch detected!")
            print(f"[indexer]    Index '{index_name}' has {existing_dim} dims")
            print(f"[indexer]    Embedding provider needs {dimension} dims")
            print(f"[indexer]    Deleting and recreating index...")
            pc.delete_index(index_name)
            # Wait for Pinecone to finish deletion
            for i in range(12):
                time.sleep(5)
                remaining = [idx.name for idx in pc.list_indexes()]
                if index_name not in remaining:
                    break
                print(f"[indexer]    Waiting for deletion... ({(i+1)*5}s)")
            print(f"[indexer]    Old index deleted.")
        else:
            print(f"[indexer] Index '{index_name}' exists with correct dimension ({dimension}).")
            return existing_index

    print(f"[indexer] Creating Pinecone index '{index_name}' (dim={dimension})...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for index to be ready
    for i in range(12):
        time.sleep(5)
        try:
            idx = pc.Index(index_name)
            idx.describe_index_stats()
            break
        except Exception:
            print(f"[indexer]    Waiting for index to be ready... ({(i+1)*5}s)")
    print(f"[indexer] Index '{index_name}' ready.")
    return pc.Index(index_name)


def _product_to_document(product) -> Document:
    """Convert a Product ORM object into a LangChain Document."""
    # Rich text for semantic similarity search
    content = (
        f"Product: {product.name}\n"
        f"Category: {product.category}\n"
        f"Price: ₹{product.price:.2f}\n"
        f"Stock: {product.stock} units available\n"
        f"Description: {product.description or 'No description provided.'}"
    )
    metadata = {
        "product_id": product.id,
        "name": product.name,
        "price": float(product.price),
        "category": product.category,
        "url": f"/product/{product.id}",
    }
    return Document(page_content=content, metadata=metadata)


def index_all_products(app=None):
    """
    Index all products into Pinecone.

    Args:
        app: Flask app instance (required when called from within the app context).
             If None, creates its own app context.
    """
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "pooja-store")

    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set in environment/.env")

    pc = Pinecone(api_key=api_key)
    index = _get_or_create_index(pc, index_name)

    embeddings = _get_embeddings()

    # Need Flask app context to query the DB
    if app is not None:
        with app.app_context():
            return _do_index(index, index_name, embeddings, app)
    else:
        # Running as standalone script — create the app ourselves
        from app import app as flask_app, Product, init_db
        init_db()
        with flask_app.app_context():
            return _do_index(index, index_name, embeddings, flask_app)


def _do_index(index, index_name, embeddings, app):
    from app import Product

    products = Product.query.all()
    if not products:
        print("[indexer] No products found in database.")
        return 0

    print(f"[indexer] Indexing {len(products)} products...")

    # Clear existing vectors and re-index fresh
    try:
        index.delete(delete_all=True)
        print("[indexer] Cleared existing vectors.")
    except Exception as e:
        print(f"[indexer] Note: could not clear index (may be empty): {e}")

    docs = [_product_to_document(p) for p in products]
    ids = [f"product_{p.id}" for p in products]

    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    vectorstore.add_documents(docs, ids=ids)

    print(f"[indexer] ✓ Successfully indexed {len(docs)} products into '{index_name}'.")
    return len(docs)


def index_single_product(product, pc=None, index=None, embeddings=None):
    """
    Upsert a single product (for add/edit operations).
    Pass pre-built pc/index/embeddings to avoid re-initializing on every call.
    """
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "pooja-store")

    if pc is None:
        pc = Pinecone(api_key=api_key)
    if index is None:
        index = _get_or_create_index(pc, index_name)
    if embeddings is None:
        embeddings = _get_embeddings()

    doc = _product_to_document(product)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    vectorstore.add_documents([doc], ids=[f"product_{product.id}"])
    print(f"[indexer] Upserted product {product.id} ({product.name}).")


def delete_product_vector(product_id: int):
    """Remove a product's vector from Pinecone (call on product delete)."""
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "pooja-store")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    index.delete(ids=[f"product_{product_id}"])
    print(f"[indexer] Deleted vector for product {product_id}.")


if __name__ == "__main__":
    count = index_all_products()
    print(f"[indexer] Done. {count} products indexed.")
