"""
setup_pinecone.py
─────────────────
One-time setup: verify Pinecone connection, auto-fix dimension mismatches,
create the index if needed, then index all current products.

Run once before starting the app:
    python setup_pinecone.py
"""

import os
import sys
# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "pooja-store")

    if not api_key or api_key == "your_pinecone_api_key_here":
        print("ERROR: PINECONE_API_KEY is not set in your .env file.")
        print("       Copy .env.example to .env and fill in your credentials.")
        return

    from rag.embeddings import get_embedding_provider, get_embedding_dimension
    emb_provider = get_embedding_provider()
    emb_dim = get_embedding_dimension()
    print(f"[setup] Embedding provider : {emb_provider} ({emb_dim} dims)")

    print(f"[setup] Connecting to Pinecone...")
    from pinecone import Pinecone
    pc = Pinecone(api_key=api_key)
    print(f"[setup] Connected. Target index: '{index_name}'")

    print(f"\n[setup] Indexing products into Pinecone...")
    from rag.indexer import index_all_products
    count = index_all_products()

    print(f"\n[setup] Done! {count} products indexed into '{index_name}'.")
    print("[setup] You can now run:  python app.py")

if __name__ == "__main__":
    main()
