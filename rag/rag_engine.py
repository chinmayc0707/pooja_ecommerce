"""
rag/rag_engine.py
─────────────────
Agentic RAG pipeline for Pooja Ecommerce.

The LLM receives retrieved product context and produces STRUCTURED output:
  - answer          : the text response shown to the user
  - cited_product_ids: list of product IDs the LLM explicitly chose to recommend

Only products the LLM decided to mention appear as citation cards — not the
full retrieval set.

Supports pluggable LLMs (openai | openrouter | google | anthropic | groq)
via LLM_PROVIDER env var, and pluggable embeddings via EMBEDDING_PROVIDER.
"""

import os
import logging
import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessageChunk

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
logger = logging.getLogger(__name__)



# ─── Pinecone vector store (lazy singleton) ───────────────────────────────────
_vectorstore = None

def _get_vectorstore():
    """Lazy-init the Pinecone vectorstore."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore
    from rag.embeddings import get_embeddings

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.getenv("PINECONE_INDEX_NAME", "pooja-store")
    index = pc.Index(index_name)
    embeddings = get_embeddings()

    _vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    return _vectorstore


def _product_to_document(product) -> Document:
    content = (
        f"Product: {product.name}\n"
        f"Category: {product.category}\n"
        f"Price: Rs. {float(product.price):.2f}\n"
        f"Stock: {product.stock} units available\n"
        f"Description: {product.description or 'No description provided.'}"
    )
    metadata = {
        "product_id": int(product.id),
        "name": product.name,
        "price": float(product.price),
        "category": product.category,
        "url": f"/product/{product.id}",
    }
    return Document(page_content=content, metadata=metadata)


def _catalog_docs_from_supabase(question: str, limit: int = 6) -> list[Document]:
    from app import app as flask_app, _get_all_products

    with flask_app.app_context():
        products = _get_all_products()

    docs = [_product_to_document(product) for product in products]
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", (question or "").lower())
        if len(token) > 2
    }
    if not tokens:
        return docs[:limit]

    def score(doc: Document) -> int:
        haystack = f"{doc.page_content} {doc.metadata.get('name', '')} {doc.metadata.get('category', '')}".lower()
        return sum(1 for token in tokens if token in haystack)

    ranked = sorted(docs, key=score, reverse=True)
    matched = [doc for doc in ranked if score(doc) > 0]
    return (matched or ranked)[:limit]


def _retrieve_docs(question: str) -> list[Document]:
    """Retrieve relevant product docs via Pinecone, with Supabase fallback."""
    if os.environ.get("PINECONE_API_KEY"):
        try:
            vectorstore = _get_vectorstore()
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6},
            )
            docs = retriever.invoke(question)
            if docs:
                return docs
            logger.warning("Pinecone returned no matching documents; falling back to Supabase catalog.")
        except Exception as exc:
            logger.warning("Pinecone retrieval failed; falling back to Supabase catalog: %s", exc)

    return _catalog_docs_from_supabase(question)



# ─── Pluggable LLM factory ────────────────────────────────────────────────────
def _get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model_name = os.getenv("LLM_MODEL_NAME", "")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "meta-llama/llama-3.1-8b-instruct:free",
            temperature=0.2,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            request_timeout=90,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:5000"),
                "X-Title": os.getenv("OPENROUTER_SITE_NAME", "Pooja Ecommerce"),
            },
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-1.5-flash",
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name or "claude-3-haiku-20240307",
            temperature=0.2,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name or "llama-3.1-8b-instant",
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            "Choose from: openai, openrouter, google, anthropic, groq"
        )


# ─── Format retrieved docs for the prompt ────────────────────────────────────
def _format_docs(docs: list[Document]) -> str:
    """Render retrieved docs into a readable block with IDs clearly labelled."""
    if not docs:
        return "No matching products found in catalog."
    parts = []
    for doc in docs:
        meta = doc.metadata
        parts.append(
            f"--- Product ID: {meta.get('product_id')} ---\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def _clean_answer(text: str) -> str:
    """
    Safety net: if the LLM returned raw JSON instead of plain text,
    extract just the 'answer' field so the user never sees raw JSON.
    """
    import json
    text = text.strip()
    # Strip ```json ... ``` fence if present
    fenced = re.match(r'^```(?:json)?\s*([\s\S]+?)\s*```$', text)
    candidate = fenced.group(1) if fenced else text
    if candidate.lstrip().startswith('{'):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and isinstance(data.get('answer'), str):
                return data['answer']
        except (json.JSONDecodeError, ValueError):
            pass
    return text


def _extract_cited_ids(text: str) -> list[int]:
    """
    Try to extract cited_product_ids from the LLM's text response.
    Handles JSON blocks, fenced JSON, or inline JSON fragments.
    """
    import json
    text = text.strip()
    # Strip ```json ... ``` fence if present
    fenced = re.match(r'^```(?:json)?\s*([\s\S]+?)\s*```$', text)
    candidate = fenced.group(1) if fenced else text
    if candidate.lstrip().startswith('{'):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and 'cited_product_ids' in data:
                return [int(x) for x in data['cited_product_ids']]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: look for cited_product_ids anywhere in the text
    match = re.search(r'cited_product_ids["\s:]*\[([^\]]*)\]', text)
    if match:
        try:
            return [int(x.strip()) for x in match.group(1).split(',') if x.strip()]
        except (ValueError, TypeError):
            pass
    return []



@tool
def add_to_cart_tool(product_id: int) -> str:
    """Add a specific product to the user's shopping cart. Call this when the user explicitly asks to buy or add a product to their cart. You must provide the exact product_id."""
    from app import db, CartItem, _get_cart_identity, app as flask_app
    try:
        # Since this tool is invoked by LangChain inside a Flask request context,
        # we can access flask globals like session.
        # But just to be safe if the context is lost (it shouldn't be):
        user_id, session_id = _get_cart_identity()
        item = CartItem.query.filter_by(user_id=user_id, session_id=session_id, product_id=product_id).first()
        if item:
            item.quantity += 1
        else:
            item = CartItem(user_id=user_id, session_id=session_id, product_id=product_id, quantity=1)
            db.session.add(item)
        db.session.commit()
        return f"Successfully added product ID {product_id} to the cart. Tell the user it has been added."
    except Exception as e:
        logger.error(f"Error adding to cart via AI: {e}")
        return f"Failed to add product to cart: {str(e)}"

# ─── Agentic system prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable and warm shopping assistant for Pooja Ecommerce, \
an online store selling authentic Indian devotional and ritual products.

STRICT RULES — follow these without exception:
1. You may ONLY recommend or mention products that appear in the CONTEXT section below.
2. Do NOT invent, guess, or hallucinate any product names, prices, or descriptions.
3. If no products in the context match the user's question, say:
   "I'm sorry, we don't currently carry anything that matches your request. \
Please browse our full collection on the homepage."
4. Always mention the exact product name and price (as shown in the context) when recommending.
5. Be warm, respectful, and concise — like a knowledgeable store assistant.
6. Do not answer questions unrelated to the store or its products.
7. If the user asks to add an item to their cart, buy an item, or purchase it, use the `add_to_cart_tool` with the correct `product_id` from the CONTEXT.

CONTEXT (retrieved products from our catalog):
{context}"""


# ─── Public API ───────────────────────────────────────────────────────────────
def ask(question: str, chat_history: list[dict] | None = None) -> dict:
    """
    Run the RAG pipeline.

    Args:
        question:     The user's question string.
        chat_history: List of dicts [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        {
            "answer": str,
            "sources": [{"product_id", "name", "price", "category", "url"}, ...]
        }
    """
    chat_history = chat_history or []

    # 1. Retrieve relevant products from Pinecone, with Supabase catalog fallback.
    docs = _retrieve_docs(question)

    # 2. Build a fast ID -> metadata lookup from retrieved docs
    doc_lookup: dict[int, dict] = {}
    for doc in docs:
        pid = doc.metadata.get("product_id")
        if pid is not None:
            doc_lookup[int(pid)] = doc.metadata

    context = _format_docs(docs)

    # 3. Build prompt (plain text, no tool calling)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    lc_history = []
    for msg in chat_history:
        if msg.get("role") == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))

    invoke_input = {
        "context": context,
        "input": question,
        "chat_history": lc_history,
    }

    # 4. Call LLM (using create_react_agent to support tools)
    llm = _get_llm()
    agent = create_react_agent(llm, tools=[add_to_cart_tool], state_modifier=prompt)

    # create_react_agent manages the prompt template internally using the state_modifier,
    # but we can pass our pre-rendered messages instead.

    # Actually, simpler: prompt template evaluates to messages. Let's just use it:
    messages = prompt.invoke(invoke_input).to_messages()

    result = agent.invoke({"messages": messages})

    # The last message is the AI's final answer
    raw_answer = result["messages"][-1].content if result.get("messages") else ""

    # 5. Clean up answer (strip JSON leaks) and extract citations
    answer = _clean_answer(raw_answer)

    # Try to get cited IDs from structured JSON in the response
    cited_ids = _extract_cited_ids(raw_answer)
    # Filter to only IDs that were actually in the retrieved docs
    cited_ids = [pid for pid in cited_ids if pid in doc_lookup]

    # If no explicit citations, heuristic: match product names in the answer
    if not cited_ids:
        cited_ids = [
            pid for pid, meta in doc_lookup.items()
            if meta.get("name", "").lower() in answer.lower()
        ]

    # 6. Build sources from ONLY the IDs the LLM chose to cite
    sources = []
    for pid in cited_ids:
        meta = doc_lookup.get(pid, {})
        sources.append({
            "product_id": pid,
            "name": meta.get("name", ""),
            "price": meta.get("price", 0),
            "category": meta.get("category", ""),
            "url": meta.get("url", f"/product/{pid}"),
        })

    return {"answer": answer, "sources": sources}


def ask_stream(question: str, chat_history: list[dict] | None = None):
    """
    Streaming version of ask(). Yields dicts:
      {"token": str}     — for each LLM token
      {"done": True, "sources": [...], "full_answer": str} — final event
    """
    chat_history = chat_history or []

    # 1. Retrieve relevant products from Pinecone, with Supabase catalog fallback.
    docs = _retrieve_docs(question)

    # 2. Build a fast ID -> metadata lookup from retrieved docs
    doc_lookup: dict[int, dict] = {}
    for doc in docs:
        pid = doc.metadata.get("product_id")
        if pid is not None:
            doc_lookup[int(pid)] = doc.metadata

    context = _format_docs(docs)

    # 3. Build prompt (plain text, no tool calling)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    lc_history = []
    for msg in chat_history:
        if msg.get("role") == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))

    invoke_input = {
        "context": context,
        "input": question,
        "chat_history": lc_history,
    }

    # 4. Stream LLM tokens
    llm = _get_llm()
    agent = create_react_agent(llm, tools=[add_to_cart_tool])
    messages = prompt.invoke(invoke_input).to_messages()

    full_raw = ""
    for msg, metadata in agent.stream({"messages": messages}, stream_mode="messages"):
        # We only stream the final assistant message chunks back to the user, not tool calls
        if isinstance(msg, AIMessageChunk) and msg.content:
            token = msg.content
            if isinstance(token, list):
                # Sometimes content is a list of dicts (e.g. Claude)
                token_str = "".join([t.get("text", "") for t in token if isinstance(t, dict) and "text" in t])
                if not token_str and isinstance(token, str):
                     token_str = token
                token = token_str
            if token:
                full_raw += token
                yield {"token": token}


    # 5. Compute sources from accumulated answer
    answer = _clean_answer(full_raw)
    cited_ids = _extract_cited_ids(full_raw)
    cited_ids = [pid for pid in cited_ids if pid in doc_lookup]

    if not cited_ids:
        cited_ids = [
            pid for pid, meta in doc_lookup.items()
            if meta.get("name", "").lower() in answer.lower()
        ]

    sources = []
    for pid in cited_ids:
        meta = doc_lookup.get(pid, {})
        sources.append({
            "product_id": pid,
            "name": meta.get("name", ""),
            "price": meta.get("price", 0),
            "category": meta.get("category", ""),
            "url": meta.get("url", f"/product/{pid}"),
        })

    yield {"done": True, "sources": sources, "full_answer": answer}
