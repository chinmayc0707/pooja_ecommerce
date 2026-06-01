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
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

load_dotenv()


# ─── Structured output schema ─────────────────────────────────────────────────
class RAGResponse(BaseModel):
    """Structured response the LLM must produce."""
    answer: str = Field(
        description="Your helpful, warm response to the user's question."
    )
    cited_product_ids: List[int] = Field(
        default_factory=list,
        description=(
            "List of product IDs (integers) that you explicitly recommended or "
            "mentioned by name in your answer. "
            "ONLY include IDs of products from the provided context that you "
            "actually referenced. Leave empty if no products matched the question. When mentioning products use tables for better UX"
        )
    )


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
def _format_docs(docs: List[Document]) -> str:
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
    Safety net: if the LLM returned raw JSON instead of plain text
    (e.g. fallback path with a model that outputs structured JSON as text),
    extract just the 'answer' field so the user never sees raw JSON.
    """
    import json, re
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

CITATION RULE (important):
After forming your answer, populate `cited_product_ids` with ONLY the integer IDs \
of products you actually mentioned or recommended in your answer text. \
If you recommended two products, include exactly those two IDs. \
Do not include IDs of products you retrieved but did not mention.

CONTEXT (retrieved products from our catalog):
{context}"""


# ─── Public API ───────────────────────────────────────────────────────────────
def ask(question: str, chat_history: list[dict] | None = None) -> dict:
    """
    Run the agentic RAG pipeline.

    Args:
        question:     The user's question string.
        chat_history: List of dicts [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        {
            "answer": str,
            "sources": [{"product_id", "name", "price", "category", "url"}, ...]
            — only products the LLM explicitly chose to cite.
        }
    """
    chat_history = chat_history or []

    # 1. Retrieve semantically relevant products
    vectorstore = _get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )
    docs = retriever.invoke(question)

    # 2. Build a fast ID → metadata lookup from retrieved docs
    doc_lookup: dict[int, dict] = {}
    for doc in docs:
        pid = doc.metadata.get("product_id")
        if pid is not None:
            doc_lookup[int(pid)] = doc.metadata

    context = _format_docs(docs)

    # 3. Build prompt
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

    # 4. Try structured output (tool calling) so LLM picks its own citations
    llm = _get_llm()
    cited_ids: list[int] = []
    answer: str = ""

    try:
        structured_llm = llm.with_structured_output(RAGResponse)
        chain = prompt | structured_llm
        result: RAGResponse = chain.invoke(invoke_input)

        answer = result.answer
        # Filter to only IDs that were actually in the retrieved docs
        cited_ids = [
            int(pid) for pid in result.cited_product_ids
            if int(pid) in doc_lookup
        ]

    except Exception:
        # Fallback for models that don't support tool calling (e.g. some OpenRouter free tiers)
        plain_chain = prompt | llm
        plain_result = plain_chain.invoke(invoke_input)
        raw_answer = (
            plain_result.content
            if hasattr(plain_result, "content")
            else str(plain_result)
        )
        # Strip raw JSON leaks before sending to frontend
        answer = _clean_answer(raw_answer)
        # Heuristic fallback: include products whose name appears in the answer
        cited_ids = [
            pid for pid, meta in doc_lookup.items()
            if meta.get("name", "").lower() in answer.lower()
        ]

    # 5. Build sources from ONLY the IDs the LLM chose to cite
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
