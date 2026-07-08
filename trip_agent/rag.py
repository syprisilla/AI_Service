from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from . import _runtime

logger = logging.getLogger(__name__)

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - optional dependency fallback
    Document = None

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # pragma: no cover - optional dependency fallback
    OpenAIEmbeddings = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma, FAISS
except ImportError:  # pragma: no cover - optional dependency fallback
    HuggingFaceEmbeddings = None
    Chroma = None
    FAISS = None


VECTORSTORE_DIR = _runtime.DATA_DIR / "vectorstore"
_VECTORSTORE_CACHE: Any | None = None


def load_place_records(path: Path | None = None) -> list[dict[str, Any]]:
    db_path = path or _runtime.PLACE_DB_PATH
    payload = json.loads(db_path.read_text(encoding="utf-8"))
    places = payload.get("places", []) if isinstance(payload, dict) else []
    return [place for place in places if isinstance(place, dict)]


def place_to_document(place: dict[str, Any]) -> Any:
    page_content = (
        f"{place.get('name', '')}\n"
        f"category={place.get('category', '')}; role={place.get('role', '')}; "
        f"tags={', '.join(place.get('tags', []))}; "
        f"address={place.get('address', '')}; phone={place.get('phone', '')}; "
        f"source={place.get('source', '')}; score={place.get('score', '')}; "
        f"quality_score={place.get('quality_score', '')}"
    )
    metadata = {
        "name": place.get("name"),
        "category": place.get("category"),
        "role": place.get("role"),
        "tags": ",".join(place.get("tags", [])),
        "address": place.get("address"),
        "source": place.get("source"),
    }
    if Document:
        return Document(page_content=page_content, metadata=metadata)
    return {"page_content": page_content, "metadata": metadata}


def documents_to_context(documents: list[Any]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for document in documents:
        if Document and isinstance(document, Document):
            contexts.append({"page_content": document.page_content, "metadata": document.metadata})
        else:
            contexts.append(
                {
                    "page_content": document.get("page_content", ""),
                    "metadata": document.get("metadata", {}),
                }
            )
    return contexts


def build_embeddings() -> Any | None:
    if OpenAIEmbeddings and os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings()
    if HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(model_name=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    return None


def build_vectorstore(force_rebuild: bool = False) -> Any | None:
    global _VECTORSTORE_CACHE
    if _VECTORSTORE_CACHE is not None and not force_rebuild:
        return _VECTORSTORE_CACHE

    embeddings = build_embeddings()
    if embeddings is None:
        logger.warning("RAG vectorstore skipped: no embeddings backend available")
        return None

    places = load_place_records()
    documents = [place_to_document(place) for place in places]
    try:
        if FAISS:
            _VECTORSTORE_CACHE = FAISS.from_documents(documents, embeddings)
        elif Chroma:
            VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
            _VECTORSTORE_CACHE = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=str(VECTORSTORE_DIR),
            )
        else:
            logger.warning("RAG vectorstore skipped: FAISS/Chroma is not installed")
            return None
    except Exception as error:  # pragma: no cover - backend-specific failures
        logger.warning("RAG vectorstore build failed: %s", error)
        return None
    return _VECTORSTORE_CACHE


def keyword_retrieve(query: str, max_documents: int = 8) -> list[dict[str, Any]]:
    terms = [term for term in query.replace(",", " ").split() if term]
    scored: list[tuple[int, dict[str, Any]]] = []
    for place in load_place_records():
        text = " ".join(
            [
                str(place.get("name", "")),
                str(place.get("category", "")),
                str(place.get("role", "")),
                " ".join(place.get("tags", [])),
                str(place.get("address", "")),
            ]
        )
        score = sum(1 for term in terms if term in text)
        if score:
            scored.append((score, place))
    scored.sort(key=lambda item: (item[0], float(item[1].get("score", 0) or 0)), reverse=True)
    return documents_to_context([place_to_document(place) for _, place in scored[:max_documents]])


def retrieve_relevant_place_documents(query: str, max_documents: int = 8) -> list[dict[str, Any]]:
    vectorstore = build_vectorstore()
    if vectorstore is None:
        return keyword_retrieve(query, max_documents=max_documents)

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": max_documents})
        documents = retriever.invoke(query)
    except Exception as error:  # pragma: no cover - backend-specific failures
        logger.warning("RAG retriever failed: %s", error)
        return keyword_retrieve(query, max_documents=max_documents)
    return documents_to_context(documents)

