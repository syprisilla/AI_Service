from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLACE_DB_PATH = DATA_DIR / "places_cache.json"
VECTORSTORE_DIR = DATA_DIR / "faiss_places_index"

load_dotenv(BASE_DIR / ".env")


def load_place_items(path: Path = PLACE_DB_PATH) -> list[dict[str, Any]]:
    """places_cache.json에서 장소 목록을 읽는다."""
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        places = payload.get("places", [])
        return places if isinstance(places, list) else []

    return []


def place_to_document(place: dict[str, Any]) -> Document:
    """장소 1개를 LangChain Document로 변환한다."""
    tags = place.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    page_content = "\n".join(
        [
            f"장소명: {place.get('name', '')}",
            f"카테고리: {place.get('category', '')}",
            f"역할: {place.get('role', '')}",
            f"태그: {', '.join(map(str, tags))}",
            f"주소: {place.get('address', '')}",
            f"카카오 카테고리: {place.get('kakao_category', '')}",
            f"검색어: {place.get('search_query', '')}",
            f"출처: {place.get('source', '')}",
            f"전화번호 존재: {bool(place.get('phone'))}",
            f"지도 URL 존재: {bool(place.get('map_url') or place.get('url'))}",
            f"예상 비용: {place.get('cost', 0)}원",
            f"실내 여부: {place.get('indoor', False)}",
            f"체류 시간: {place.get('stay_minutes', '')}분",
            f"품질 점수: {place.get('quality_score', '')}",
        ]
    )

    return Document(
        page_content=page_content,
        metadata={
            "name": place.get("name"),
            "category": place.get("category"),
            "role": place.get("role"),
            "tags": ",".join(map(str, tags)),
            "address": place.get("address"),
            "source": place.get("source"),
            "lat": place.get("lat"),
            "lng": place.get("lng"),
            "cost": place.get("cost"),
            "indoor": place.get("indoor"),
            "quality_score": place.get("quality_score"),
        },
    )


def load_place_documents() -> list[Document]:
    """장소 DB 전체를 Document 리스트로 변환한다."""
    places = load_place_items()
    docs = []

    for place in places:
        name = str(place.get("name", "")).strip()
        if not name:
            continue
        docs.append(place_to_document(place))

    return docs


def get_embeddings() -> OpenAIEmbeddings:
    """OpenAI Embedding 모델을 만든다."""
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )


def rebuild_place_vectorstore() -> int:
    """
    places_cache.json을 읽어서 FAISS 벡터스토어를 새로 만든다.
    /api/places/sync 이후 호출하면 최신 장소 DB가 RAG에 반영된다.
    """
    docs = load_place_documents()

    if not docs:
        if VECTORSTORE_DIR.exists():
            shutil.rmtree(VECTORSTORE_DIR)
        return 0

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    return len(docs)


def build_or_load_vectorstore() -> FAISS | None:
    """
    저장된 FAISS 인덱스가 있으면 로드하고,
    없으면 places_cache.json으로 새로 만든다.
    """
    embeddings = get_embeddings()

    if VECTORSTORE_DIR.exists():
        try:
            return FAISS.load_local(
                str(VECTORSTORE_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            # 인덱스가 깨졌으면 재생성
            shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)

    docs = load_place_documents()
    if not docs:
        return None

    vectorstore = FAISS.from_documents(docs, embeddings)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    return vectorstore


def documents_to_context(docs: list[Document]) -> list[dict[str, Any]]:
    """LangChain Document를 기존 server.py가 쓰는 dict 형식으로 변환한다."""
    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


def retrieve_relevant_place_documents(query: str, max_documents: int = 8) -> list[dict[str, Any]]:
    """
    진짜 VectorStore RAG 검색.
    사용자 쿼리 → embedding → FAISS similarity search → 관련 Document 반환.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return []

    vectorstore = build_or_load_vectorstore()
    if vectorstore is None:
        return []

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": max_documents}
    )

    docs = retriever.invoke(query)
    return documents_to_context(docs)

def candidate_documents_to_context(
    candidates: list[dict[str, Any]],
    max_documents: int = 12,
) -> list[dict[str, Any]]:
    """후보 장소 리스트를 LLM Context용 Document dict로 변환한다."""
    docs = [place_to_document(place) for place in candidates[:max_documents]]
    return documents_to_context(docs)