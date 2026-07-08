from __future__ import annotations

import json
from typing import Any

from .. import _runtime
from ..rag import retrieve_relevant_place_documents
from .retrieval import candidate_lookup_tool
from .route import distance_tool

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover - optional dependency fallback
    def tool(func: Any) -> Any:
        return func


@tool
def search_place_candidates(query: str) -> str:
    """사용자 여행 요청에 맞는 청주 장소 후보를 검색한다."""
    tags, intent = _runtime.local_intent_analysis(query, [])
    state: _runtime.AgentState = {
        "session_id": "langchain-tool",
        "start_name": "충북대",
        "start_point": _runtime.START_POINTS["충북대"],
        "duration": "당일치기",
        "style_text": query,
        "selected_keywords": [],
        "transport": "대중교통",
        "budget": 50000,
        "weather": "맑음",
        "tags": tags,
        "warnings": [],
        "memory_context": [],
    }
    slots, candidates = candidate_lookup_tool(state, state["tags"], intent)
    rag_documents = retrieve_relevant_place_documents(query, max_documents=6)
    payload = {
        "slots": slots,
        "candidates": [
            {
                "name": place.get("name"),
                "category": place.get("category"),
                "role": place.get("role"),
                "score": place.get("agent_score"),
                "address": place.get("address"),
            }
            for place in candidates[:10]
        ],
        "rag_documents": rag_documents,
        "warnings": state["warnings"],
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def calculate_route_distance(route_json: str) -> str:
    """선택된 장소들의 거리와 이동 시간을 계산한다."""
    payload = json.loads(route_json)
    route = payload if isinstance(payload, list) else payload.get("route", [])
    start_name = payload.get("start_name", "충북대") if isinstance(payload, dict) else "충북대"
    transport = payload.get("transport", "대중교통") if isinstance(payload, dict) else "대중교통"
    duration = payload.get("duration", "당일치기") if isinstance(payload, dict) else "당일치기"
    start_point = _runtime.START_POINTS.get(start_name, _runtime.START_POINTS["충북대"])
    legs = distance_tool(start_point, route, transport, duration)
    return json.dumps({"legs": legs}, ensure_ascii=False)


LANGCHAIN_TOOLS = [search_place_candidates, calculate_route_distance]
