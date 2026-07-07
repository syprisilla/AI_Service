from __future__ import annotations

from typing import Any

from . import _runtime
from .data_sources.local_db import PLACE_DB_PATH, sync_place_db
from .memory import remember_turn, used_place_names_from_memory
from .middleware import normalize_session_id, validate_and_normalize
from .output import output_parser
from .state import AgentGraphState
from .tools.intent import llm_intent_tool
from .tools.planner import append_missing_keyword_warnings, llm_route_planner_tool
from .tools.retrieval import candidate_lookup_tool, retrieve_place_documents
from .tools.review import tavily_review_boost_tool
from .tools.route import (
    distance_tool,
    enforce_day_trip_constraints,
    recommendation_tool,
)


BASE_DIR = _runtime.BASE_DIR
GRAPH_MEMORY = _runtime.GRAPH_MEMORY
END = _runtime.END
StateGraph = _runtime.StateGraph


def validate_input_node(graph_state: AgentGraphState) -> AgentGraphState:
    state, errors = validate_and_normalize(graph_state["payload"])
    if errors:
        return {"errors": errors, "status": 400, "result": {"errors": errors}}
    assert state is not None
    return {"state": state, "errors": [], "status": 200}


def analyze_intent_node(graph_state: AgentGraphState) -> AgentGraphState:
    state = graph_state["state"]
    tags, intent, intent_mode = llm_intent_tool(state)
    state["tags"] = tags
    return {"state": state, "tags": tags, "intent": intent, "intent_mode": intent_mode}


def retrieve_places_node(graph_state: AgentGraphState) -> AgentGraphState:
    state = graph_state["state"]
    tags = graph_state.get("tags", state["tags"])
    intent = graph_state.get("intent", {})
    try:
        slots, candidates = candidate_lookup_tool(state, tags, intent)
    except RuntimeError as error:
        errors = [
            str(error),
            (
                "TourAPI를 사용하려면 TOUR_API_KEY 또는 TOURAPI_SERVICE_KEY 환경변수를 설정한 뒤 "
                "/api/places/sync를 호출하세요."
            ),
        ]
        return {"errors": errors, "status": 503, "result": {"errors": errors}}

    candidates, tool_events = tavily_review_boost_tool(state, intent, candidates)
    return {
        "slots": slots,
        "candidates": candidates,
        "retrieved_documents": retrieve_place_documents(candidates),
        "tool_events": tool_events,
    }


def plan_route_node(graph_state: AgentGraphState) -> AgentGraphState:
    state = graph_state["state"]
    recommended, planner_mode, planner_decision = llm_route_planner_tool(
        state,
        graph_state.get("tags", state["tags"]),
        graph_state.get("intent", {}),
        graph_state.get("slots", []),
        graph_state.get("candidates", []),
        graph_state.get("retrieved_documents", []),
    )
    return {
        "recommended": recommended,
        "planner_mode": planner_mode,
        "planner_decision": planner_decision,
    }


def fallback_route_node(graph_state: AgentGraphState) -> AgentGraphState:
    state = graph_state["state"]
    excluded_place_names = used_place_names_from_memory(state.get("memory_context", []))
    excluded_place_names.discard(state["start_name"])
    recommended = recommendation_tool(
        graph_state.get("tags", state["tags"]),
        state["budget"],
        state["weather"],
        state["duration"],
        state["start_point"],
        state["transport"],
        excluded_place_names,
        graph_state.get("intent", {}),
    )
    return {
        "recommended": recommended,
        "planner_mode": graph_state.get("planner_mode") or "규칙 기반 Fallback",
        "planner_decision": {
            "decision_summary": "LLM 동선 선택을 사용할 수 없어 규칙 기반 추천으로 대체했습니다.",
        },
    }


def final_response_node(graph_state: AgentGraphState) -> AgentGraphState:
    if graph_state.get("result") and graph_state.get("status", 200) != 200:
        return graph_state

    state = graph_state["state"]
    recommended = graph_state.get("recommended", [])
    if not recommended:
        errors = [
            "로컬 JSON DB에 추천 가능한 청주 장소가 없습니다.",
            "/api/places/sync로 장소 데이터를 다시 수집해 주세요.",
        ]
        return {"errors": errors, "status": 503, "result": {"errors": errors}}

    planner_mode = graph_state.get("planner_mode", "규칙 기반 추천")
    if "Fallback" in planner_mode:
        state["warnings"].append(planner_mode)
    else:
        state["warnings"].append(f"{graph_state.get('intent_mode', '규칙 기반 의도 해석')} / {planner_mode}")

    constrained = enforce_day_trip_constraints(
        state,
        recommended,
        graph_state.get("candidates", []),
        graph_state.get("intent", {}),
    )
    if constrained != recommended:
        state["warnings"].append("당일치기 제약에 맞춰 예산 초과 또는 도보 15분 초과 후보를 제외했습니다.")
    recommended = constrained
    if not recommended:
        errors = [
            "예산과 이동 조건 안에서 추천 가능한 장소를 찾지 못했습니다.",
            "예산을 조금 늘리거나 여행 스타일을 넓게 입력해 주세요.",
        ]
        return {"errors": errors, "status": 422, "result": {"errors": errors}}
    append_missing_keyword_warnings(state, recommended)

    accommodation = None
    legs = distance_tool(state["start_point"], recommended, state["transport"], state["duration"], accommodation)
    result = output_parser(
        state,
        recommended,
        legs,
        accommodation,
        planner_mode=planner_mode,
        intent=graph_state.get("intent", {}),
        planner_decision=graph_state.get("planner_decision", {}),
        rag_document_count=len(graph_state.get("retrieved_documents", [])),
        tool_events=graph_state.get("tool_events", []),
    )
    remember_turn(state["session_id"], graph_state["payload"], result)
    return {"state": state, "accommodation": accommodation, "legs": legs, "result": result, "status": 200}


def route_after_validate(graph_state: AgentGraphState) -> str:
    return "final_response" if graph_state.get("errors") else "analyze_intent"


def route_after_retrieve(graph_state: AgentGraphState) -> str:
    return "final_response" if graph_state.get("errors") else "plan_route"


def route_after_plan(graph_state: AgentGraphState) -> str:
    return "fallback_route" if not graph_state.get("recommended") else "final_response"


def build_agent_graph() -> Any:
    if StateGraph is None:
        return None
    graph_builder = StateGraph(AgentGraphState)
    graph_builder.add_node("validate_input", validate_input_node)
    graph_builder.add_node("analyze_intent", analyze_intent_node)
    graph_builder.add_node("retrieve_places", retrieve_places_node)
    graph_builder.add_node("plan_route", plan_route_node)
    graph_builder.add_node("fallback_route", fallback_route_node)
    graph_builder.add_node("final_response", final_response_node)
    graph_builder.set_entry_point("validate_input")
    graph_builder.add_conditional_edges(
        "validate_input",
        route_after_validate,
        {"analyze_intent": "analyze_intent", "final_response": "final_response"},
    )
    graph_builder.add_edge("analyze_intent", "retrieve_places")
    graph_builder.add_conditional_edges(
        "retrieve_places",
        route_after_retrieve,
        {"plan_route": "plan_route", "final_response": "final_response"},
    )
    graph_builder.add_conditional_edges(
        "plan_route",
        route_after_plan,
        {"fallback_route": "fallback_route", "final_response": "final_response"},
    )
    graph_builder.add_edge("fallback_route", "final_response")
    graph_builder.add_edge("final_response", END)
    return graph_builder.compile(checkpointer=GRAPH_MEMORY) if GRAPH_MEMORY else graph_builder.compile()


AGENT_GRAPH = build_agent_graph()


def run_agent(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    initial_state: AgentGraphState = {"payload": payload}
    if AGENT_GRAPH is None:
        graph_state = dict(initial_state)
        graph_state.update(validate_input_node(graph_state))
        if graph_state.get("errors"):
            return graph_state["result"], graph_state.get("status", 400)
        graph_state.update(analyze_intent_node(graph_state))
        graph_state.update(retrieve_places_node(graph_state))
        if graph_state.get("errors"):
            return graph_state["result"], graph_state.get("status", 503)
        graph_state.update(plan_route_node(graph_state))
        if not graph_state.get("recommended"):
            graph_state.update(fallback_route_node(graph_state))
        graph_state.update(final_response_node(graph_state))
    else:
        session_id = normalize_session_id(payload.get("session_id"))
        config = {"configurable": {"thread_id": session_id}}
        graph_state = AGENT_GRAPH.invoke(initial_state, config=config)
    return graph_state.get("result", {"errors": ["Agent 실행 결과가 비어 있습니다."]}), graph_state.get("status", 500)
