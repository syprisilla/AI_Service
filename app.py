from __future__ import annotations

import math
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from flask import Flask, jsonify, render_template, request


app = Flask(__name__)


PLACE_DB = {
    "청주": [
        {
            "name": "수암골",
            "category": "관광지",
            "lat": 36.6486,
            "lng": 127.4925,
            "tags": ["사진", "산책", "카페", "야경"],
            "score": 4.6,
            "cost": 0,
            "indoor": False,
            "stay_minutes": 80,
        },
        {
            "name": "청주 성안길",
            "category": "상권",
            "lat": 36.6355,
            "lng": 127.4893,
            "tags": ["맛집", "쇼핑", "카페", "실내"],
            "score": 4.5,
            "cost": 10000,
            "indoor": True,
            "stay_minutes": 90,
        },
        {
            "name": "청남대",
            "category": "관광지",
            "lat": 36.4626,
            "lng": 127.4906,
            "tags": ["자연", "산책", "역사", "사진"],
            "score": 4.7,
            "cost": 6000,
            "indoor": False,
            "stay_minutes": 150,
        },
        {
            "name": "상당산성",
            "category": "관광지",
            "lat": 36.6617,
            "lng": 127.5384,
            "tags": ["역사", "산책", "자연", "사진"],
            "score": 4.6,
            "cost": 0,
            "indoor": False,
            "stay_minutes": 120,
        },
        {
            "name": "국립청주박물관",
            "category": "박물관",
            "lat": 36.6497,
            "lng": 127.5120,
            "tags": ["역사", "실내", "전시", "비오는날"],
            "score": 4.4,
            "cost": 0,
            "indoor": True,
            "stay_minutes": 90,
        },
        {
            "name": "운리단길",
            "category": "카페거리",
            "lat": 36.6376,
            "lng": 127.4604,
            "tags": ["카페", "맛집", "사진", "데이트"],
            "score": 4.4,
            "cost": 8000,
            "indoor": True,
            "stay_minutes": 80,
        },
        {
            "name": "무심천",
            "category": "산책",
            "lat": 36.6383,
            "lng": 127.4747,
            "tags": ["산책", "자연", "벚꽃", "무료"],
            "score": 4.3,
            "cost": 0,
            "indoor": False,
            "stay_minutes": 60,
        },
        {
            "name": "청주고인쇄박물관",
            "category": "박물관",
            "lat": 36.6461,
            "lng": 127.4716,
            "tags": ["역사", "실내", "전시", "직지"],
            "score": 4.3,
            "cost": 0,
            "indoor": True,
            "stay_minutes": 80,
        },
        {
            "name": "청주 육거리시장",
            "category": "시장",
            "lat": 36.6295,
            "lng": 127.4892,
            "tags": ["맛집", "시장", "먹거리", "로컬"],
            "score": 4.4,
            "cost": 10000,
            "indoor": False,
            "stay_minutes": 70,
        },
        {
            "name": "오송호수공원",
            "category": "공원",
            "lat": 36.6286,
            "lng": 127.3297,
            "tags": ["산책", "자연", "사진", "무료"],
            "score": 4.2,
            "cost": 0,
            "indoor": False,
            "stay_minutes": 70,
        },
    ]
}

START_POINTS = {
    "청주고속버스터미널": {"lat": 36.6260, "lng": 127.4317},
    "청주역": {"lat": 36.6487, "lng": 127.3927},
    "충북대": {"lat": 36.6283, "lng": 127.4565},
    "오송역": {"lat": 36.6200, "lng": 127.3275},
}

STYLE_KEYWORDS = {
    "카페": ["카페", "커피", "디저트", "감성"],
    "맛집": ["맛집", "밥", "식사", "먹거리", "시장", "로컬"],
    "산책": ["산책", "걷", "걷기", "느긋", "힐링"],
    "사진": ["사진", "포토", "인생샷", "야경", "감성"],
    "역사": ["역사", "박물관", "전시", "직지", "문화"],
    "실내": ["실내", "비", "전시", "박물관"],
    "자연": ["자연", "공원", "호수", "산성", "벚꽃"],
    "쇼핑": ["쇼핑", "상권", "거리"],
}

TRANSPORT_SPEED_KMH = {
    "도보 중심": 4,
    "대중교통": 18,
    "자동차": 35,
}

ACCOMMODATION_DB = [
    {
        "name": "성안길 비즈니스 호텔",
        "category": "숙소",
        "lat": 36.6358,
        "lng": 127.4888,
        "tags": ["실내", "교통", "가성비", "중심가"],
        "score": 4.3,
        "cost": 30000,
        "indoor": True,
    },
    {
        "name": "충북대 인근 게스트하우스",
        "category": "숙소",
        "lat": 36.6292,
        "lng": 127.4560,
        "tags": ["실내", "가성비", "대중교통", "조용함"],
        "score": 4.1,
        "cost": 25000,
        "indoor": True,
    },
    {
        "name": "오송역 스테이",
        "category": "숙소",
        "lat": 36.6208,
        "lng": 127.3290,
        "tags": ["실내", "교통", "역세권"],
        "score": 4.0,
        "cost": 35000,
        "indoor": True,
    },
]


@dataclass
class AgentState:
    start_name: str
    start_point: dict[str, float]
    duration: str
    style_text: str
    transport: str
    budget: int
    weather: str
    tags: list[str]
    warnings: list[str]


def remove_private_info(text: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    sanitized = re.sub(r"01[016789]-?\d{3,4}-?\d{4}", "[전화번호 제거]", text)
    sanitized = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[이메일 제거]", sanitized)
    if sanitized != text:
        warnings.append("개인정보 제거 Middleware가 전화번호 또는 이메일 형식을 제거했습니다.")
    return sanitized, warnings


def validate_and_normalize(payload: dict[str, Any]) -> tuple[AgentState | None, list[str]]:
    warnings: list[str] = []
    style_text, privacy_warnings = remove_private_info(str(payload.get("style_text", "")).strip())
    warnings.extend(privacy_warnings)

    start_name = str(payload.get("start_name", "")).strip()
    if start_name not in START_POINTS:
        warnings.append(
            "Fallback Middleware: 입력한 출발지를 찾을 수 없어 청주고속버스터미널로 계산했습니다."
        )
        start_name = "청주고속버스터미널"

    duration = str(payload.get("duration", "당일치기")).strip()
    transport = str(payload.get("transport", "대중교통")).strip()
    weather = str(payload.get("weather", "맑음")).strip()

    if not style_text:
        return None, ["여행 스타일을 하나 이상 입력해주세요."]

    try:
        budget = int(str(payload.get("budget", "0")).replace(",", "").strip())
    except ValueError:
        return None, ["예산은 숫자로 입력해주세요."]

    if budget <= 0:
        return None, ["예산은 1원 이상으로 입력해주세요."]

    if transport not in TRANSPORT_SPEED_KMH:
        transport = "대중교통"
        warnings.append("이동수단 값이 올바르지 않아 대중교통 기준으로 계산했습니다.")

    if duration not in ["당일치기", "1박 2일"]:
        duration = "당일치기"
        warnings.append("여행 기간 값이 올바르지 않아 당일치기 기준으로 계산했습니다.")

    state = AgentState(
        start_name=start_name,
        start_point=START_POINTS[start_name],
        duration=duration,
        style_text=style_text,
        transport=transport,
        budget=budget,
        weather=weather,
        tags=[],
        warnings=warnings,
    )
    return state, []


def style_analysis_tool(style_text: str) -> list[str]:
    normalized = style_text.replace(",", " ")
    tags = []
    for tag, keywords in STYLE_KEYWORDS.items():
        if tag in normalized or any(keyword in normalized for keyword in keywords):
            tags.append(tag)
    return tags or ["카페", "맛집", "사진"]


def weather_filter_score(place: dict[str, Any], weather: str) -> float:
    if weather == "비":
        return 2.4 if place["indoor"] else -3.0
    if weather == "더움":
        return 1.0 if place["indoor"] else -0.3
    if weather == "추움":
        return 1.2 if place["indoor"] else -0.5
    return 0.4 if not place["indoor"] else 0.0


def indoor_preference_score(place: dict[str, Any], tags: list[str]) -> float:
    if "실내" not in tags:
        return 0
    return 3.0 if place["indoor"] else -5.0


def recommendation_tool(tags: list[str], budget: int, weather: str, duration: str) -> list[dict[str, Any]]:
    places = []
    target_count = 6 if duration == "1박 2일" else 5
    per_place_budget = budget / target_count
    indoor_first = "실내" in tags or weather == "비"

    for place in PLACE_DB["청주"]:
        matched_tags = sorted(set(tags).intersection(place["tags"]))
        budget_penalty = 1.0 if place["cost"] > per_place_budget and place["cost"] > 0 else 0
        score = (
            place["score"]
            + len(matched_tags) * 2
            + weather_filter_score(place, weather)
            + indoor_preference_score(place, tags)
            - budget_penalty
        )
        places.append({**place, "matched_tags": matched_tags, "agent_score": round(score, 2)})

    ranked_places = sorted(places, key=lambda item: item["agent_score"], reverse=True)
    if indoor_first:
        indoor_places = [place for place in ranked_places if place["indoor"]]
        selected = indoor_places[:target_count]
    else:
        selected = ranked_places[:target_count]
    return balance_categories(selected, places, target_count, indoor_first)


def balance_categories(
    selected: list[dict[str, Any]],
    all_places: list[dict[str, Any]],
    target_count: int,
    indoor_first: bool = False,
) -> list[dict[str, Any]]:
    if indoor_first:
        return selected[:target_count]

    has_food = any(place["category"] in ["시장", "상권", "카페거리"] for place in selected)
    has_culture = any(place["category"] in ["관광지", "박물관"] for place in selected)
    has_walk = any("산책" in place["tags"] or place["category"] == "공원" for place in selected)

    required_categories = []
    if not has_food:
        required_categories.append(["시장", "상권", "카페거리"])
    if not has_culture:
        required_categories.append(["관광지", "박물관"])
    if not has_walk:
        required_categories.append(["산책", "공원"])

    names = {place["name"] for place in selected}
    for categories in required_categories:
        candidate = next(
            (place for place in sorted(all_places, key=lambda item: item["agent_score"], reverse=True)
             if place["category"] in categories and place["name"] not in names),
            None,
        )
        if candidate:
            selected[-1] = candidate
            names.add(candidate["name"])

    return sorted(selected[:target_count], key=lambda item: item["agent_score"], reverse=True)


def haversine_km(a: dict[str, float], b: dict[str, float]) -> float:
    radius = 6371
    lat1, lon1 = math.radians(a["lat"]), math.radians(a["lng"])
    lat2, lon2 = math.radians(b["lat"]), math.radians(b["lng"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def optimize_route_tool(start_point: dict[str, float], places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    route = []
    remaining = places[:]
    current = start_point

    while remaining:
        next_place = min(remaining, key=lambda place: haversine_km(current, place))
        route.append(next_place)
        remaining.remove(next_place)
        current = next_place

    return route


def distance_tool(start_point: dict[str, float], route: list[dict[str, Any]], transport: str) -> list[dict[str, Any]]:
    speed = TRANSPORT_SPEED_KMH[transport]
    legs = []
    current_name = "출발지"
    current = start_point

    for place in route:
        distance = haversine_km(current, place)
        move_minutes = max(5, round(distance / speed * 60))
        legs.append(
            {
                "from": current_name,
                "to": place["name"],
                "distance_km": round(distance, 2),
                "move_minutes": move_minutes,
            }
        )
        current_name = place["name"]
        current = place

    return legs


def accommodation_tool(
    state: AgentState,
    route: list[dict[str, Any]],
    place_cost: int,
) -> dict[str, Any] | None:
    if state.duration != "1박 2일":
        return None

    day1_last_place = route[max(0, len(route) // 2 - 1)] if route else state.start_point
    remaining_budget = max(0, state.budget - place_cost)

    candidates = []
    for accommodation in ACCOMMODATION_DB:
        distance = haversine_km(day1_last_place, accommodation)
        budget_penalty = 3 if accommodation["cost"] > remaining_budget else 0
        transport_bonus = 1.0 if state.transport in accommodation["tags"] or "교통" in accommodation["tags"] else 0
        score = accommodation["score"] + transport_bonus - distance * 0.25 - budget_penalty
        candidates.append(
            {
                **accommodation,
                "distance_from_day1_km": round(distance, 2),
                "agent_score": round(score, 2),
                "matched_tags": sorted(set(state.tags).intersection(accommodation["tags"])),
            }
        )

    return max(candidates, key=lambda item: item["agent_score"])


def build_schedule(
    route: list[dict[str, Any]],
    legs: list[dict[str, Any]],
    duration: str,
    accommodation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    start_hour = 10
    current_minutes = start_hour * 60
    schedule = []
    split_index = math.ceil(len(route) / 2) if duration == "1박 2일" else len(route)

    for index, place in enumerate(route):
        if duration == "1박 2일" and index == split_index:
            current_minutes = 10 * 60
        current_minutes += legs[index]["move_minutes"]
        start = format_time(current_minutes)
        current_minutes += place["stay_minutes"]
        end = format_time(current_minutes)
        schedule.append(
            {
                "day": 2 if duration == "1박 2일" and index >= split_index else 1,
                "time": f"{start} - {end}",
                "place": place["name"],
                "category": place["category"],
                "reason": build_reason(place),
                "cost": place["cost"],
                "indoor": place["indoor"],
            }
        )

        if duration == "1박 2일" and accommodation and index == split_index - 1:
            schedule.append(
                {
                    "day": 1,
                    "time": "18:00 - 다음날 10:00",
                    "place": accommodation["name"],
                    "category": "숙소",
                    "reason": (
                        f"Day 1 마지막 장소와 약 {accommodation['distance_from_day1_km']}km 거리이고, "
                        f"남은 예산을 고려해 선택한 1박 숙소입니다."
                    ),
                    "cost": accommodation["cost"],
                    "indoor": True,
                }
            )

    return schedule


def format_time(total_minutes: int) -> str:
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def build_reason(place: dict[str, Any]) -> str:
    matched = ", ".join(place["matched_tags"]) if place["matched_tags"] else "균형 일정"
    weather_note = "실내" if place["indoor"] else "야외"
    return f"{matched} 선호와 맞고, {weather_note} 일정으로 활용하기 좋습니다."


def output_parser(state: AgentState, route: list[dict[str, Any]], legs: list[dict[str, Any]]) -> dict[str, Any]:
    place_cost = sum(place["cost"] for place in route)
    accommodation = accommodation_tool(state, route, place_cost)
    lodging_cost = accommodation["cost"] if accommodation else 0
    total_cost = place_cost + lodging_cost
    total_distance = round(sum(leg["distance_km"] for leg in legs), 2)
    total_move_minutes = sum(leg["move_minutes"] for leg in legs)
    finish = f"{accommodation['name']} 체크인" if accommodation else "출발지 복귀 또는 성안길 카페 마무리"
    ai_comment, ai_mode = llm_response_tool(
        state,
        route,
        legs,
        total_cost,
        total_distance,
        total_move_minutes,
        accommodation,
    )
    route_names = [state.start_name] + [place["name"] for place in route]
    if accommodation:
        split_index = math.ceil(len(route) / 2)
        route_names = (
            [state.start_name]
            + [place["name"] for place in route[:split_index]]
            + [f"{accommodation['name']}(숙소)"]
            + [place["name"] for place in route[split_index:]]
        )

    return {
        "summary": {
            "title": "Cheongju Trip Agent 추천 결과",
            "tags": state.tags,
            "route_text": " → ".join(route_names),
            "total_cost": total_cost,
            "budget": state.budget,
            "total_distance": total_distance,
            "total_move_minutes": total_move_minutes,
            "finish": finish,
            "weather": state.weather,
            "transport": state.transport,
            "ai_comment": ai_comment,
            "ai_mode": ai_mode,
            "lodging_cost": lodging_cost,
        },
        "schedule": build_schedule(route, legs, state.duration, accommodation),
        "accommodation": accommodation,
        "places": [
            {
                "name": place["name"],
                "category": place["category"],
                "score": place["agent_score"],
                "tags": place["tags"],
                "matched_tags": place["matched_tags"],
                "cost": place["cost"],
                "indoor": place["indoor"],
                "lat": place["lat"],
                "lng": place["lng"],
            }
            for place in route
        ],
        "legs": legs,
        "warnings": state.warnings,
        "agent_flow": [
            "입력 검증 Middleware",
            "개인정보 제거 Middleware",
            "Fallback Middleware",
            "여행 스타일 분석 Tool",
            "청주 장소 자동 추천 Tool",
            "날씨 대응 Tool",
            "거리 계산 Tool",
            "동선 최적화 Tool",
            "숙소 추천 Tool",
            "Context 생성",
            "LLM 응답 생성 Tool",
            "Output Parser",
        ],
    }


def build_llm_context(
    state: AgentState,
    route: list[dict[str, Any]],
    legs: list[dict[str, Any]],
    total_cost: int,
    total_distance: float,
    total_move_minutes: int,
    accommodation: dict[str, Any] | None = None,
) -> str:
    places = [
        {
            "name": place["name"],
            "category": place["category"],
            "matched_tags": place["matched_tags"],
            "cost": place["cost"],
            "indoor": place["indoor"],
            "agent_score": place["agent_score"],
        }
        for place in route
    ]
    context = {
        "user_request": {
            "start": state.start_name,
            "duration": state.duration,
            "style_text": state.style_text,
            "tags": state.tags,
            "transport": state.transport,
            "budget": state.budget,
            "weather": state.weather,
        },
        "optimized_route": [place["name"] for place in route],
        "places": places,
        "move_legs": legs,
        "total_cost": total_cost,
        "total_distance_km": total_distance,
        "total_move_minutes": total_move_minutes,
        "accommodation": accommodation,
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def extract_openai_text(data: dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"]).strip()

    text_parts: list[str] = []
    for output in data.get("output", []):
        for content in output.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                text_parts.append(str(content["text"]))
    return "\n".join(text_parts).strip()


def llm_response_tool(
    state: AgentState,
    route: list[dict[str, Any]],
    legs: list[dict[str, Any]],
    total_cost: int,
    total_distance: float,
    total_move_minutes: int,
    accommodation: dict[str, Any] | None = None,
) -> tuple[str, str]:
    context = build_llm_context(
        state,
        route,
        legs,
        total_cost,
        total_distance,
        total_move_minutes,
        accommodation,
    )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "현재 실행 환경에는 OPENAI_API_KEY가 없어 규칙 기반 Fallback으로 최종 문장을 생성했습니다. "
            "추천 결과는 Tool들이 만든 Context를 기반으로 구성되었습니다.",
            "규칙 기반 Fallback",
        )

    prompt = (
        "너는 청주 여행 추천 Agent의 최종 응답 생성기다. "
        "아래 JSON Context만 근거로 사용해서 한국어로 3문장 이내의 짧은 추천 코멘트를 작성해라. "
        "예산, 날씨, 동선 최적화 이유를 자연스럽게 포함해라.\n\n"
        f"{context}"
    )
    payload = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "input": prompt,
        "temperature": 0.4,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            data = json.loads(response.read().decode("utf-8"))
            llm_text = extract_openai_text(data)
            if llm_text:
                return llm_text, "OpenAI LLM"
            return (
                "OpenAI 호출은 성공했지만 응답 텍스트가 비어 있어 규칙 기반 Fallback 문장을 사용했습니다.",
                "LLM 응답 없음 Fallback",
            )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
        return (
            f"LLM 호출이 실패해 규칙 기반 Fallback으로 응답했습니다. 실패 사유: {error}",
            "LLM 실패 Fallback",
        )


def run_agent(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    state, errors = validate_and_normalize(payload)
    if errors:
        return {"errors": errors}, 400

    assert state is not None
    state.tags = style_analysis_tool(state.style_text)
    recommended = recommendation_tool(state.tags, state.budget, state.weather, state.duration)
    route = optimize_route_tool(state.start_point, recommended)
    legs = distance_tool(state.start_point, route, state.transport)
    return output_parser(state, route, legs), 200


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/recommend")
def recommend():
    result, status = run_agent(request.get_json(force=True))
    return jsonify(result), status


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
