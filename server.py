from __future__ import annotations

import math
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from prompts import FINAL_COMMENT_PROMPT, INTENT_ANALYSIS_PROMPT, ROUTE_PLANNER_PROMPT

from rag import (
    retrieve_relevant_place_documents,
    rebuild_place_vectorstore,
    candidate_documents_to_context,
)

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None


class OpenAIProvider:
    name = "OpenAI LangChain"

    def _llm(self, temperature: float, timeout: int) -> Any:
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai 미설치")
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            timeout=timeout,
        )

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()
        return str(content).strip()

    def _prompt_chain(self, temperature: float, timeout: int, parser: Any | None = None) -> Any:
        if PromptTemplate is None:
            raise RuntimeError("langchain-core PromptTemplate 미설치")
        output_parser = parser or (StrOutputParser() if StrOutputParser else None)
        chain = PromptTemplate.from_template("{prompt}") | self._llm(temperature, timeout)
        return chain | output_parser if output_parser else chain

    def text_tool(self, prompt: str, temperature: float = 0.2, timeout: int = 18) -> tuple[str | None, str]:
        if not os.getenv("OPENAI_API_KEY"):
            return None, "OPENAI_API_KEY 없음"
        try:
            message = self._prompt_chain(temperature, timeout).invoke({"prompt": prompt})
        except Exception as error:
            return None, str(error)
        return self._message_text(message), "OpenAI LangChain Runnable"

    def json_tool(
        self,
        prompt: str,
        parser: Any | None = None,
        temperature: float = 0.2,
        timeout: int = 18,
    ) -> tuple[dict[str, Any] | None, str]:
        if not os.getenv("OPENAI_API_KEY"):
            return None, "OPENAI_API_KEY 없음"
        try:
            parsed = self._prompt_chain(temperature, timeout, parser).invoke({"prompt": prompt})
            return parsed.model_dump() if hasattr(parsed, "model_dump") else parsed, "OpenAI LangChain Runnable"
        except (ValidationError, ValueError, Exception) as error:
            return None, f"LangChain OutputParser 실패: {error}"

    def final_comment(self, prompt: str, temperature: float = 0.4, timeout: int = 12) -> tuple[str | None, str]:
        return self.text_tool(prompt, temperature, timeout)


def get_model_provider(name: str | None = None) -> OpenAIProvider:
    return OpenAIProvider()

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field, ValidationError

try:
    from langchain_core.documents import Document
    from langchain_core.messages import SystemMessage
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent
except ImportError:
    Document = None
    SystemMessage = None
    PydanticOutputParser = None
    StrOutputParser = None
    PromptTemplate = None
    tool = None
    MemorySaver = None
    create_react_agent = None



BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
APP_CONFIG_PATH = DATA_DIR / "app_config.json"
PLACE_DB_PATH = DATA_DIR / "places_cache.json"
KAKAO_KEYWORDS_PATH = DATA_DIR / "kakao_keywords.json"


load_dotenv(BASE_DIR / ".env")


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


APP_CONFIG = load_json_file(APP_CONFIG_PATH)



KAKAO_LOCAL_API_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY") or os.getenv("KAKAO_API_KEY")
ODSAY_TRANSIT_API_URL = "https://api.odsay.com/v1/api/searchPubTransPathT"
ODSAY_API_KEY = os.getenv("ODSAY_API_KEY") or os.getenv("ODSAY_KEY")
ODSAY_CACHE_PATH = DATA_DIR / "odsay_transit_cache.json"
TAVILY_SEARCH_API_URL = os.getenv("TAVILY_SEARCH_API_URL", "https://api.tavily.com/search")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY_KEY")
WALK_ONLY_MAX_MINUTES = 15
DAY_TRIP_MAX_TRANSIT_LEGS = 2
TRANSIT_OPTION_LIMIT = 4

KAKAO_KEYWORD_SEARCHES = [
    (str(item["query"]), str(item["role"]))
    for item in load_json_file(KAKAO_KEYWORDS_PATH)
]

START_POINTS = APP_CONFIG["start_points"]
STYLE_KEYWORDS = APP_CONFIG["style_keywords"]
SELECTABLE_KEYWORDS = set(APP_CONFIG["selectable_keywords"])
TRANSPORT_SPEED_KMH = APP_CONFIG["transport_speed_kmh"]
ALLOWED_TRANSPORTS = set(APP_CONFIG["allowed_transports"])
ALLOWED_DURATIONS = set(APP_CONFIG["allowed_durations"])

WALK_ONLY_MAX_MINUTES = 15
WALK_ONLY_MAX_KM = TRANSPORT_SPEED_KMH["도보 중심"] * WALK_ONLY_MAX_MINUTES / 60

CHEONGJU_LAT_RANGE = tuple(APP_CONFIG["cheongju_lat_range"])
CHEONGJU_LNG_RANGE = tuple(APP_CONFIG["cheongju_lng_range"])


def load_places_cache_seed() -> list[dict[str, Any]]:
    if not PLACE_DB_PATH.exists():
        return []
    payload = json.loads(PLACE_DB_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("places", []) if isinstance(payload, dict) else []


LOCAL_FALLBACK_PLACES = load_places_cache_seed()
PLACE_FILTERS = APP_CONFIG["place_filters"]
LOW_PRIORITY_REPEATED_CAFE_NAMES = set(PLACE_FILTERS["low_priority_repeated_cafe_names"])
OVERUSED_CHUNGBUK_PLACE_NAMES = set(PLACE_FILTERS["overused_chungbuk_place_names"])
BLOCKED_PLACE_NAMES = set(PLACE_FILTERS["blocked_place_names"])
LOW_CONFIDENCE_PLACE_NAMES = set(PLACE_FILTERS["low_confidence_place_names"])
GENERIC_PLACE_NAMES = set(PLACE_FILTERS["generic_place_names"])
GENERIC_PLACE_SUFFIXES = tuple(PLACE_FILTERS["generic_place_suffixes"])
EVIDENCE_TERMS = APP_CONFIG["evidence_terms"]
ANIMAL_EVIDENCE_TERMS = EVIDENCE_TERMS["animal"]
ACTIVITY_EVIDENCE_TERMS = EVIDENCE_TERMS["activity"]
MIN_RECOMMENDATION_QUALITY = 3.0
CAFE_EVIDENCE_TERMS = EVIDENCE_TERMS["cafe"]
MEAL_EVIDENCE_TERMS = EVIDENCE_TERMS["meal"]
DEFAULT_SCORE_RULES = {
    "weather_filter_score": {
        "비": {"indoor": 2.4, "outdoor": -3.0},
        "더움": {"indoor": 1.0, "outdoor": -0.3},
        "추움": {"indoor": 1.2, "outdoor": -0.5},
        "default": {"indoor": 0.0, "outdoor": 0.4},
    },
    "indoor_preference_score": {"matching_tag": "실내", "indoor": 3.0, "outdoor": -5.0},
    "outdoor_preference_score": {"matching_tag": "실외", "indoor": -3.8, "outdoor": 2.6},
    "start_proximity_score": {
        "도보 중심": {
            "ranges": [
                {"max_km": 1.5, "score": 4.5},
                {"max_km": 3.0, "score": 3.0},
                {"max_km": 5.0, "score": 1.0},
                {"max_km": 8.0, "score": -2.0},
            ],
            "fallback_multiplier": -0.55,
            "fallback_min_score": -10.0,
        },
        "default": {
            "ranges": [
                {"max_km": 3.0, "score": 2.2},
                {"max_km": 7.0, "score": 1.4},
                {"max_km": 12.0, "score": 0.7},
                {"max_km": 20.0, "score": 0.0},
            ],
            "fallback_multiplier": -0.12,
            "fallback_min_score": -4.0,
        },
    },
    "penalties": {
        "budget_penalty": 8.0,
        "repeated_cafe_name_penalty": 4.0,
        "repeated_cafe_bathhouse_penalty": 3.0,
        "overused_chungbuk_place_penalty": 5.0,
        "pet_care_penalty": 20.0,
    },
    "agent_score_formula": {
        "quality_cap": 6.0,
        "quality_weight": 0.35,
        "matched_tag_weight": 2.0,
    },
    "intent_adjustment_score": [
        {
            "intent": "wants_family",
            "category_any": ["공원", "관광지", "박물관", "카페", "동물체험"],
            "tag_any": ["산책", "자연", "동물", "실내"],
            "score": 2.4,
        },
        {"intent": "wants_family", "role_any": ["meal", "cafe"], "score": 0.8},
        {
            "intent": "needs_quiet",
            "tag_any": ["산책", "자연"],
            "category_any": ["공원", "박물관", "관광지"],
            "score": 2.0,
        },
        {
            "intent": "needs_quiet",
            "category_any": ["상권", "시장", "카페거리"],
            "text_any": ["번화가", "중심가", "핫플"],
            "score": -2.0,
        },
        {
            "intent": "prefer_tourism_over_food",
            "role_any": ["walk", "activity"],
            "category_any": ["관광지", "공원", "박물관", "동물체험"],
            "score": 3.2,
        },
        {"intent": "prefer_tourism_over_food", "role_any": ["meal"], "score": -2.8},
        {
            "intent": "needs_short_route",
            "distance_ranges": [
                {"max_km": 2.0, "score": 3.0},
                {"max_km": 5.0, "score": 1.4},
            ],
            "fallback_multiplier": -0.35,
            "fallback_min_score": -6.0,
        },
    ],
    "slot_label_bonus_rules": [
        {"label_terms": ["동물", "동물체험"], "place_terms": ["동물", "동물원", "체험"], "score": 10.0},
        {"label_terms": ["노래방"], "place_terms": ["노래방", "노래연습", "코인노래"], "score": 10.0},
        {"label_terms": ["PC방"], "place_terms": ["PC방", "피시방", "게임방"], "score": 10.0},
        {"label_terms": ["보드게임"], "place_terms": ["보드게임"], "score": 10.0},
        {"label_terms": ["볼링"], "place_terms": ["볼링"], "score": 10.0},
        {"label_terms": ["방탈출"], "place_terms": ["방탈출"], "score": 10.0},
        {"label_terms": ["영화관"], "place_terms": ["영화관", "CGV", "롯데시네마", "메가박스"], "score": 10.0},
        {"label_terms": ["전시", "박물관", "미술관", "역사", "문화"], "place_terms": ["박물관", "미술관", "전시", "문화", "역사"], "score": 8.0},
        {"label_terms": ["쇼핑", "상권", "시장"], "place_terms": ["시장", "상권", "거리", "쇼핑", "백화점", "몰"], "score": 8.0},
        {"label_terms": ["산책", "자연", "공원", "야외"], "place_terms": ["공원", "산책", "자연", "호수", "수목원", "무심천"], "score": 7.0},
        {"label_terms": ["사진", "포토", "인생샷", "야경"], "place_terms": ["사진", "포토", "야경", "전망", "공원", "거리", "카페"], "score": 6.0},
        {"label_terms": ["데이트", "감성"], "place_terms": ["카페", "공원", "거리", "사진", "맛집", "디저트"], "score": 5.0},
        {"label_terms": ["부모님", "가족", "어른", "아이"], "place_terms": ["공원", "박물관", "카페", "식당", "한식", "동물"], "score": 5.0},
        {"label_terms": ["조용", "한적", "힐링"], "place_terms": ["공원", "카페", "산책", "자연"], "score": 5.0},
        {"label_terms": ["인기", "핫플", "유명"], "source_equals": "카카오 Local API", "score": 4.0},
        {"label_terms": ["가성비", "혼밥"], "cost_lte": 10000, "score": 4.0},
    ],
}
DEFAULT_OUTPUT_TEMPLATES = {
    "reason": {
        "with_slot": "{slot} 슬롯에 맞춰 선택했고, {matched} 선호와 맞는 {weather_note} 장소입니다.",
        "default": "{matched} 선호와 맞고, {weather_note} 일정으로 활용하기 좋습니다.",
        "matched_default": "거리/예산",
        "indoor_note": "실내",
        "outdoor_note": "야외",
    },
    "tool_decision": {
        "input_validation": "사용자 입력의 출발지, 예산, 이동수단, 여행 기간, 자연어 요청을 검증했습니다.",
        "privacy": "전화번호나 이메일 등 개인정보 형식은 제거하도록 처리했습니다.",
        "memory_used": "세션 메모리의 최근 대화 맥락을 참고했습니다.",
        "memory_unused": "이전 대화 맥락을 이어받는 요청이 아니어서 메모리 병합을 사용하지 않았습니다.",
        "intent_analysis": "자연어 요청과 선택 키워드를 태그, 필수 조건, 회피 조건으로 분석했습니다.",
        "rag": "장소 DB/RAG 문서를 검색해 후보 점수와 근거에 반영했습니다.",
        "tavily_used": "외부 리뷰/후기 근거를 검색해 후보 점수에 반영했습니다.",
        "tavily_skipped": "외부 리뷰 근거가 필수 조건이 아니거나 API 키가 없어 생략했습니다.",
        "tavily_failed": "외부 리뷰 검색을 시도했지만 실패해 로컬 점수를 유지했습니다.",
        "llm_route_used": "LLM이 후보를 비교해 일정 순서를 선택했습니다.",
        "llm_route_failed": "LLM 동선 선택 실패 또는 빈 결과를 보완하기 위해 규칙 기반 추천을 사용했습니다.",
        "fallback_used": "품질 검사 또는 LLM 실패를 보완하기 위해 규칙 기반 추천을 사용했습니다.",
        "fallback_unused": "LLM/tool 기반 후보 선택이 성공하여 규칙 기반 fallback은 사용하지 않았습니다.",
        "distance": "최종 추천 전 장소 간 거리와 이동 시간을 계산했습니다.",
        "odsay_used": "대중교통 이동 구간에서 ODSay 경로 후보를 반영했습니다.",
        "odsay_failed": "대중교통 경로를 시도했지만 경로가 없거나 API 오류로 상세 후보를 반영하지 못했습니다.",
        "odsay_skipped": "도보 가능 구간이거나 도보 중심 조건이라 대중교통 상세 검색을 생략했습니다.",
        "output_parser": "최종 응답을 TripPlan Pydantic 구조로 검증했습니다.",
    },
    "middleware_decision": {
        "input_validation": "사용자 입력의 출발지, 예산, 이동수단, 여행 기간, 자연어 요청을 검증했습니다.",
        "privacy_used": "사용자 입력에서 전화번호 또는 이메일 형식을 감지해 제거했습니다.",
        "privacy_unused": "전화번호나 이메일 형식이 없어 제거할 개인정보가 없었습니다.",
        "memory_used": "이전 대화 {memory_turns}턴을 참고해 현재 요청과 병합했습니다.",
        "memory_unused": "이전 대화 맥락을 이어받는 요청이 아니어서 메모리 병합을 사용하지 않았습니다.",
    },
    "fast_final_comment": "{transport} 이동과 자연어 요청을 기준으로 {route_preview} 순서가 가장 무난합니다. 예상 비용은 {total_cost:,}원, 이동은 약 {total_move_minutes}분/{total_distance}km입니다.",
}
SCORE_RULES = APP_CONFIG.get("score_rules", DEFAULT_SCORE_RULES)
SCORE_PENALTIES = SCORE_RULES["penalties"]
AGENT_SCORE_FORMULA = SCORE_RULES["agent_score_formula"]
OUTPUT_TEMPLATES = APP_CONFIG.get("output_templates", DEFAULT_OUTPUT_TEMPLATES)


def http_get(
    url: str,
    params: dict[str, str] | None = None,
    timeout: int = 12,
    headers: dict[str, str] | None = None,
) -> bytes:
    target_url = url
    if params:
        target_url = f"{url}?{urllib.parse.urlencode(params, safe='%')}"
    request_headers = {
        "Accept": "application/json, application/xml, text/xml, */*",
        "User-Agent": "CheongjuTripAgent/1.0",
    }
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        target_url,
        headers=request_headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def http_post_json(
    url: str,
    payload: dict[str, Any],
    timeout: int = 10,
    headers: dict[str, str] | None = None,
) -> bytes:
    request_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "CheongjuTripAgent/1.0",
    }
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_api_payload(raw: bytes) -> Any:
    text = raw.decode("utf-8-sig", errors="replace").strip()
    if not text:
        return {}
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    return xml_to_dict(ET.fromstring(text))


def xml_to_dict(element: ET.Element) -> Any:
    children = list(element)
    if not children:
        return element.text.strip() if element.text else ""

    grouped: dict[str, Any] = {}
    for child in children:
        value = xml_to_dict(child)
        if child.tag in grouped:
            if not isinstance(grouped[child.tag], list):
                grouped[child.tag] = [grouped[child.tag]]
            grouped[child.tag].append(value)
        else:
            grouped[child.tag] = value
    return grouped


def deep_find_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        items: list[dict[str, Any]] = []
        for entry in data:
            items.extend(deep_find_items(entry))
        return items
    if not isinstance(data, dict):
        return []

    for key in ("item", "items", "list", "data", "body", "response", "result"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = deep_find_items(value)
            if nested:
                return nested

    if any(key.lower() in {"title", "name", "addr1", "mapx", "mapy"} for key in data):
        return [data]

    items = []
    for value in data.values():
        items.extend(deep_find_items(value))
    return items


def first_text(item: dict[str, Any], keys: list[str]) -> str:
    lowered = {str(key).lower(): value for key, value in item.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value not in (None, ""):
            return str(value).strip()
    return ""


def first_float(item: dict[str, Any], keys: list[str]) -> float | None:
    value = first_text(item, keys)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def is_cheongju_place(name: str, address: str, admin_area: str, lat: float, lng: float) -> bool:
    if admin_area:
        return "청주" in admin_area
    if address:
        return "청주" in address
    if "청주" in name:
        return True
    return CHEONGJU_LAT_RANGE[0] <= lat <= CHEONGJU_LAT_RANGE[1] and CHEONGJU_LNG_RANGE[0] <= lng <= CHEONGJU_LNG_RANGE[1]


def infer_category(item: dict[str, Any], name: str, address: str, description: str) -> str:
    raw = first_text(item, ["tourSe", "cat3", "cat2", "cat1", "contenttypeid", "category", "type", "분류"])
    haystack = f"{name} {address} {description} {raw}"
    direct_evidence = f"{name} {address} {raw}"
    name_text = name
    if any(word in direct_evidence for word in ANIMAL_EVIDENCE_TERMS):
        return "동물체험"
    if any(word in haystack for word in ACTIVITY_EVIDENCE_TERMS):
        return "놀거리"
    if "카페" in raw or any(word in name_text for word in CAFE_EVIDENCE_TERMS):
        return "카페"
    if any(word in raw for word in MEAL_EVIDENCE_TERMS):
        return "맛집"
    if any(word in name_text for word in ["박물관", "미술관", "전시관", "기념관", "체험관", "교육원", "공예관"]):
        return "박물관"
    if any(word in name_text for word in ["성안길", "상권"]):
        return "상권"
    if "시장" in name_text:
        return "시장"
    if any(word in name_text for word in ["공원", "수목원", "휴양림", "무심천"]):
        return "공원"
    if any(word in name_text for word in ["산", "걷기길", "트레킹", "산성", "댐", "전망대", "벚꽃길", "구곡"]):
        return "관광지"

    if any(word in haystack for word in ["성안길", "상권", "쇼핑"]):
        return "상권"
    if any(word in haystack for word in ["박물관", "전시", "미술관", "기념관"]):
        return "박물관"
    if any(word in haystack for word in ["시장", "먹거리", "맛집", "음식", "식당"]):
        return "시장"
    if any(word in haystack for word in ["공원", "호수", "수목원", "자연휴양림"]):
        return "공원"
    if any(word in haystack for word in ["카페"]):
        return "카페거리"
    return "관광지"


def infer_tags(category: str, name: str, address: str, description: str) -> list[str]:
    haystack = f"{category} {name} {address} {description}"
    direct_evidence = f"{category} {name} {address}"
    tags = {"사진"}
    if category == "동물체험" or any(word in direct_evidence for word in ANIMAL_EVIDENCE_TERMS):
        tags.update(["동물", "산책", "자연"])
    if category in {"맛집", "시장", "카페", "카페거리"} or any(word in haystack for word in ["맛집", "먹거리", "시장", "카페"]):
        tags.update(["맛집", "카페"])
    if category in {"박물관"} or any(word in haystack for word in ["박물관", "전시", "역사", "문화", "유적"]):
        tags.update(["역사", "실내"])
    if category in {"공원"} or any(word in haystack for word in ["공원", "호수", "산", "숲", "둘레길", "산책"]):
        tags.update(["자연", "산책", "실외"])
    if category in {"관광지", "동물체험"}:
        tags.update(["산책", "자연", "실외"])
    if category == "놀거리":
        tags.update(["실내"])
    if any(word in haystack for word in ["노래방", "노래연습장", "노래연습실", "노래궁", "코인노래"]):
        tags.update(["노래방", "실내"])
    if any(word in haystack for word in ["PC방", "피시방", "게임방"]):
        tags.update(["PC방", "피시방", "놀거리", "실내"])
    if any(word in haystack for word in ["보드게임", "영화관", "메가박스", "CGV", "롯데시네마"]):
        tags.update(["놀거리", "실내"])
    return sorted(tags)


def place_role_for_category(category: str) -> str:
    for role, categories in APP_CONFIG["role_category_map"].items():
        if category in categories:
            return role
    return "activity"


def role_label(role: str) -> str:
    return APP_CONFIG["role_labels"].get(role, APP_CONFIG["role_labels"]["default"])


def activity_signature(place: dict[str, Any]) -> str | None:
    haystack = f"{place.get('name', '')} {place.get('category', '')} {' '.join(place.get('tags', []))}"
    for rule in APP_CONFIG["activity_signatures"]:
        if any(term in haystack for term in rule["terms"]):
            return str(rule["signature"])
    if place.get("category") == "놀거리":
        return place.get("name")
    return None


def duplicates_activity_type(place: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
    signature = activity_signature(place)
    if signature is None:
        return False
    return any(activity_signature(item) == signature for item in selected)


def kakao_map_search_url(name: str, address: str = "") -> str:
    query = " ".join(part for part in [name.strip(), address.strip()] if part)
    return f"https://map.kakao.com/link/search/{urllib.parse.quote(query)}"


def is_generic_area_place(name: str, address: str, category: str, source: str) -> bool:
    normalized_name = re.sub(r"\s+", "", name)
    if name in GENERIC_PLACE_NAMES:
        return True
    if any(name.endswith(suffix) for suffix in GENERIC_PLACE_SUFFIXES):
        return True
    if "일대" in address:
        return True
    if source != "카카오 Local API" and category in {"상권", "카페거리"}:
        return True
    if source != "카카오 Local API" and normalized_name in {"성안길", "운리단길", "수암골"}:
        return True
    if source == "로컬 보강 DB" and "인근" in name:
        return True
    return False


def has_animal_evidence(name: str, address: str, category: str, kakao_category: str = "") -> bool:
    direct_evidence = f"{name} {address} {category} {kakao_category}"
    return any(word in direct_evidence for word in ANIMAL_EVIDENCE_TERMS)


def is_low_confidence_place(
    name: str,
    source: str,
    role: str,
    category: str,
    tags: list[str],
    phone: str,
) -> bool:
    if name in BLOCKED_PLACE_NAMES:
        return True
    if name in LOW_CONFIDENCE_PLACE_NAMES:
        return True
    if source != "카카오 Local API":
        return False
    if role != "activity":
        return False
    if category in {"동물체험", "놀거리"} and not phone.strip():
        return True
    if "동물" in tags and not phone.strip():
        return True
    return False


def is_pet_care_place(place: dict[str, Any]) -> bool:
    haystack = f"{place['name']} {place.get('kakao_category', '')} {' '.join(place.get('tags', []))}"
    return any(word in haystack for word in APP_CONFIG["pet_care_terms"])


def kakao_role_matches_query(role: str, name: str, kakao_category: str, query: str) -> bool:
    evidence = f"{name} {kakao_category}"
    if role == "meal":
        if any(word in evidence for word in CAFE_EVIDENCE_TERMS) and not any(word in evidence for word in MEAL_EVIDENCE_TERMS):
            return False
    if role == "cafe":
        if any(word in evidence for word in MEAL_EVIDENCE_TERMS) and not any(word in evidence for word in CAFE_EVIDENCE_TERMS):
            return False
    if role == "activity":
        activity_match = any(word in evidence for word in ACTIVITY_EVIDENCE_TERMS)
        animal_match = any(word in evidence for word in ANIMAL_EVIDENCE_TERMS)
        if not activity_match and not animal_match and any(word in evidence for word in CAFE_EVIDENCE_TERMS + MEAL_EVIDENCE_TERMS):
            return False
    return True


SOURCE_BASE_QUALITY = APP_CONFIG["source_base_quality"]


def place_quality_score(place: dict[str, Any]) -> float:
    name = str(place.get("name", "")).strip()
    address = str(place.get("address", "")).strip()
    category = str(place.get("category", "")).strip()
    source = str(place.get("source", "")).strip()
    role = str(place.get("role", "")).strip()
    phone = str(place.get("phone", "")).strip()
    url = str(place.get("url") or place.get("map_url") or "").strip()
    kakao_category = str(place.get("kakao_category", "")).strip()
    tags = [str(tag).strip() for tag in place.get("tags", []) if str(tag).strip()]

    if is_generic_area_place(name, address, category, source):
        return 0.0
    if is_low_confidence_place(name, source, role, category, tags, phone):
        return 0.0

    score = SOURCE_BASE_QUALITY.get(source, 1.0)
    if source == "카카오 Local API":
        if url:
            score += 1.5
        if phone:
            score += 1.5
        if address and "일대" not in address:
            score += 1.0
        if kakao_category:
            score += 1.0
        if role in {"meal", "cafe"}:
            score += 0.7
        if role == "activity" and category in {"놀거리", "동물체험"}:
            score += 0.7
    elif source == "충청북도 관광명소정보 API":
        if address and "일대" not in address:
            score += 0.8
        if category in {"박물관", "공원", "관광지", "동물체험", "놀거리"}:
            score += 0.7
    elif source == "로컬 보강 DB":
        if address:
            score += 0.8

    if name in LOW_CONFIDENCE_PLACE_NAMES:
        score -= 10.0
    return round(max(0.0, score), 2)


def estimate_kakao_cost(role: str, category: str, name: str, query: str, kakao_category: str) -> int:
    haystack = f"{name} {query} {kakao_category} {category}"
    if role == "meal":
        return 12000
    if role == "cafe":
        return 7000
    for rule in APP_CONFIG["kakao_cost_rules"]:
        if not any(word in haystack for word in rule["terms"]):
            continue
        if rule.get("zero_if_all_terms") and all(term in haystack for term in rule["zero_if_all_terms"]):
            return 0
        return int(rule["cost"])
    if category in APP_CONFIG["kakao_area_cost_categories"]:
        return 10000
    return 5000


def pet_care_penalty(place: dict[str, Any], tags: list[str]) -> float:
    if "동물먹이" not in tags:
        return 0.0
    if is_pet_care_place(place):
        return float(SCORE_PENALTIES["pet_care_penalty"])
    return 0.0


def normalize_kakao_place(item: dict[str, Any], query: str, role: str) -> dict[str, Any] | None:
    name = first_text(item, ["place_name"])
    address = first_text(item, ["road_address_name", "address_name"])
    lat = first_float(item, ["y"])
    lng = first_float(item, ["x"])
    if not name or lat is None or lng is None:
        return None
    if not is_cheongju_place(name, address, "", lat, lng):
        return None

    kakao_category = first_text(item, ["category_group_name", "category_name"])
    phone = first_text(item, ["phone"])
    place_url = first_text(item, ["place_url"])
    if not kakao_role_matches_query(role, name, kakao_category, query):
        return None
    if role == "meal":
        category = "맛집"
        tags = ["맛집", "식사", "로컬"]
        stay_minutes = 60
        indoor = True
    elif role == "cafe":
        category = "카페"
        tags = ["카페", "디저트", "사진"]
        stay_minutes = 50
        indoor = True
    else:
        category = infer_category({"category": kakao_category}, name, address, query)
        tags = infer_tags(category, name, address, query)
        stay_minutes = 80
        indoor = "실내" in tags or category in {"박물관", "카페거리", "상권", "놀거리"}
    cost = estimate_kakao_cost(role, category, name, query, kakao_category)
    if is_low_confidence_place(name, "카카오 Local API", role, category, tags, phone):
        return None

    return {
        "name": name,
        "category": category,
        "role": role,
        "lat": lat,
        "lng": lng,
        "tags": sorted(set(tags)),
        "score": 4.4 if role in {"meal", "cafe"} else 4.2,
        "cost": cost,
        "indoor": indoor,
        "stay_minutes": stay_minutes,
        "source": "카카오 Local API",
        "address": address,
        "phone": phone,
        "url": place_url,
        "map_url": place_url or kakao_map_search_url(name, address),
        "kakao_category": kakao_category,
        "search_query": query,
    }


def normalize_place(item: dict[str, Any], source: str) -> dict[str, Any] | None:
    name = first_text(item, ["title", "name", "placeName", "tourNm", "attrNm", "관광지명", "명칭"])
    address = first_text(item, ["addr1", "addr", "address", "adres", "roadAddr", "newAddr", "주소"])
    admin_area = first_text(item, ["areaSe", "sigungu", "sigunguName", "시군구"])
    description = first_text(item, ["overview", "summary", "content", "description", "desc", "intrcn", "설명"])
    lat = first_float(item, ["mapy", "lat", "latitude", "y", "위도"])
    lng = first_float(item, ["mapx", "lng", "lon", "longitude", "x", "경도"])

    if not name or lat is None or lng is None:
        return None
    if not is_cheongju_place(name, address, admin_area, lat, lng):
        return None

    category = infer_category(item, name, address, description)
    tags = infer_tags(category, name, address, description)
    indoor = "실내" in tags or category in {"박물관", "카페거리", "놀거리"}
    cost = 8000 if category in {"시장", "카페거리"} else 0
    stay_minutes = 90 if category in {"관광지", "박물관"} else 70

    return {
        "name": name,
        "category": category,
        "role": place_role_for_category(category),
        "lat": lat,
        "lng": lng,
        "tags": tags,
        "score": 4.2,
        "cost": cost,
        "indoor": indoor,
        "stay_minutes": stay_minutes,
        "source": source,
        "address": address,
        "map_url": kakao_map_search_url(name, address),
    }


def dedupe_places(places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for place in places:
        key = re.sub(r"\s+", "", place["name"]).lower()
        deduped.setdefault(key, place)
    return sorted(deduped.values(), key=lambda place: place["name"])


def save_place_db(
    places: list[dict[str, Any]],
    source: str,
    errors: list[str] | None = None,
    source_counts: dict[str, int] | None = None,
) -> None:
    global PLACE_DB_CACHE
    DATA_DIR.mkdir(exist_ok=True)
    payload = {
        "city": "청주",
        "source": source,
        "count": len(places),
        "source_counts": source_counts or dict(sorted(source_counter(places).items())),
        "sync_errors": errors or [],
        "places": places,
    }
    PLACE_DB_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    PLACE_DB_CACHE = None


def source_counter(places: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for place in places:
        source = str(place.get("source", "")).strip() or "출처 없음"
        counts[source] = counts.get(source, 0) + 1
    return counts


def source_summary(places: list[dict[str, Any]]) -> str:
    sources = sorted(source_counter(places))
    return " / ".join(sources) if sources else "로컬 JSON DB"


def load_place_db() -> list[dict[str, Any]]:
    global PLACE_DB_CACHE
    if PLACE_DB_CACHE is not None:
        return PLACE_DB_CACHE
    if not PLACE_DB_PATH.exists():
        PLACE_DB_CACHE = sync_place_db()
        return PLACE_DB_CACHE
    payload = json.loads(PLACE_DB_PATH.read_text(encoding="utf-8"))
    places = payload if isinstance(payload, list) else payload.get("places", [])
    source_counts = {} if isinstance(payload, list) else payload.get("source_counts", {})
    has_synced_sources = bool(source_counts)
    has_kakao_places = int(source_counts.get("카카오 Local API", 0) or 0) > 0
    local_only_cache = len(places) <= len(LOCAL_FALLBACK_PLACES) and all(
        str(place.get("source", "")) == "로컬 보강 DB" for place in places
    )
    if KAKAO_REST_API_KEY and (not has_synced_sources or local_only_cache or not has_kakao_places):
        try:
            PLACE_DB_CACHE = sync_place_db()
            return PLACE_DB_CACHE
        except (RuntimeError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ET.ParseError, OSError) as error:
            logger.warning("Place DB auto-sync failed, using cached places: %s", error)
    PLACE_DB_CACHE = sanitize_place_db(places + LOCAL_FALLBACK_PLACES)
    return PLACE_DB_CACHE


ROLE_DEFAULTS = {
    role: (config["category"], config["tags"])
    for role, config in APP_CONFIG["role_defaults"].items()
}


def sanitize_place_db(places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = []
    for place in places:
        name = str(place.get("name", "")).strip()
        if name in BLOCKED_PLACE_NAMES:
           continue
        address = str(place.get("address", "")).strip()
        lat = place.get("lat")
        lng = place.get("lng")
        if not name or lat is None or lng is None:
            continue
        if not is_cheongju_place(name, address, "", float(lat), float(lng)):
            continue

        source = str(place.get("source", "")).strip()
        search_query = str(place.get("search_query", "")).strip()
        kakao_category = str(place.get("kakao_category", "")).strip()
        original_role = str(place.get("role", "")).strip()
        if source == "카카오 Local API" and original_role and not kakao_role_matches_query(original_role, name, kakao_category, search_query):
            continue
        raw_category = place.get("kakao_category") if source == "카카오 Local API" else place.get("category", "")
        category = infer_category({"category": raw_category or ""}, name, address, search_query)
        if is_generic_area_place(name, address, category, source):
            continue
        if original_role in ROLE_DEFAULTS:
            role = original_role
            category, tags = ROLE_DEFAULTS[role]
        else:
            role = place_role_for_category(category)
            tags = infer_tags(category, name, address, "")
        existing_tags = [str(tag).strip() for tag in place.get("tags", []) if str(tag).strip()]
        if not has_animal_evidence(name, address, category, str(place.get("kakao_category", ""))):
            existing_tags = [tag for tag in existing_tags if tag != "동물"]
        tags = sorted(set(tags).union(existing_tags))
        kakao_category = str(place.get("kakao_category", place.get("category", ""))).strip()
        phone = str(place.get("phone", "")).strip()
        if is_low_confidence_place(name, source, role, category, tags, phone):
            continue
        cost = int(place.get("cost") or 0)
        if source == "카카오 Local API":
            cost = estimate_kakao_cost(role, category, name, search_query, kakao_category)
        map_url = str(place.get("map_url") or place.get("url") or "").strip() or kakao_map_search_url(name, address)
        normalized_place = {
            **place,
            "category": category,
            "role": role,
            "tags": tags,
            "phone": phone,
            "cost": cost,
            "indoor": bool(place.get("indoor")) or "실내" in tags or category in {"박물관", "카페거리", "카페", "맛집", "상권", "놀거리"},
            "map_url": map_url,
        }
        normalized_place["quality_score"] = place_quality_score(normalized_place)
        sanitized.append(
            normalized_place
        )
    return dedupe_places(sanitized)


def sync_place_db() -> list[dict[str, Any]]:
    errors: list[str] = []
    collected: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    fetchers = (
        ("카카오 Local API", fetch_kakao_places),
    )
    for source, fetcher in fetchers:
        try:
            places = dedupe_places(fetcher())
            if places:
                source_counts[source] = len(places)
                collected.extend(places)
                continue
            errors.append(f"{source}: 청주 장소 데이터 없음")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ET.ParseError, OSError) as error:
            errors.append(f"{source}: {error}")

    collected = sanitize_place_db(collected + LOCAL_FALLBACK_PLACES)
    if collected:
        save_place_db(collected, source_summary(collected), errors, source_counter(collected))
        return collected

    raise RuntimeError("청주 장소 데이터를 수집하지 못했습니다. " + " / ".join(errors))



def fetch_kakao_places() -> list[dict[str, Any]]:
    if not KAKAO_REST_API_KEY:
        return []

    places: list[dict[str, Any]] = []
    kakao_key = KAKAO_REST_API_KEY.strip()
    authorization = kakao_key if kakao_key.startswith("KakaoAK ") else f"KakaoAK {kakao_key}"
    headers = {"Authorization": authorization}
    for query, role in KAKAO_KEYWORD_SEARCHES:
        for page in range(1, 3):
            raw = http_get(
                KAKAO_LOCAL_API_URL,
                {
                    "query": query,
                    "size": "15",
                    "page": str(page),
                },
                headers=headers,
            )
            payload = parse_api_payload(raw)
            documents = payload.get("documents", []) if isinstance(payload, dict) else []
            places.extend(
                place
                for item in documents
                if (place := normalize_kakao_place(item, query, role))
            )
            meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
            if meta.get("is_end", True):
                break
    return places


def infer_kakao_search_roles(query: str) -> list[str]:
    role_scores = {
        "meal": sum(1 for term in MEAL_EVIDENCE_TERMS if term in query),
        "cafe": sum(1 for term in CAFE_EVIDENCE_TERMS if term in query),
        "activity": sum(1 for term in ACTIVITY_EVIDENCE_TERMS + ANIMAL_EVIDENCE_TERMS if term in query),
    }
    ranked_roles = [role for role, score in sorted(role_scores.items(), key=lambda item: item[1], reverse=True) if score]
    return ranked_roles or ["activity", "cafe", "meal"]


def search_kakao_places_by_query(query: str) -> list[dict[str, Any]]:
    if not KAKAO_REST_API_KEY:
        return []

    kakao_key = KAKAO_REST_API_KEY.strip()
    authorization = kakao_key if kakao_key.startswith("KakaoAK ") else f"KakaoAK {kakao_key}"
    headers = {"Authorization": authorization}
    places: list[dict[str, Any]] = []
    for page in range(1, 3):
        raw = http_get(
            KAKAO_LOCAL_API_URL,
            {
                "query": query,
                "size": "15",
                "page": str(page),
            },
            headers=headers,
        )
        payload = parse_api_payload(raw)
        documents = payload.get("documents", []) if isinstance(payload, dict) else []
        for item in documents:
            for role in infer_kakao_search_roles(query):
                place = normalize_kakao_place(item, query, role)
                if place:
                    places.append(place)
                    break
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if meta.get("is_end", True):
            break
    return dedupe_places(places)


def transit_cache_key(start: dict[str, float], end: dict[str, float]) -> str:
    return f"{start['lat']:.6f},{start['lng']:.6f}->{end['lat']:.6f},{end['lng']:.6f}"


def load_odsay_cache() -> dict[str, Any]:
    if not ODSAY_CACHE_PATH.exists():
        return {}
    try:
        payload = json.loads(ODSAY_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_odsay_cache(cache: dict[str, Any]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    ODSAY_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def unique_bus_numbers(option: dict[str, Any]) -> list[str]:
    numbers: list[str] = []
    for segment in option.get("segments", []):
        for bus_no in segment.get("bus_numbers", []):
            if bus_no and bus_no not in numbers:
                numbers.append(bus_no)
    return numbers


def format_transit_summary(option: dict[str, Any]) -> str:
    bus_numbers = unique_bus_numbers(option)
    bus_text = ", ".join(bus_numbers[:4]) if bus_numbers else "버스"
    stations = [
        f"{segment.get('start_station', '')} 승차 → {segment.get('end_station', '')} 하차"
        for segment in option.get("segments", [])
        if segment.get("start_station") or segment.get("end_station")
    ]
    station_text = " / ".join(stations[:2]) if stations else "정류장 정보 확인 필요"
    return f"버스 {bus_text} / {station_text} / 약 {option.get('total_time', 0)}분"


def parse_odsay_path(path: dict[str, Any]) -> dict[str, Any] | None:
    info = path.get("info", {}) if isinstance(path, dict) else {}
    total_time = int(info.get("totalTime") or 0)
    if total_time <= 0:
        return None

    segments: list[dict[str, Any]] = []
    for sub_path in path.get("subPath", []):
        if int(sub_path.get("trafficType") or 0) != 2:
            continue
        lanes = sub_path.get("lane", [])
        bus_numbers: list[str] = []
        for lane in lanes:
            bus_no = str(lane.get("busNo") or lane.get("name") or "").strip()
            if bus_no and bus_no not in bus_numbers:
                bus_numbers.append(bus_no)
        if not bus_numbers:
            continue
        segments.append(
            {
                "bus_numbers": bus_numbers,
                "start_station": str(sub_path.get("startName") or "").strip(),
                "end_station": str(sub_path.get("endName") or "").strip(),
                "section_time": int(sub_path.get("sectionTime") or 0),
            }
        )

    if not segments:
        return None

    option = {
        "total_time": total_time,
        "payment": int(info.get("payment") or 0),
        "bus_count": len(segments),
        "segments": segments,
    }
    option["summary"] = format_transit_summary(option)
    return option


def odsay_transit_options(start: dict[str, float], end: dict[str, float]) -> tuple[list[dict[str, Any]], str | None]:
    api_key = os.getenv("ODSAY_API_KEY") or os.getenv("ODSAY_KEY")
    if not api_key:
        return [], "ODSAY_API_KEY 없음"

    cache = load_odsay_cache()
    key = transit_cache_key(start, end)
    if key in cache and cache[key]:
        return cache[key], None

    try:
        raw = http_get(
            ODSAY_TRANSIT_API_URL,
            {
                "SX": f"{start['lng']:.7f}",
                "SY": f"{start['lat']:.7f}",
                "EX": f"{end['lng']:.7f}",
                "EY": f"{end['lat']:.7f}",
                "apiKey": api_key,
            },
            timeout=10,
        )
        payload = parse_api_payload(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ET.ParseError, OSError) as error:
        logger.warning("ODsay route failed: %s", error)
        return [], str(error)

    if isinstance(payload, dict) and payload.get("error"):
        errors = payload.get("error")
        if isinstance(errors, list) and errors:
            message = errors[0].get("message") if isinstance(errors[0], dict) else str(errors[0])
            code = errors[0].get("code") if isinstance(errors[0], dict) else ""
            logger.warning("ODsay API error %s: %s", code, message)
            return [], f"ODsay API 오류 {code}: {message}".strip()
        logger.warning("ODsay API error: %s", payload.get("error"))
        return [], f"ODsay API 오류: {payload.get('error')}"

    paths = payload.get("result", {}).get("path", []) if isinstance(payload, dict) else []
    options = []
    for path in paths:
        option = parse_odsay_path(path)
        if option:
            options.append(option)
        if len(options) >= TRANSIT_OPTION_LIMIT:
            break

    if options:
        cache[key] = options
        save_odsay_cache(cache)
        return options, None
    logger.warning("ODsay route failed: no transit path")
    return [], "ODsay 대중교통 경로 없음"


class AgentState(TypedDict):
    session_id: str
    start_name: str
    start_point: dict[str, float]
    duration: str
    style_text: str
    selected_keywords: list[str]
    transport: str
    budget: int
    weather: str
    tags: list[str]
    warnings: list[str]
    memory_context: list[dict[str, Any]]


class TravelIntent(BaseModel):
    tags: list[str] = Field(default_factory=list)
    must_have: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    pace: Literal["느긋", "보통", "빡빡"] = "보통"
    needs_external_review: bool = False
    needs_short_route: bool = False
    needs_quiet: bool = False
    wants_family: bool = False
    wants_popular: bool = False
    prefer_tourism_over_food: bool = False
    tool_plan: list[str] = Field(default_factory=list)
    intent_summary: str = "규칙 기반 태그 분석"


class TripSummary(BaseModel):
    title: str
    tags: list[str]
    route_text: str
    total_cost: int
    budget: int
    total_distance: float
    total_move_minutes: int
    finish: str
    weather: str
    transport: str
    ai_comment: str
    ai_mode: str
    planner_mode: str
    intent: dict[str, Any] = Field(default_factory=dict)
    planner_decision_summary: str | None = None
    session_id: str
    memory_turns: int = 0
    rag_document_count: int = 0


class TripPlan(BaseModel):
    summary: TripSummary
    schedule: list[dict[str, Any]]
    places: list[dict[str, Any]]
    legs: list[dict[str, Any]]
    warnings: list[str]
    agent_flow: list[str]
    tool_decision: list[dict[str, Any]] = Field(default_factory=list)
    rag_sources: list[dict[str, Any]] = Field(default_factory=list)
    middleware_decision: list[dict[str, Any]] = Field(default_factory=list)


class RoutePlannerDecision(BaseModel):
    selected: list[dict[str, Any]] = Field(default_factory=list)
    decision_summary: str = ""
    rejected_notes: list[str] = Field(default_factory=list)


INTENT_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=TravelIntent) if PydanticOutputParser else None
ROUTE_DECISION_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=RoutePlannerDecision) if PydanticOutputParser else None
TRIP_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=TripPlan) if PydanticOutputParser else None
GRAPH_MEMORY = MemorySaver() if MemorySaver else None
SESSION_STORE: dict[str, list[dict[str, Any]]] = {}
PLACE_DB_CACHE: list[dict[str, Any]] | None = None

RECENT_MEMORY_EXCLUSION_TURNS = 3


def normalize_place_name_for_memory(name: str) -> str:
    return re.sub(r"\s+", "", str(name).strip()).lower()


def used_place_names_from_memory(memory_context: list[dict[str, Any]], recent_turns: int = RECENT_MEMORY_EXCLUSION_TURNS) -> set[str]:
    used_names: set[str] = set()

    for item in memory_context[-recent_turns:]:
        summary = item.get("summary", {})
        if isinstance(summary, dict):
            route_text = str(summary.get("route_text", ""))
            for name in route_text.split("→"):
                cleaned = name.strip()
                if cleaned:
                    used_names.add(cleaned)

        places = item.get("places", [])
        if isinstance(places, list):
            for place in places:
                if isinstance(place, dict) and place.get("name"):
                    used_names.add(str(place["name"]).strip())

    return used_names


def memory_name_keys(memory_context: list[dict[str, Any]], recent_turns: int = RECENT_MEMORY_EXCLUSION_TURNS) -> set[str]:
    return {
        normalize_place_name_for_memory(name)
        for name in used_place_names_from_memory(memory_context, recent_turns)
        if normalize_place_name_for_memory(name)
    }


def is_memory_excluded_place(place: dict[str, Any], excluded_name_keys: set[str]) -> bool:
    return normalize_place_name_for_memory(str(place.get("name", ""))) in excluded_name_keys


def model_provider():
    return get_model_provider()



def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def remove_private_info(text: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    sanitized = re.sub(r"01[016789]-?\d{3,4}-?\d{4}", "[전화번호 제거]", text)
    sanitized = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[이메일 제거]", sanitized)
    if sanitized != text:
        warnings.append("개인정보 제거 Middleware가 전화번호 또는 이메일 형식을 제거했습니다.")
    return sanitized, warnings


def normalize_selected_keywords(raw_keywords: Any) -> list[str]:
    if raw_keywords is None:
        return []
    if isinstance(raw_keywords, str):
        values = re.split(r"[,/\s]+", raw_keywords.strip())
    elif isinstance(raw_keywords, list):
        values = [str(value).strip() for value in raw_keywords]
    else:
        values = [str(raw_keywords).strip()]

    selected: list[str] = []
    for value in values:
        if value in SELECTABLE_KEYWORDS and value not in selected:
            selected.append(value)
        if len(selected) >= 5:
            break

    indoor_selected = "실내" in selected
    outdoor_selected = "실외" in selected
    if indoor_selected == outdoor_selected:
        selected = [keyword for keyword in selected if keyword not in {"실내", "실외"}]
    return selected


def normalize_session_id(raw_session_id: Any) -> str:
    session_id = str(raw_session_id or "default").strip()
    session_id = re.sub(r"[^a-zA-Z0-9_.:-]", "-", session_id)
    return session_id[:80] or "default"


def latest_session_payload(session_id: str) -> dict[str, Any]:
    history = SESSION_STORE.get(session_id, [])
    for item in reversed(history):
        if isinstance(item.get("payload"), dict):
            return dict(item["payload"])
    return {}


def merge_memory_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    session_id = normalize_session_id(payload.get("session_id"))
    history = SESSION_STORE.get(session_id, [])
    previous_payload = latest_session_payload(session_id)
    if not previous_payload:
        return {**payload, "session_id": session_id}, history

    style_text = str(payload.get("style_text", "")).strip()
    memory_terms = ("아까", "이전", "방금", "그 조건", "저번", "기존", "바꿔", "변경", "만 바꿔")
    should_merge = any(term in style_text for term in memory_terms)
    if not should_merge:
        return {**payload, "session_id": session_id}, history

    merged = {**previous_payload, **{key: value for key, value in payload.items() if value not in (None, "")}}
    if any(term in style_text for term in ("비", "우천", "장마")):
        merged["weather"] = "비"
    elif "더움" in style_text or "더운" in style_text:
        merged["weather"] = "더움"
    elif "추움" in style_text or "추운" in style_text:
        merged["weather"] = "추움"

    if "도보" in style_text or "걸어서" in style_text:
        merged["transport"] = "도보 중심"
    elif "대중교통" in style_text or "버스" in style_text:
        merged["transport"] = "대중교통"

    if "당일" in style_text:
        merged["duration"] = "당일치기"
    merged["session_id"] = session_id
    return merged, history


def remember_turn(session_id: str, payload: dict[str, Any], result: dict[str, Any]) -> None:
    history = SESSION_STORE.setdefault(session_id, [])
    history.append(
        {
            "payload": {
                key: payload.get(key)
                for key in ["session_id", "start_name", "duration", "style_text", "keywords", "transport", "budget", "weather"]
            },
            "summary": result.get("summary", {}),
            "places": result.get("places", []),
        }
    )
    del history[:-6]


def validate_and_normalize(payload: dict[str, Any]) -> tuple[AgentState | None, list[str]]:
    warnings: list[str] = []
    merged_payload, memory_context = merge_memory_payload(payload)
    session_id = normalize_session_id(merged_payload.get("session_id"))
    selected_keywords = normalize_selected_keywords(merged_payload.get("keywords"))
    raw_style_text = str(merged_payload.get("style_text", "")).strip()
    style_text, privacy_warnings = remove_private_info(raw_style_text)
    warnings.extend(privacy_warnings)
    if memory_context:
        warnings.append(f"Session Memory: {session_id}의 최근 {len(memory_context)}턴 대화 이력을 참고했습니다.")

    start_name = str(merged_payload.get("start_name", "")).strip()
    if start_name not in START_POINTS:
        return None, ["지원하지 않는 출발지입니다."]

    duration = str(merged_payload.get("duration", "당일치기")).strip()
    transport = str(merged_payload.get("transport", "대중교통")).strip()
    weather = str(merged_payload.get("weather", "맑음")).strip()

    if not style_text:
        return None, ["원하는 여행 스타일을 자연어로 입력해주세요."]

    try:
        budget = int(str(merged_payload.get("budget", "0")).replace(",", "").strip())
    except ValueError:
        return None, ["예산은 숫자로 입력해주세요."]

    if budget <= 0:
        return None, ["예산은 1원 이상으로 입력해주세요."]

    if transport not in ALLOWED_TRANSPORTS:
        transport = "대중교통"
        warnings.append("이동수단은 대중교통과 도보 중심만 지원해 대중교통 기준으로 계산했습니다.")

    if duration not in ALLOWED_DURATIONS:
        duration = "당일치기"
        warnings.append("여행 기간은 당일치기만 지원해 당일치기 기준으로 계산했습니다.")

    state: AgentState = {
        "session_id": session_id,
        "start_name": start_name,
        "start_point": START_POINTS[start_name],
        "duration": duration,
        "style_text": style_text,
        "selected_keywords": selected_keywords,
        "transport": transport,
        "budget": budget,
        "weather": weather,
        "tags": [],
        "warnings": warnings,
        "memory_context": memory_context[-6:],
    }
    return state, []


def style_analysis_tool(style_text: str) -> list[str]:
    normalized = style_text.replace(",", " ")
    tags = []
    for tag, keywords in STYLE_KEYWORDS.items():
        if tag in normalized or any(keyword in normalized for keyword in keywords):
            tags.append(tag)
    indoor_selected = "실내" in tags
    outdoor_selected = "실외" in tags
    if indoor_selected == outdoor_selected:
        tags = [tag for tag in tags if tag not in {"실내", "실외"}]
    return tags or ["카페", "맛집", "사진"]


def merge_tags(primary: list[str], supplemental: list[str]) -> list[str]:
    merged: list[str] = []
    for tag in primary + supplemental:
        if tag in STYLE_KEYWORDS and tag not in merged:
            merged.append(tag)
    indoor_selected = "실내" in merged
    outdoor_selected = "실외" in merged
    if indoor_selected == outdoor_selected:
        merged = [tag for tag in merged if tag not in {"실내", "실외"}]
    return merged or ["카페", "맛집", "사진"]


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


INTENT_TERMS = APP_CONFIG["intent_terms"]


def local_intent_analysis(style_text: str, selected_keywords: list[str] | None = None) -> tuple[list[str], dict[str, Any]]:
    selected_keywords = selected_keywords or []
    text = re.sub(r"\s+", " ", style_text).strip()
    tags = merge_tags(style_analysis_tool(text), selected_keywords)

    flags = {key: has_any(text, terms) for key, terms in INTENT_TERMS.items()}
    prefer_tourism_over_food = (
        has_any(text, ("관광지 위주", "관광 중심", "볼거리 위주", "맛집보다 관광", "먹는 것보다 관광"))
        or ("맛집보다" in text and has_any(text, ("관광", "볼거리", "사진")))
    )
    if flags["needs_quiet"]:
        tags = merge_tags(tags, ["산책", "자연", "카페"])
    if flags["wants_family"]:
        tags = merge_tags(tags, ["산책", "자연", "카페", "사진"])

    must_have: list[str] = []
    avoid: list[str] = []
    if selected_keywords:
        must_have.extend(selected_keywords)
    if flags["needs_external_review"]:
        must_have.append("외부 리뷰 근거")
    if flags["needs_short_route"]:
        must_have.append("짧은 이동거리")
    if flags["wants_family"]:
        must_have.append("가족 적합")
    if flags["wants_popular"]:
        must_have.append("인기/최신성")
    if flags["needs_quiet"]:
        must_have.append("조용한 장소")
        avoid.extend(["혼잡", "사람 많은 곳"])
    if prefer_tourism_over_food:
        must_have.append("관광지 우선")
        tags = merge_tags(["사진", "산책", "자연", "역사"], [tag for tag in tags if tag != "맛집"])

    tool_plan = ["자연어 의도 분석 Tool", "RAG 후보 검색 Tool"]
    if flags["needs_external_review"]:
        tool_plan.append("Tavily 리뷰 근거 검색 Tool")
    tool_plan.append("거리/동선 계산 Tool")
    tool_plan.append("최종 동선 추천")

    intent = {
        "tags": tags,
        "must_have": list(dict.fromkeys(must_have)),
        "avoid": list(dict.fromkeys(avoid)),
        "pace": "느긋" if flags["needs_quiet"] or flags["needs_short_route"] else "보통",
        **flags,
        "prefer_tourism_over_food": prefer_tourism_over_food,
        "tool_plan": tool_plan,
        "intent_summary": "자연어 요청을 로컬 규칙으로 분석하고 키워드는 보조 조건으로 병합했습니다.",
    }
    return tags, intent


def merge_intent_with_local(
    parsed: dict[str, Any],
    local_intent: dict[str, Any],
    selected_keywords: list[str],
) -> dict[str, Any]:
    merged = {**local_intent, **parsed}
    merged["tags"] = merge_tags(
        [str(tag).strip() for tag in parsed.get("tags", [])],
        [str(tag).strip() for tag in local_intent.get("tags", [])] + selected_keywords,
    )
    merged["must_have"] = list(
        dict.fromkeys(
            [str(item).strip() for item in local_intent.get("must_have", []) + parsed.get("must_have", []) if str(item).strip()]
        )
    )
    merged["avoid"] = list(
        dict.fromkeys(
            [str(item).strip() for item in local_intent.get("avoid", []) + parsed.get("avoid", []) if str(item).strip()]
        )
    )
    for key in (
        "needs_external_review",
        "needs_short_route",
        "needs_quiet",
        "wants_family",
        "wants_popular",
        "prefer_tourism_over_food",
    ):
        merged[key] = bool(local_intent.get(key) or parsed.get(key))

    tool_plan = ["자연어 의도 분석 Tool", "RAG 후보 검색 Tool"]
    if merged["needs_external_review"]:
        tool_plan.append("Tavily 리뷰 근거 검색 Tool")
    tool_plan.append("거리/동선 계산 Tool")
    tool_plan.append("최종 동선 추천")
    merged["tool_plan"] = tool_plan
    return merged


def llm_intent_tool(state: AgentState) -> tuple[list[str], dict[str, Any], str]:
    fallback_tags, local_intent = local_intent_analysis(
        state["style_text"],
        state.get("selected_keywords", []),
    )
    if not env_flag("ENABLE_INTENT_LLM", default=False):
        return (
            fallback_tags,
            local_intent,
            "로컬 자연어 의도 해석",
        )

    format_instructions = (
        INTENT_OUTPUT_PARSER.get_format_instructions()
        if INTENT_OUTPUT_PARSER
        else (
            "JSON 스키마: {\"tags\":[\"...\"],\"must_have\":[\"...\"],\"avoid\":[\"...\"],"
            "\"pace\":\"느긋|보통|빡빡\",\"needs_external_review\":true,"
            "\"needs_short_route\":false,\"needs_quiet\":false,\"wants_family\":false,"
            "\"wants_popular\":false,\"prefer_tourism_over_food\":false,"
            "\"tool_plan\":[\"...\"],\"intent_summary\":\"짧은 한국어 요약\"}."
        )
    )
    prompt = INTENT_ANALYSIS_PROMPT.format(
        style_keywords=list(STYLE_KEYWORDS.keys()),
        format_instructions=format_instructions,
        memory_context=json.dumps(state.get("memory_context", []), ensure_ascii=False),
        style_text=state["style_text"],
        selected_keywords=json.dumps(state.get("selected_keywords", []), ensure_ascii=False),
        local_intent=json.dumps(local_intent, ensure_ascii=False),
        duration=state["duration"],
        transport=state["transport"],
        weather=state["weather"],
        budget=state["budget"],
    )
    provider = model_provider()
    parsed, mode = provider.json_tool(prompt, parser=INTENT_OUTPUT_PARSER)
    if not parsed:
        return fallback_tags, local_intent, f"의도 해석 Fallback: {mode}"

    try:
        intent_model = TravelIntent.model_validate(parsed)
        parsed = intent_model.model_dump()
    except (ValidationError, ValueError) as error:
        fallback_intent = {**local_intent, "intent_summary": f"Pydantic OutputParser 검증 실패: {error}"}
        return fallback_tags, fallback_intent, "의도 해석 OutputParser Fallback"

    parsed = merge_intent_with_local(parsed, local_intent, state.get("selected_keywords", []))
    return parsed["tags"], parsed, f"{provider.name} LLM 자연어 의도 해석"


def weather_filter_score(place: dict[str, Any], weather: str) -> float:
    rule = SCORE_RULES["weather_filter_score"].get(weather, SCORE_RULES["weather_filter_score"]["default"])
    key = "indoor" if place["indoor"] else "outdoor"
    return float(rule[key])


def indoor_preference_score(place: dict[str, Any], tags: list[str]) -> float:
    rule = SCORE_RULES["indoor_preference_score"]
    if rule["matching_tag"] not in tags:
        return 0
    return float(rule["indoor"] if place["indoor"] else rule["outdoor"])


def outdoor_preference_score(place: dict[str, Any], tags: list[str]) -> float:
    rule = SCORE_RULES["outdoor_preference_score"]
    if rule["matching_tag"] not in tags:
        return 0
    return float(rule["indoor"] if place["indoor"] else rule["outdoor"])


def start_proximity_score(distance_km: float, transport: str) -> float:
    rule = SCORE_RULES["start_proximity_score"].get(
        transport,
        SCORE_RULES["start_proximity_score"]["default"],
    )
    for range_rule in rule["ranges"]:
        if distance_km <= float(range_rule["max_km"]):
            return float(range_rule["score"])
    return max(float(rule["fallback_min_score"]), distance_km * float(rule["fallback_multiplier"]))


CATEGORY_SCORE_RULES = APP_CONFIG["category_score_rules"]


def category_preference_score(place: dict[str, Any], tags: list[str]) -> float:
    category = place["category"]
    score = 0.0
    place_tags = set(place["tags"])
    if "동물" in tags:
        if "동물" in place_tags or any(word in place["name"] for word in ["동물원", "반려견", "애견"]):
            score += 7.0
        elif category in {"공원", "관광지"}:
            score += 0.8
    if "동물먹이" in tags:
        haystack = f"{place['name']} {place.get('kakao_category', '')} {' '.join(place_tags)}"
        if "동물원" in haystack:
            score += 12.0
        elif category == "동물체험" and not any(word in haystack for word in ["반려견", "애견", "펫", "강아지", "댕댕", "멍뭉", "퍼피"]):
            score += 7.0
    score += sum(
        float(rule["score"])
        for rule in CATEGORY_SCORE_RULES
        if rule["tag"] in tags and category in rule["categories"]
    )
    if "노래방" in tags:
        if "노래방" in place_tags:
            score += 8.0
        elif category == "놀거리":
            score += 1.5
    if "PC방" in tags:
        if "PC방" in place_tags or "피시방" in place_tags:
            score += 8.0
        elif category == "놀거리":
            score += 1.5
    if "보드게임" in tags:
        if "보드게임" in place_tags:
            score += 8.0
        elif category == "놀거리":
            score += 1.5
    return score


def intent_rule_matches(rule: dict[str, Any], category: str, role: str, tags: set[str], text: str) -> bool:
    checks: list[bool] = []
    if rule.get("category_any"):
        checks.append(category in set(rule["category_any"]))
    if rule.get("role_any"):
        checks.append(role in set(rule["role_any"]))
    if rule.get("tag_any"):
        checks.append(bool(tags.intersection(rule["tag_any"])))
    if rule.get("text_any"):
        checks.append(any(word in text for word in rule["text_any"]))
    return any(checks) if checks else True


def intent_adjustment_score(place: dict[str, Any], intent: dict[str, Any], distance_km: float = 0.0) -> float:
    score = 0.0
    category = str(place.get("category", ""))
    role = str(place.get("role") or place_role_for_category(category))
    tags = set(place.get("tags", []))
    text = f"{place.get('name', '')} {category} {' '.join(tags)} {place.get('address', '')}"

    for rule in SCORE_RULES["intent_adjustment_score"]:
        if not intent.get(rule["intent"]):
            continue
        if rule.get("distance_ranges"):
            for distance_rule in rule["distance_ranges"]:
                if distance_km <= float(distance_rule["max_km"]):
                    score += float(distance_rule["score"])
                    break
            else:
                score += max(
                    float(rule["fallback_min_score"]),
                    distance_km * float(rule["fallback_multiplier"]),
                )
            continue
        if intent_rule_matches(rule, category, role, tags, text):
            score += float(rule["score"])

    return score


def repeated_cafe_penalty(place: dict[str, Any]) -> float:
    if place["name"] in LOW_PRIORITY_REPEATED_CAFE_NAMES:
        return float(SCORE_PENALTIES["repeated_cafe_name_penalty"])
    if "목욕탕" in place["name"] and place["category"] in {"카페", "카페거리"}:
        return float(SCORE_PENALTIES["repeated_cafe_bathhouse_penalty"])
    return 0.0


def overused_chungbuk_place_penalty(place: dict[str, Any]) -> float:
    return float(SCORE_PENALTIES["overused_chungbuk_place_penalty"]) if place["name"] in OVERUSED_CHUNGBUK_PLACE_NAMES else 0.0


def diversity_adjusted_selection(
    ranked_places: list[dict[str, Any]],
    target_count: int,
    tags: list[str],
    transport: str,
    indoor_first: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    candidate_pool = ranked_places[: max(target_count * 5, 15)]

    while candidate_pool and len(selected) < target_count:
        best_place = max(
            candidate_pool,
            key=lambda place: diversity_candidate_score(place, selected, tags, transport, indoor_first),
        )
        selected.append(best_place)
        candidate_pool.remove(best_place)

    return selected


def preferred_categories(tags: list[str], weather: str) -> set[str]:
    categories: set[str] = set()
    if "동물" in tags:
        categories.update({"동물체험", "관광지", "공원", "카페", "카페거리", "놀거리"})
    if "동물먹이" in tags:
        categories.update({"동물체험", "관광지"})
    if "놀거리" in tags:
        categories.update({"놀거리", "카페"})
    if "노래방" in tags:
        categories.update({"놀거리"})
    if "PC방" in tags or "보드게임" in tags:
        categories.update({"놀거리"})
    if {"맛집", "카페", "쇼핑"}.intersection(tags):
        categories.update({"맛집", "카페", "시장", "상권", "카페거리", "놀거리"})
    if "역사" in tags or "실내" in tags or weather == "비":
        categories.update({"박물관", "카페", "카페거리", "상권", "맛집", "놀거리"})
    if {"산책", "자연", "사진"}.intersection(tags):
        categories.update({"공원", "관광지", "카페거리", "박물관"})
    return categories


def diversity_candidate_score(
    place: dict[str, Any],
    selected: list[dict[str, Any]],
    tags: list[str],
    transport: str,
    indoor_first: bool,
) -> float:
    score = place["agent_score"]
    same_category_count = sum(1 for item in selected if item["category"] == place["category"])
    score -= same_category_count * 2.4

    for item in selected:
        distance = haversine_km(place, item)
        if distance < 1.0:
            score -= 2.2
        elif distance < 2.5:
            score -= 1.0
        if transport == "도보 중심" and distance > 4.0:
            score -= min(5.0, (distance - 4.0) * 0.9)

    selected_categories = {item["category"] for item in selected}
    if selected and place["category"] not in selected_categories:
        score += 1.4

    if "카페" in tags and not any(item["category"] in {"카페", "카페거리", "상권"} for item in selected):
        score += 2.0 if place["category"] in {"카페", "카페거리", "상권"} else 0
    if "맛집" in tags and not any(item["category"] in {"맛집", "시장", "상권", "카페거리"} for item in selected):
        score += 2.0 if place["category"] in {"맛집", "시장", "상권", "카페거리"} else 0
    if {"산책", "자연"}.intersection(tags) and not any(item["category"] in {"공원", "관광지"} for item in selected):
        score += 1.8 if place["category"] in {"공원", "관광지"} else 0
    if "역사" in tags and not any(item["category"] == "박물관" for item in selected):
        score += 1.8 if place["category"] == "박물관" else 0
    if "동물" in tags and not any("동물" in item["tags"] for item in selected):
        score += 6.0 if "동물" in place["tags"] else 0

    if indoor_first and not place["indoor"]:
        score -= 4.0

    return score


def transport_filtered_places(
    ranked_places: list[dict[str, Any]],
    target_count: int,
    transport: str,
) -> list[dict[str, Any]]:
    if transport == "도보 중심":
        # 도보 중심은 출발지 기준으로도 15분 이내 후보만 우선 사용
        nearby = [
            place
            for place in ranked_places
            if place.get("start_distance_km", 999) <= WALK_ONLY_MAX_KM
        ]
        return nearby

    elif transport == "대중교통":
        # 대중교통은 너무 먼 곳만 제한하고, 15분 초과는 distance_tool에서 버스 처리
        for max_distance in (10, 15, 22):
            reachable = [
                place
                for place in ranked_places
                if place["start_distance_km"] <= max_distance
            ]
            if len(reachable) >= target_count:
                return reachable

    return ranked_places


SLOT_RULES = APP_CONFIG["slot_rules"]
DEFAULT_SLOTS = APP_CONFIG["default_slots"]
DAY_TRIP_FALLBACK_SLOTS = APP_CONFIG["day_trip_fallback_slots"]


def itinerary_slots(duration: str, tags: list[str] | None = None) -> list[dict[str, Any]]:
    tag_set = set(tags or [])
    slots: list[dict[str, Any]] = []

    def add_slot(role: str, label: str) -> None:
        if not any(slot["role"] == role and slot["label"] == label for slot in slots):
            slots.append({"day": 1, "role": role, "label": label})

    for rule in SLOT_RULES:
        if set(rule["terms"]).intersection(tag_set):
            for role, label in rule["slots"]:
                add_slot(role, label)

    if {"실내", "비", "우천", "더움", "추움"}.intersection(tag_set) and not {"산책", "자연", "공원"}.intersection(tag_set):
        slots = [slot for slot in slots if slot["role"] != "walk"]

    if not slots:
        slots = [dict(slot) for slot in DEFAULT_SLOTS]

    has_meal = any(slot["role"] == "meal" for slot in slots)
    has_cafe = any(slot["role"] == "cafe" for slot in slots)

    if len(slots) == 1:
        if not has_meal:
            slots.insert(0, {"day": 1, "role": "meal", "label": "밥집"})
        if not has_cafe:
            slots.append({"day": 1, "role": "cafe", "label": "카페"})

    elif len(slots) == 2:
        if not has_meal and "맛집" not in tag_set:
            slots.insert(0, {"day": 1, "role": "meal", "label": "밥집"})
        elif not has_cafe and "카페" not in tag_set:
            slots.append({"day": 1, "role": "cafe", "label": "카페"})

    if duration == "당일치기":
        slots.extend(dict(slot) for slot in DAY_TRIP_FALLBACK_SLOTS[: max(0, 5 - len(slots))])

    return slots[:5]


def role_matches(place: dict[str, Any], role: str) -> bool:
    place_role = place.get("role") or place_role_for_category(place["category"])
    if role == "meal":
        return place_role == "meal" or place["category"] in {"맛집", "시장", "상권", "카페거리"}
    if role == "cafe":
        return place_role == "cafe" or place["category"] in {"카페", "카페거리", "상권"}
    if role == "walk":
        return (
            place_role == "walk"
            or place["category"] in {"공원", "관광지"}
            or "산책" in place["tags"]
        )
    if role == "activity":
        return place_role in {"activity", "walk"} or place["category"] in {"관광지", "박물관", "공원", "상권", "놀거리", "동물체험"} or "동물" in place["tags"]
    return place_role == role


def enforce_preferred_tag(
    selected: list[dict[str, Any]],
    ranked_places: list[dict[str, Any]],
    tags: list[str],
) -> list[dict[str, Any]]:
    if "동물" not in tags or any("동물" in place["tags"] for place in selected):
        return selected

    selected_names = {place["name"] for place in selected}
    animal_candidate = next(
        (place for place in ranked_places if "동물" in place["tags"] and place["name"] not in selected_names),
        None,
    )
    if not animal_candidate:
        return selected

    replace_index = next(
        (
            index
            for index, place in enumerate(selected)
            if place.get("slot_role") in {"activity", "walk", "cafe"} and not place["matched_tags"]
        ),
        len(selected) - 1,
    )
    slot_role = selected[replace_index].get("slot_role", animal_candidate.get("role"))
    slot_label = selected[replace_index].get("slot_label", role_label(slot_role))
    day = selected[replace_index].get("day", 1)
    selected[replace_index] = {
        **animal_candidate,
        "matched_tags": sorted(set(tags).intersection(animal_candidate["tags"])),
        "slot_role": slot_role,
        "slot_label": slot_label,
        "day": day,
        "agent_score": round(animal_candidate.get("agent_score", animal_candidate["score"]) + 3.0, 2),
    }
    return selected


def slot_candidate_score(
    place: dict[str, Any],
    slot: dict[str, Any],
    current: dict[str, float],
    selected: list[dict[str, Any]],
    tags: list[str],
    transport: str,
    per_place_budget: float,
    weather: str,
) -> float:
    matched_tags = sorted(set(tags).intersection(place["tags"]))
    current_distance = haversine_km(current, place)
    duplicate_category_count = sum(1 for item in selected if item["category"] == place["category"])
    budget_penalty = float(SCORE_PENALTIES["budget_penalty"]) if place["cost"] > per_place_budget and place["cost"] > 0 else 0

    place_role = place.get("role") or place_role_for_category(place["category"])

    # 기본 role 점수
    if place_role == slot["role"]:
        role_bonus = 14.0
    elif role_matches(place, slot["role"]):
        role_bonus = 2.0
    else:
        role_bonus = -5.0

    # slot label에 맞는 세부 장소 보너스
    slot_label = slot.get("label", "")
    place_text = (
        f"{place.get('name', '')} "
        f"{place.get('category', '')} "
        f"{' '.join(place.get('tags', []))} "
        f"{place.get('address', '')} "
        f"{place.get('kakao_category', '')}"
    )

    for rule in SCORE_RULES["slot_label_bonus_rules"]:
        if not any(word in slot_label for word in rule["label_terms"]):
            continue
        place_matches = not rule.get("place_terms") or any(word in place_text for word in rule["place_terms"])
        source_matches = not rule.get("source_equals") or place.get("source") == rule["source_equals"]
        cost_matches = not rule.get("cost_lte") or place.get("cost", 0) <= int(rule["cost_lte"])
        if place_matches and source_matches and cost_matches:
            role_bonus += float(rule["score"])

    return (
        place["score"]
        + role_bonus
        + len(matched_tags) * 1.8
        + weather_filter_score(place, weather)
        + indoor_preference_score(place, tags)
        + outdoor_preference_score(place, tags)
        + category_preference_score(place, tags)
        + start_proximity_score(place.get("start_distance_km", current_distance), transport)
        - route_candidate_cost(current, place, selected, transport) * 0.35
        - duplicate_category_count * 1.2
        - budget_penalty
        - repeated_cafe_penalty(place)
        - overused_chungbuk_place_penalty(place)
        - pet_care_penalty(place, tags)
    )


def select_place_for_slot(
    places: list[dict[str, Any]],
    slot: dict[str, Any],
    current: dict[str, float],
    selected: list[dict[str, Any]],
    tags: list[str],
    transport: str,
    per_place_budget: float,
    weather: str,
) -> dict[str, Any] | None:
    selected_names = {place["name"] for place in selected}

    candidates = [
        place
        for place in places
        if place["name"] not in selected_names
        and role_matches(place, slot["role"])
        and not duplicates_activity_type(place, selected)
        and (
            transport != "도보 중심"
            or haversine_km(current, place) <= WALK_ONLY_MAX_KM
        )
    ]

    if not candidates and slot["role"] == "walk":
        candidates = [
            place
            for place in places
            if place["name"] not in selected_names
            and role_matches(place, "activity")
            and not duplicates_activity_type(place, selected)
            and (
                transport != "도보 중심"
                or haversine_km(current, place) <= WALK_ONLY_MAX_KM
            )
        ]

    if not candidates:
        return None

    chosen = max(
        candidates,
        key=lambda place: slot_candidate_score(
            place,
            slot,
            current,
            selected,
            tags,
            transport,
            per_place_budget,
            weather,
        ),
    )

    matched_tags = sorted(set(tags).intersection(chosen["tags"]))

    return {
        **chosen,
        "matched_tags": matched_tags,
        "slot_role": slot["role"],
        "slot_label": slot["label"],
        "day": slot["day"],
        "agent_score": round(
            slot_candidate_score(
                chosen,
                slot,
                current,
                selected,
                tags,
                transport,
                per_place_budget,
                weather,
            ),
            2,
        ),
    }


def calculate_agent_score(
    place: dict[str, Any],
    tags: list[str],
    weather: str,
    transport: str,
    per_place_budget: float,
    start_distance: float,
    intent: dict[str, Any] | None = None,
    matched_tags: list[str] | None = None,
    quality: float | None = None,
) -> float:
    matched_tags = matched_tags if matched_tags is not None else sorted(set(tags).intersection(place["tags"]))
    quality = float(quality if quality is not None else place.get("quality_score") or place_quality_score(place))
    budget_penalty = (
        float(SCORE_PENALTIES["budget_penalty"])
        if place["cost"] > per_place_budget and place["cost"] > 0
        else 0.0
    )
    return (
        place["score"]
        + min(quality, float(AGENT_SCORE_FORMULA["quality_cap"])) * float(AGENT_SCORE_FORMULA["quality_weight"])
        + len(matched_tags) * float(AGENT_SCORE_FORMULA["matched_tag_weight"])
        + weather_filter_score(place, weather)
        + indoor_preference_score(place, tags)
        + outdoor_preference_score(place, tags)
        + category_preference_score(place, tags)
        + start_proximity_score(start_distance, transport)
        + intent_adjustment_score(place, intent or {}, start_distance)
        + float(place.get("review_boost", 0))
        - budget_penalty
        - repeated_cafe_penalty(place)
        - overused_chungbuk_place_penalty(place)
        - pet_care_penalty(place, tags)
    )


def recommendation_tool(
    tags: list[str],
    budget: int,
    weather: str,
    duration: str,
    start_point: dict[str, float],
    transport: str,
    excluded_place_names: set[str] | None = None,
    intent: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    places = []
    slots = itinerary_slots(duration, tags)
    excluded_name_keys = {
        normalize_place_name_for_memory(name)
        for name in (excluded_place_names or set())
        if normalize_place_name_for_memory(name)
    }

    # 도보 중심이어도 당일치기 기본 슬롯 5개 유지
    MIN_DAY_TRIP_SLOTS = 5


    target_count = len(slots)
    per_place_budget = budget / target_count
    place_db = load_place_db()

    for place in place_db:
        if is_memory_excluded_place(place, excluded_name_keys):
            continue
        quality = float(place.get("quality_score") or place_quality_score(place))
        if quality < MIN_RECOMMENDATION_QUALITY:
            continue
        matched_tags = sorted(set(tags).intersection(place["tags"]))
        start_distance = haversine_km(start_point, place)
        score = calculate_agent_score(
            place=place,
            tags=tags,
            weather=weather,
            transport=transport,
            per_place_budget=per_place_budget,
            start_distance=start_distance,
            intent=intent,
            matched_tags=matched_tags,
            quality=quality,
        )
        places.append(
            {
                **place,
                "matched_tags": matched_tags,
                "start_distance_km": round(start_distance, 2),
                "role": place.get("role") or place_role_for_category(place["category"]),
                "quality_score": quality,
                "agent_score": round(score, 2),
            }
        )

    ranked_places = sorted(
        places,
        key=lambda item: (item["agent_score"], -item["start_distance_km"]),
        reverse=True,
    )
    full_ranked_places = ranked_places[:]
    if "동물먹이" in tags:
        ranked_places = [
            place
            for place in ranked_places
            if not ((place.get("role") == "activity" or place["category"] == "동물체험") and is_pet_care_place(place))
        ]
    preferred = preferred_categories(tags, weather)
    preferred_ranked_places = [place for place in ranked_places if place["category"] in preferred]
    slot_roles = {slot["role"] for slot in slots}
    keeps_slot_coverage = all(
        any(role_matches(place, role) for place in preferred_ranked_places)
        for role in slot_roles
    )
    if len(preferred_ranked_places) >= target_count and keeps_slot_coverage:
        ranked_places = preferred_ranked_places
    all_ranked_places = ranked_places
    ranked_places = transport_filtered_places(ranked_places, target_count, transport)
    if "동물" in tags:
        ranked_names = {place["name"] for place in ranked_places}
        ranked_places.extend(
            place
            for place in all_ranked_places
            if "동물" in place["tags"] and place["name"] not in ranked_names
        )

    selected: list[dict[str, Any]] = []
    current = start_point
    for slot in slots:
        chosen = select_place_for_slot(
            ranked_places,
            slot,
            current,
            selected,
            tags,
            transport,
            per_place_budget,
            weather,
        )
        if chosen:
            selected.append(chosen)
            current = chosen

    selected = enforce_preferred_tag(selected, ranked_places, tags)
    return fill_route_to_minimum(
        selected,
        transport_filtered_places(full_ranked_places, target_count, transport),
        tags,
        budget,
        start_point,
        transport,
        min_count=5 if duration == "당일치기" else len(slots),
    )


def candidate_lookup_tool(
    state: AgentState,
    tags: list[str],
    intent: dict[str, Any] | None = None,
    max_candidates: int = 48,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slots = itinerary_slots(state["duration"], tags)
    used_place_names = used_place_names_from_memory(state.get("memory_context", []))
    used_place_names.discard(state["start_name"])
    used_name_keys = {
        normalize_place_name_for_memory(name)
        for name in used_place_names
        if normalize_place_name_for_memory(name)
    }

    # 도보 중심이어도 당일치기 기본 슬롯 5개를 유지
    MIN_DAY_TRIP_SLOTS = 5

    if state["duration"] == "당일치기" and len(slots) < MIN_DAY_TRIP_SLOTS:
        slots = [
            {"day": 1, "role": "meal", "label": "밥집"},
            {"day": 1, "role": "activity", "label": "놀거리"},
            {"day": 1, "role": "cafe", "label": "카페"},
            {"day": 1, "role": "activity", "label": "놀거리"},
            {"day": 1, "role": "walk", "label": "산책/마무리"},
        ]

    places: list[dict[str, Any]] = []
    place_db = load_place_db()
    target_count = len(slots)
    per_place_budget = max(1, state["budget"] / max(1, target_count))

    for place in place_db:
        if is_memory_excluded_place(place, used_name_keys):
            continue
        quality = float(place.get("quality_score") or place_quality_score(place))
        if quality < MIN_RECOMMENDATION_QUALITY:
            continue
        if "동물먹이" in tags and ((place.get("role") == "activity" or place["category"] == "동물체험") and is_pet_care_place(place)):
            continue

        matched_tags = sorted(set(tags).intersection(place["tags"]))
        start_distance = haversine_km(state["start_point"], place)
        score = calculate_agent_score(
            place=place,
            tags=tags,
            weather=state["weather"],
            transport=state["transport"],
            per_place_budget=per_place_budget,
            start_distance=start_distance,
            intent=intent,
            matched_tags=matched_tags,
            quality=quality,
        )
        places.append(
            {
                **place,
                "matched_tags": matched_tags,
                "start_distance_km": round(start_distance, 2),
                "role": place.get("role") or place_role_for_category(place["category"]),
                "quality_score": quality,
                "agent_score": round(score, 2),
            }
        )

    ranked = sorted(
        places,
        key=lambda item: (
            item["agent_score"],
            item.get("quality_score", 0),
            -item["start_distance_km"],
        ),
        reverse=True,
    )
    ranked = transport_filtered_places(ranked, target_count, state["transport"])
    ranked = diversify_ranked_places(ranked)

    selected: list[dict[str, Any]] = []
    selected_names: set[str] = set()
    for slot in slots:
        slot_candidates = [place for place in ranked if place["name"] not in selected_names and role_matches(place, slot["role"])]
        for place in slot_candidates[:8]:
            selected.append(place)
            selected_names.add(place["name"])

    for place in ranked:
        if len(selected) >= max_candidates:
            break
        if place["name"] not in selected_names:
            selected.append(place)
            selected_names.add(place["name"])

    return slots, selected[:max_candidates]


def tavily_search(query: str, max_results: int = 4) -> list[dict[str, Any]]:
    api_key = TAVILY_API_KEY
    if not api_key:
        logger.warning("Tavily search skipped: TAVILY_API_KEY is not configured")
        return []
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    raw = http_post_json(TAVILY_SEARCH_API_URL, payload, timeout=5, headers=headers)
    data = json.loads(raw.decode("utf-8"))
    results = data.get("results", []) if isinstance(data, dict) else []
    return [result for result in results if isinstance(result, dict)]


def review_evidence_score(results: list[dict[str, Any]]) -> tuple[float, list[str]]:
    evidence_terms = ("리뷰", "후기", "평점", "별점", "방문", "인기", "추천", "맛집", "카페", "블로그")
    positive_terms = ("좋", "많", "유명", "인기", "추천", "만족", "재방문", "핫플", "깔끔", "친절")
    score = 0.0
    snippets: list[str] = []
    for result in results:
        title = str(result.get("title", ""))
        content = str(result.get("content", ""))
        url = str(result.get("url", ""))
        text = f"{title} {content}"
        evidence_hits = sum(1 for term in evidence_terms if term in text)
        positive_hits = sum(1 for term in positive_terms if term in text)
        if evidence_hits:
            score += 0.8 + min(1.2, evidence_hits * 0.25) + min(0.8, positive_hits * 0.2)
            if title:
                snippets.append(title[:80])
            elif url:
                snippets.append(url[:80])
    return round(min(score, 5.0), 2), snippets[:3]


def tavily_review_boost_tool(
    state: AgentState,
    intent: dict[str, Any],
    candidates: list[dict[str, Any]],
    max_places: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not intent.get("needs_external_review"):
        return candidates, ["Tavily 리뷰 근거 검색 Tool: 자연어 의도상 외부 리뷰 근거가 필요하지 않아 생략"]
    if not TAVILY_API_KEY:
        state["warnings"].append("Tavily 리뷰 근거 검색 Tool: TAVILY_API_KEY가 없어 외부 검색 보정 없이 진행했습니다.")
        return candidates, ["Tavily 리뷰 근거 검색 Tool: API 키 없음으로 생략"]

    if max_places is None:
        try:
            max_places = max(1, min(12, int(os.getenv("TAVILY_MAX_PLACES", "6"))))
        except ValueError:
            max_places = 6

    boosted: list[dict[str, Any]] = []
    searched = 0
    for place in candidates:
        updated = dict(place)
        if searched < max_places:
            query = f"청주 {place['name']} 리뷰 후기 평점 인기"
            try:
                results = tavily_search(query)
                review_boost, evidence = review_evidence_score(results)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError) as error:
                logger.warning("Tavily review search failed: %s", error)
                state["warnings"].append(f"Tavily 리뷰 근거 검색 Tool 실패: {error}")
                return candidates, ["Tavily 리뷰 근거 검색 Tool: 호출 실패로 기존 후보 점수 유지"]
            updated["review_boost"] = review_boost
            updated["review_evidence"] = evidence
            updated["agent_score"] = round(float(updated.get("agent_score", 0)) + review_boost, 2)
            searched += 1
        else:
            updated.setdefault("review_boost", 0.0)
            updated.setdefault("review_evidence", [])
        boosted.append(updated)

    boosted.sort(
        key=lambda item: (
            float(item.get("agent_score", 0)),
            float(item.get("review_boost", 0)),
            -float(item.get("start_distance_km", 0) or 0),
        ),
        reverse=True,
    )
    return boosted, [f"Tavily 리뷰 근거 검색 Tool: 후보 {searched}개 검색 후 review_boost를 agent_score에 반영"]


def place_area_signature(place: dict[str, Any]) -> str:
    address = str(place.get("address") or "")
    name = str(place.get("name") or "")
    text = f"{address} {name}"
    for pattern in (r"([가-힣0-9]+동)", r"([가-힣0-9]+읍)", r"([가-힣0-9]+면)", r"([가-힣0-9]+길)"):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    if "충북대" in text:
        return "충북대"
    if "터미널" in text or "가경" in text:
        return "가경/터미널"
    if "성안길" in text:
        return "성안길"
    return "기타"


def diversify_ranked_places(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = ranked[:]
    diversified: list[dict[str, Any]] = []
    used_buckets: set[tuple[str, str]] = set()

    while remaining:
        next_index = 0
        for index, place in enumerate(remaining):
            bucket = (str(place.get("role") or place.get("category")), place_area_signature(place))
            if bucket not in used_buckets:
                next_index = index
                break
        place = remaining.pop(next_index)
        diversified.append(place)
        used_buckets.add((str(place.get("role") or place.get("category")), place_area_signature(place)))

    return diversified


def compact_candidate_for_llm(index: int, place: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": index,
        "name": place["name"],
        "category": place["category"],
        "role": place.get("role"),
        "area": place_area_signature(place),
        "cost": place["cost"],
        "indoor": place["indoor"],
        "tags": place["tags"],
        "matched_tags": place.get("matched_tags", []),
        "quality_score": place.get("quality_score"),
        "agent_score": place.get("agent_score"),
        "review_boost": place.get("review_boost", 0),
        "review_evidence": place.get("review_evidence", []),
        "start_distance_km": place.get("start_distance_km"),
        "address": place.get("address"),
        "source": place.get("source"),
        "phone_exists": bool(place.get("phone")),
        "map_exists": bool(place.get("map_url") or place.get("url")),
    }


def hydrate_llm_route(
    selected_items: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    slots: list[dict[str, Any]],
    tags: list[str],
) -> list[dict[str, Any]]:
    by_id = {index: place for index, place in enumerate(candidates, start=1)}
    route: list[dict[str, Any]] = []
    used_names: set[str] = set()

    for index, item in enumerate(selected_items):
        try:
            candidate_id = int(item.get("candidate_id"))
        except (TypeError, ValueError):
            continue
        place = by_id.get(candidate_id)
        if not place or place["name"] in used_names:
            continue
        slot = slots[min(index, len(slots) - 1)] if slots else {"role": place.get("role"), "label": role_label(place.get("role", "")), "day": 1}
        reason = str(item.get("reason", "")).strip()
        route.append(
            {
                **place,
                "matched_tags": sorted(set(tags).intersection(place["tags"])),
                "slot_role": str(item.get("slot_role") or slot["role"]),
                "slot_label": str(item.get("slot_label") or slot["label"]),
                "day": int(item.get("day") or slot["day"]),
                "llm_reason": reason,
                "agent_score": round(float(place.get("agent_score", place["score"])), 2),
            }
        )
        used_names.add(place["name"])
    return route


def can_use_next_place(
    current: dict[str, float],
    place: dict[str, Any],
    transport: str,
    remaining_budget: int,
) -> bool:
    if place["cost"] > remaining_budget:
        return False
    if transport == "도보 중심" and haversine_km(current, place) > WALK_ONLY_MAX_KM:
        return False
    return True


def constrained_candidate_score(
    current: dict[str, float],
    place: dict[str, Any],
    remaining_budget: int,
    transport: str,
    intent: dict[str, Any] | None = None,
) -> float:
    distance = haversine_km(current, place)
    affordability_bonus = max(0.0, (remaining_budget - place["cost"]) / max(remaining_budget, 1)) * 3.0
    score = float(place.get("agent_score", place.get("score", 0))) + affordability_bonus
    distance_weight = 1.0 if intent and intent.get("needs_short_route") else 0.35
    score -= distance * (2.0 if transport == "도보 중심" else distance_weight)
    score -= place["cost"] / 10000
    return score


def with_slot_metadata(place: dict[str, Any], slot: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    return {
        **place,
        "matched_tags": sorted(set(tags).intersection(place["tags"])),
        "slot_role": slot["role"],
        "slot_label": slot["label"],
        "day": slot["day"],
        "agent_score": round(float(place.get("agent_score", place.get("score", 0))), 2),
    }


def fill_route_to_minimum(
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    tags: list[str],
    budget: int,
    start_point: dict[str, float],
    transport: str,
    min_count: int = 5,
    intent: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if len(selected) >= min_count:
        return selected

    filled = selected[:]
    used_names = {place["name"] for place in filled}
    remaining_budget = budget - sum(int(place.get("cost") or 0) for place in filled)
    current = filled[-1] if filled else start_point
    pool = [place for place in candidates if place["name"] not in used_names]

    while len(filled) < min_count:
        available = [
            place
            for place in pool
            if place["name"] not in used_names
            and can_use_next_place(current, place, transport, remaining_budget)
            and not duplicates_activity_type(place, filled)
        ]
        if not available:
            available = [
                place
                for place in pool
                if place["name"] not in used_names
                and can_use_next_place(current, place, transport, remaining_budget)
            ]
        if not available:
            break

        slot = {
            "day": 1,
            "role": available[0].get("role") or place_role_for_category(available[0]["category"]),
            "label": role_label(available[0].get("role") or place_role_for_category(available[0]["category"])),
        }
        chosen = max(
            available,
            key=lambda place: constrained_candidate_score(
                current,
                place,
                remaining_budget,
                transport,
                intent,
            ),
        )
        slot["role"] = chosen.get("role") or place_role_for_category(chosen["category"])
        slot["label"] = role_label(slot["role"])
        filled.append(with_slot_metadata(chosen, slot, tags))
        used_names.add(chosen["name"])
        remaining_budget -= int(chosen.get("cost") or 0)
        current = chosen

    return filled


def enforce_day_trip_constraints(
    state: AgentState,
    route: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    intent: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    slots = itinerary_slots("당일치기", state.get("tags", []))

    excluded_place_names = used_place_names_from_memory(state.get("memory_context", []))
    excluded_place_names.discard(state["start_name"])
    excluded_name_keys = {
        normalize_place_name_for_memory(name)
        for name in excluded_place_names
        if normalize_place_name_for_memory(name)
    }

    pool_by_name: dict[str, dict[str, Any]] = {}
    for place in candidates + route:
        if is_memory_excluded_place(place, excluded_name_keys):
            continue
        pool_by_name[place["name"]] = place
    pool = list(pool_by_name.values())

    selected: list[dict[str, Any]] = []
    used_names: set[str] = set()
    current = state["start_point"]
    remaining_budget = state["budget"]

    for index, slot in enumerate(slots):
        original = route[index] if index < len(route) else None
        chosen: dict[str, Any] | None = None

        if (
            original
            and original["name"] not in used_names
            and not is_memory_excluded_place(original, excluded_name_keys)
            and role_matches(original, slot["role"])
            and not duplicates_activity_type(original, selected)
            and can_use_next_place(current, original, state["transport"], remaining_budget)
        ):
            chosen = with_slot_metadata(original, slot, state["tags"])
        else:
            slot_candidates = [
                place
                for place in pool
                if place["name"] not in used_names
                and role_matches(place, slot["role"])
                and not duplicates_activity_type(place, selected)
                and can_use_next_place(current, place, state["transport"], remaining_budget)
            ]
            if slot_candidates:
                chosen = with_slot_metadata(
                    max(
                        slot_candidates,
                        key=lambda place: constrained_candidate_score(
                            current,
                            place,
                            remaining_budget,
                            state["transport"],
                            intent,
                        ),
                    ),
                    slot,
                    state["tags"],
                )

        if chosen is None:
            continue
        selected.append(chosen)
        used_names.add(chosen["name"])
        remaining_budget -= chosen["cost"]
        current = chosen

    return fill_route_to_minimum(
        selected,
        pool,
        state["tags"],
        state["budget"],
        state["start_point"],
        state["transport"],
        min_count=5,
        intent=intent,
    )



def append_missing_keyword_warnings(state: AgentState, route: list[dict[str, Any]]) -> None:
    precise_keywords = {"동물", "노래방", "PC방", "보드게임"}
    selected = set(state.get("selected_keywords") or state["tags"])
    route_tags = set()
    for place in route:
        route_tags.update(place.get("tags", []))
        signature = activity_signature(place)
        if signature:
            route_tags.add(signature)

    missing = sorted(keyword for keyword in selected.intersection(precise_keywords) if keyword not in route_tags)
    if missing:
        state["warnings"].append(
            f"충북대 기준 이동/예산 조건 안에서 {', '.join(missing)} 후보를 찾지 못해 다른 가까운 장소로 대체했습니다."
        )


def llm_route_planner_tool(
    state: AgentState,
    tags: list[str],
    intent: dict[str, Any],
    slots: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    retrieved_documents: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    if not candidates:
        return [], "후보 없음 Fallback", {}
    
    excluded_place_names = sorted(used_place_names_from_memory(state.get("memory_context", [])))
    excluded_place_names = [name for name in excluded_place_names if name != state["start_name"]]

    compact_candidates = [
        compact_candidate_for_llm(index, place)
        for index, place in enumerate(candidates, start=1)
    ]
    prompt = ROUTE_PLANNER_PROMPT.format(
        style_text=state["style_text"],
        selected_keywords=json.dumps(state.get("selected_keywords", []), ensure_ascii=False),
        memory_context=json.dumps(state.get("memory_context", []), ensure_ascii=False),
        excluded_place_names=json.dumps(excluded_place_names, ensure_ascii=False),
        intent=json.dumps(intent, ensure_ascii=False),
        tags=tags,
        start_name=state["start_name"],
        duration=state["duration"],
        transport=state["transport"],
        budget=state["budget"],
        slots=json.dumps(slots, ensure_ascii=False),
        retrieved_documents=json.dumps(retrieved_documents or [], ensure_ascii=False),
        compact_candidates=json.dumps(compact_candidates, ensure_ascii=False),
    )
    provider = model_provider()
    parsed, mode = provider.json_tool(prompt, parser=ROUTE_DECISION_OUTPUT_PARSER, temperature=0.25, timeout=25)
    if not parsed:
        logger.warning("LLM route planner failed: %s", mode)
        return [], f"LLM 동선 선택 Fallback: {mode}", {}

    selected_items = parsed.get("selected", [])
    if not isinstance(selected_items, list):
        logger.warning("LLM route planner returned invalid JSON shape")
        return [], "LLM 동선 선택 JSON 형식 오류 Fallback", parsed
    route = hydrate_llm_route(selected_items, candidates, slots, tags)
    return route, f"{provider.name} LLM 동선 선택", parsed


def balance_categories(
    selected: list[dict[str, Any]],
    all_places: list[dict[str, Any]],
    target_count: int,
    tags: list[str],
    indoor_first: bool = False,
) -> list[dict[str, Any]]:
    if not selected:
        return []
    if indoor_first:
        return selected[:target_count]

    wants_food = bool({"맛집", "카페", "쇼핑"}.intersection(tags))
    wants_culture = "역사" in tags
    wants_walk = bool({"산책", "자연", "사진"}.intersection(tags))

    has_food = any(place["category"] in ["시장", "상권", "카페거리"] for place in selected)
    has_culture = any(place["category"] in ["박물관"] for place in selected)
    has_walk = any("산책" in place["tags"] or place["category"] == "공원" for place in selected)

    required_categories = []
    if wants_food and not has_food:
        required_categories.append(["시장", "상권", "카페거리"])
    if wants_culture and not has_culture:
        required_categories.append(["박물관"])
    if wants_walk and not has_walk:
        required_categories.append(["공원", "관광지"])

    names = {place["name"] for place in selected}
    for categories in required_categories:
        candidate = next(
            (place for place in sorted(
                all_places,
                key=lambda item: (item["agent_score"], -item.get("start_distance_km", 0)),
                reverse=True,
            )
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


def route_candidate_cost(
    current: dict[str, float],
    place: dict[str, Any],
    route: list[dict[str, Any]],
    transport: str,
) -> float:
    distance = haversine_km(current, place)
    if transport == "도보 중심":
        return distance + max(0, distance - 2.5) * 1.5
    return distance + max(0, distance - 8.0) * 0.25


def optimize_route_tool(
    start_point: dict[str, float],
    places: list[dict[str, Any]],
    transport: str,
) -> list[dict[str, Any]]:
    route = []
    remaining = places[:]
    current = start_point

    while remaining:
        next_place = min(
            remaining,
            key=lambda place: route_candidate_cost(current, place, route, transport),
        )
        route.append(next_place)
        remaining.remove(next_place)
        current = next_place

    return route


def distance_tool(
    start_point: dict[str, float],
    route: list[dict[str, Any]],
    transport: str,
    duration: str,
) -> list[dict[str, Any]]:
    speed = TRANSPORT_SPEED_KMH[transport]
    legs = []
    current_name = "출발지"
    current = start_point

    for place in route:
        distance = haversine_km(current, place)
        walk_minutes = max(5, round(distance / TRANSPORT_SPEED_KMH["도보 중심"] * 60))
        mode = "walk"
        transit_options: list[dict[str, Any]] = []
        transit_error = None

        if transport == "도보 중심":
            # 도보 중심은 15분 이내만 허용
            move_minutes = walk_minutes
            mode = "walk"
            if walk_minutes > WALK_ONLY_MAX_MINUTES:
                transit_error = "도보 중심 조건에서는 15분 초과 이동 구간이므로 추천 후보에서 제외되어야 합니다."

        elif transport == "대중교통":
            if walk_minutes <= WALK_ONLY_MAX_MINUTES:
                # 가까운 곳은 도보
                move_minutes = walk_minutes
                mode = "walk"
            else:
                # 15분 넘으면 무조건 대중교통 시도
                transit_options, transit_error = odsay_transit_options(current, place)

                logger.info(
                    "ODsay route checked: %s -> %s options=%s error=%s",
                    current_name,
                    place["name"],
                    len(transit_options),
                    transit_error,
                )
                
                if transit_options:
                    mode = "transit"
                    move_minutes = max(5, int(transit_options[0]["total_time"]))
                else:
                    # 대중교통 경로를 못 찾았을 때만 임시 fallback
                    # 여기서도 "도보 처리"라고 쓰지 말고 "경로 확인 필요"로 표시
                    mode = "transit"
                    move_minutes = walk_minutes
                    transit_error = transit_error or "대중교통 경로 확인 필요"

        else:
            move_minutes = walk_minutes
            mode = "walk"

        legs.append(
            {
                "from": current_name,
                "to": place["name"],
                "distance_km": round(distance, 2),
                "move_minutes": move_minutes,
                "walk_minutes": walk_minutes,
                "mode": mode,
                "transit_options": transit_options,
                "transit_error": transit_error,
            }
        )
        current_name = place["name"]
        current = place

    return legs


def json_dumps_for_query(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def parse_json_argument(raw: str | dict[str, Any] | list[Any]) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    return json.loads(text)


def state_from_tool_input(input_json: str | dict[str, Any]) -> tuple[AgentState | None, list[str]]:
    payload = parse_json_argument(input_json)
    if not isinstance(payload, dict):
        return None, ["JSON 객체를 입력해야 합니다."]

    if {"style_text", "start_point", "budget", "transport", "duration", "weather"}.issubset(payload.keys()):
        state = dict(payload)
        state.setdefault("session_id", normalize_session_id(state.get("session_id")))
        state.setdefault("start_name", "충북대")
        state.setdefault("selected_keywords", [])
        state.setdefault("tags", [])
        state.setdefault("warnings", [])
        state.setdefault("memory_context", [])
        return state, []

    if isinstance(payload.get("payload"), dict):
        payload = payload["payload"]
    return validate_and_normalize(payload)


def quality_reasons_for_route(
    recommended: list[dict[str, Any]],
    slots: list[dict[str, Any]],
) -> list[str]:
    required_roles = {slot.get("role") for slot in slots if slot.get("role")}
    selected_roles = {
        place.get("slot_role") or place.get("role")
        for place in recommended
        if place.get("slot_role") or place.get("role")
    }
    missing_roles = sorted(required_roles - selected_roles)
    target_count = min(len(slots), 5) if slots else 3
    average_quality = (
        sum(float(place.get("quality_score") or place.get("score") or 0) for place in recommended) / len(recommended)
        if recommended
        else 0
    )
    reasons: list[str] = []
    if not recommended:
        reasons.append("LLM이 추천 장소를 선택하지 못했습니다.")
    if len(recommended) < max(1, target_count - 1):
        reasons.append(f"추천 장소 수가 부족합니다. selected={len(recommended)}, target={target_count}")
    if missing_roles:
        reasons.append(f"일정 구성에 필요한 role이 부족합니다: {', '.join(missing_roles)}")
    if recommended and average_quality < 3.2:
        reasons.append(f"추천 평균 품질 점수가 낮습니다: {average_quality:.2f}")
    return reasons


def select_trip_candidates_core(
    state: AgentState,
    tags: list[str] | None = None,
    intent: dict[str, Any] | None = None,
    retry_count: int = 0,
) -> dict[str, Any]:
    if tags is None or intent is None:
        tags, intent, intent_mode = llm_intent_tool(state)
    else:
        intent_mode = "Tool 입력 의도 사용"
    state["tags"] = tags

    try:
        slots, candidates = candidate_lookup_tool(
            state,
            tags,
            intent,
            max_candidates=72 if retry_count else 48,
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"{error} 장소 데이터를 사용하려면 KAKAO_REST_API_KEY를 설정하거나 /api/places/sync로 장소 캐시를 다시 생성하세요."
        ) from error

    tool_events: list[str] = [intent_mode]
    candidates, review_events = tavily_review_boost_tool(state, intent, candidates)
    tool_events.extend(review_events)

    user_query = " ".join([state["style_text"], " ".join(tags), json_dumps_for_query(intent)])
    vector_documents = retrieve_relevant_place_documents(user_query, max_documents=10 if retry_count else 8)
    vector_names = {
        str(document.get("metadata", {}).get("name"))
        for document in vector_documents
        if document.get("metadata", {}).get("name")
    }
    if vector_names:
        for candidate in candidates:
            if candidate.get("name") in vector_names:
                candidate["agent_score"] = round(float(candidate.get("agent_score", 0)) + 1.5, 2)
        candidates.sort(key=lambda item: float(item.get("agent_score", 0)), reverse=True)
        tool_events.append(f"FAISS VectorStore RAG Retriever: 관련 장소 문서 {len(vector_documents)}개를 후보 점수에 반영")
    else:
        tool_events.append("FAISS VectorStore RAG Retriever: 관련 문서 없음 또는 임베딩 backend 없음")

    recommended, planner_mode, planner_decision = llm_route_planner_tool(
        state,
        tags,
        intent,
        slots,
        candidates,
        vector_documents + candidate_documents_to_context(candidates),
    )
    quality_reasons = quality_reasons_for_route(recommended, slots)
    if quality_reasons:
        state["warnings"].append("추천 품질 검사: " + " / ".join(quality_reasons))
        excluded_place_names = used_place_names_from_memory(state.get("memory_context", []))
        excluded_place_names.discard(state["start_name"])
        recommended = recommendation_tool(
            tags,
            state["budget"],
            state["weather"],
            state["duration"],
            state["start_point"],
            state["transport"],
            excluded_place_names,
            intent,
        )
        planner_mode = planner_mode if recommended and "Fallback" not in planner_mode else "규칙 기반 Fallback"
        planner_decision = {
            "decision_summary": "품질 검사 결과를 반영해 규칙 기반 후보 생성/점수화 결과로 보정했습니다.",
            **(planner_decision or {}),
        }

    constrained = enforce_day_trip_constraints(state, recommended, candidates, intent)
    if constrained != recommended:
        state["warnings"].append("당일치기 제약에 맞춰 예산 초과 또는 도보 15분 초과 후보를 제외했습니다.")
    recommended = constrained
    append_missing_keyword_warnings(state, recommended)

    return {
        "state": state,
        "tags": tags,
        "intent": intent,
        "slots": slots,
        "candidates": candidates,
        "recommended": recommended,
        "planner_mode": planner_mode,
        "planner_decision": planner_decision,
        "retrieved_documents": vector_documents + candidate_documents_to_context(candidates),
        "tool_events": tool_events,
        "quality_reasons": quality_reasons,
    }


if tool:
    @tool
    def analyze_travel_intent(input_json: str) -> str:
        """사용자 여행 요청 JSON을 분석해 tags, must_have, avoid, pace, needs_external_review 등을 반환한다."""
        state, errors = state_from_tool_input(input_json)
        if errors or state is None:
            return json.dumps({"errors": errors}, ensure_ascii=False)
        tags, intent, mode = llm_intent_tool(state)
        return json.dumps({"tags": tags, **intent, "mode": mode}, ensure_ascii=False)


    @tool
    def rag_place_retriever(query: str) -> str:
        """청주 장소 DB를 FAISS VectorStore RAG로 검색한다."""
        documents = retrieve_relevant_place_documents(query, max_documents=8)
        return json.dumps(documents, ensure_ascii=False)


    @tool
    def kakao_place_search(query: str) -> str:
        """카카오 Local API로 청주 장소 후보를 검색한다."""
        if not KAKAO_REST_API_KEY:
            return "KAKAO_REST_API_KEY 없음"
        places = search_kakao_places_by_query(query)
        return json.dumps(places[:10], ensure_ascii=False)


    @tool
    def tavily_review_search(query: str) -> str:
        """장소의 리뷰, 후기, 인기 근거를 Tavily Search로 검색한다."""
        if not TAVILY_API_KEY:
            return "TAVILY_API_KEY 없음"
        results = tavily_search(query, max_results=4)
        return json.dumps(results, ensure_ascii=False)


    @tool
    def select_trip_candidates(state_json: str) -> str:
        """후보 생성, 점수화, 리뷰 보정, 당일치기 제약 필터링을 수행해 추천 코스 후보를 반환한다."""
        payload = parse_json_argument(state_json)
        if not isinstance(payload, dict):
            return json.dumps({"errors": ["JSON 객체를 입력해야 합니다."]}, ensure_ascii=False)
        state, errors = state_from_tool_input(payload.get("state") or payload.get("payload") or payload)
        if errors or state is None:
            return json.dumps({"errors": errors}, ensure_ascii=False)
        tags = payload.get("tags")
        intent = payload.get("intent")
        retry_count = int(payload.get("retry_count") or 0)
        try:
            result = select_trip_candidates_core(
                state,
                tags=[str(tag) for tag in tags] if isinstance(tags, list) else None,
                intent=intent if isinstance(intent, dict) else None,
                retry_count=retry_count,
            )
        except RuntimeError as error:
            return json.dumps({"errors": [str(error)]}, ensure_ascii=False)
        compact = {
            key: result[key]
            for key in [
                "tags",
                "intent",
                "slots",
                "candidates",
                "recommended",
                "planner_mode",
                "planner_decision",
                "tool_events",
                "quality_reasons",
            ]
        }
        compact["warnings"] = result["state"].get("warnings", [])
        return json.dumps(compact, ensure_ascii=False)


    @tool
    def odsay_transit_route(route_json: str) -> str:
        """장소 간 대중교통 경로를 ODSay API로 확인한다."""
        payload = parse_json_argument(route_json)
        route = payload.get("route") if isinstance(payload, dict) else payload
        start_point = payload.get("start_point", START_POINTS["충북대"]) if isinstance(payload, dict) else START_POINTS["충북대"]
        duration = payload.get("duration", "당일치기") if isinstance(payload, dict) else "당일치기"
        legs = distance_tool(start_point, route, "대중교통", duration)
        return json.dumps(legs, ensure_ascii=False)


    @tool
    def calculate_route_distance(route_json: str) -> str:
        """장소 목록의 거리와 이동 시간을 계산한다."""
        payload = parse_json_argument(route_json)
        route = payload.get("route") if isinstance(payload, dict) else payload
        start_point = payload.get("start_point", START_POINTS["충북대"]) if isinstance(payload, dict) else START_POINTS["충북대"]
        transport = payload.get("transport", "대중교통") if isinstance(payload, dict) else "대중교통"
        duration = payload.get("duration", "당일치기") if isinstance(payload, dict) else "당일치기"
        legs = distance_tool(start_point, route, transport, duration)
        return json.dumps(legs, ensure_ascii=False)
else:
    analyze_travel_intent = None
    rag_place_retriever = None
    kakao_place_search = None
    tavily_review_search = None
    select_trip_candidates = None
    odsay_transit_route = None
    calculate_route_distance = None


AGENT_TOOLS = [
    candidate
    for candidate in [
        analyze_travel_intent,
        rag_place_retriever,
        kakao_place_search,
        tavily_review_search,
        select_trip_candidates,
        calculate_route_distance,
        odsay_transit_route,
    ]
    if candidate is not None
]


def build_tool_agent():
    if ChatOpenAI is None or create_react_agent is None or not AGENT_TOOLS or not os.getenv("OPENAI_API_KEY"):
        return None

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        timeout=25,
    )

    system_prompt = (
        "당신은 청주 여행 추천 TOOL_AGENT입니다. "
        "반드시 @tool로 제공된 도구를 직접 선택해 의도 분석, 후보 검색/생성, 리뷰 근거 확인, 거리 계산을 수행하세요. "
        "최종 추천 전에는 반드시 calculate_route_distance tool을 실행해야 합니다. "
        "최종 답변은 설명 문장 없이 TripPlan 스키마에 맞는 JSON 객체 하나만 출력하세요."
    )
    kwargs = {
        "model": llm,
        "tools": AGENT_TOOLS,
    }
    if SystemMessage is not None:
        kwargs["prompt"] = SystemMessage(content=system_prompt)
    try:
        return create_react_agent(**kwargs)
    except TypeError:
        kwargs.pop("prompt", None)
        return create_react_agent(**kwargs)


TOOL_AGENT = build_tool_agent()


def message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def message_name(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("name") or "")
    return str(getattr(message, "name", "") or "")


def message_tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = message.get("tool_calls", []) if isinstance(message, dict) else getattr(message, "tool_calls", [])
    return calls if isinstance(calls, list) else []


def build_tool_agent_prompt(state: AgentState, payload: dict[str, Any]) -> str:
    return (
        "청주 당일치기 추천을 실행하세요.\n"
        "규칙:\n"
        "1. analyze_travel_intent로 의도를 분석하세요.\n"
        "2. rag_place_retriever와 필요 시 kakao_place_search/tavily_review_search를 사용하세요.\n"
        "3. select_trip_candidates로 추천 후보 또는 코스 후보를 생성/점수화/필터링하세요.\n"
        "4. 최종 추천 전에는 반드시 calculate_route_distance를 실행하세요.\n"
        "5. 최종 응답은 반드시 TripPlan JSON 객체 하나만 출력하세요. Markdown, 설명, 코드블록은 금지합니다.\n"
        f"원본 payload: {json.dumps(payload, ensure_ascii=False)}\n"
        f"정규화된 state: {json.dumps(state, ensure_ascii=False)}\n"
        f"출발지: {state['start_name']}\n"
        f"이동수단: {state['transport']}\n"
        f"날씨: {state['weather']}\n"
        f"예산: {state['budget']}\n"
        f"사용자 요청: {state['style_text']}\n"
        f"TripPlan JSON 스키마 요약: {TripPlan.model_json_schema() if hasattr(TripPlan, 'model_json_schema') else 'summary, schedule, places, legs, warnings, agent_flow 포함'}"
    )


def extract_tool_agent_result(response: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    messages = response.get("messages", []) if isinstance(response, dict) else []
    events: list[str] = []
    retrieved_documents: list[dict[str, Any]] = []
    selected_tools: list[str] = []

    for message in messages:
        for call in message_tool_calls(message):
            name = str(call.get("name") or "")
            if name and name not in selected_tools:
                selected_tools.append(name)

        name = message_name(message)
        content = message_content(message)
        if not name:
            continue
        if name not in selected_tools:
            selected_tools.append(name)
        if name == "rag_place_retriever":
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                payload = []
            if isinstance(payload, list):
                retrieved_documents = [item for item in payload if isinstance(item, dict)]

    for name in selected_tools:
        events.append(f"ReAct Tool Agent 선택: {name}")
    if retrieved_documents:
        events.append(f"ReAct Tool Agent: RAG 문서 {len(retrieved_documents)}개를 후보 근거로 반영")
    return retrieved_documents, events


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_tool_agent_trip_plan(response: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    messages = response.get("messages", []) if isinstance(response, dict) else []
    events: list[str] = []
    for message in messages:
        for call in message_tool_calls(message):
            name = str(call.get("name") or "")
            if name:
                events.append(f"ReAct Tool Agent 선택: {name}")
    for message in reversed(messages):
        content = message_content(message)
        parsed = extract_json_object(content)
        if not parsed:
            continue
        try:
            validated = TripPlan.model_validate(parsed)
            return validated.model_dump(), events
        except (ValidationError, ValueError) as error:
            events.append(f"ReAct Tool Agent 최종 JSON 검증 실패: {error}")
            return parsed, events
    return None, events


def run_tool_agent(state: AgentState, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    if TOOL_AGENT is None:
        return None, ["ReAct Tool Agent: create_react_agent 사용 불가로 로컬 Tool fallback 실행"]
    if not os.getenv("OPENAI_API_KEY"):
        return None, ["ReAct Tool Agent: OPENAI_API_KEY 없음으로 로컬 Tool fallback 실행"]

    prompt = build_tool_agent_prompt(state, payload)
    try:
        response = TOOL_AGENT.invoke(
            {"messages": [("user", prompt)]},
        )
    except Exception as error:
        logger.warning("ReAct tool agent failed: %s", error)
        return None, [f"ReAct Tool Agent 실패: {error}"]
    return parse_tool_agent_trip_plan(response)


def build_schedule(
    route: list[dict[str, Any]],
    legs: list[dict[str, Any]],
    duration: str,
) -> list[dict[str, Any]]:
    start_hour = 10
    current_minutes = start_hour * 60
    schedule = []

    for index, place in enumerate(route):
        day = int(place.get("day", 1))
        current_minutes += legs[index]["move_minutes"]
        start = format_time(current_minutes)
        current_minutes += place["stay_minutes"]
        end = format_time(current_minutes)
        schedule.append(
            {
                "day": day,
                "time": f"{start} - {end}",
                "place": place["name"],
                "category": place.get("slot_label", place["category"]),
                "reason": build_reason(place),
                "cost": place["cost"],
                "indoor": place["indoor"],
            }
        )

    return schedule


def format_time(total_minutes: int) -> str:
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def build_reason(place: dict[str, Any]) -> str:
    if place.get("llm_reason"):
        return str(place["llm_reason"])
    templates = OUTPUT_TEMPLATES["reason"]
    matched = ", ".join(place["matched_tags"]) if place["matched_tags"] else templates["matched_default"]
    slot = place.get("slot_label")
    weather_note = templates["indoor_note"] if place["indoor"] else templates["outdoor_note"]
    if slot:
        return templates["with_slot"].format(slot=slot, matched=matched, weather_note=weather_note)
    return templates["default"].format(matched=matched, weather_note=weather_note)

def build_tool_decision(
    state: AgentState,
    intent: dict[str, Any] | None,
    planner_mode: str,
    tool_events: list[str] | None,
    legs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
    intent = intent or {}
    tool_events = tool_events or []

    def item(tool: str, used: bool, reason: str) -> dict[str, Any]:
        return {"tool": tool, "used": used, "reason": reason}

    tavily_used = any("Tavily" in event and "검색 후" in event for event in tool_events)
    tavily_skipped = any("Tavily" in event and "생략" in event for event in tool_events)
    transit_used = any(leg.get("mode") == "transit" and leg.get("transit_options") for leg in legs)
    transit_attempted = any(leg.get("mode") == "transit" for leg in legs)
    llm_route_used = "Fallback" not in planner_mode
    templates = OUTPUT_TEMPLATES["tool_decision"]

    return [
        item("입력 검증 Middleware", True, templates["input_validation"]),
        item("개인정보 제거 Middleware", True, templates["privacy"]),
        item("Session Memory", bool(state.get("memory_context")), templates["memory_used"] if state.get("memory_context") else templates["memory_unused"]),
        item("자연어 의도 분석 Tool", True, templates["intent_analysis"]),
        item("RAG 후보 검색 Tool", True, templates["rag"]),
        item("Tavily 리뷰 근거 검색 Tool", tavily_used, templates["tavily_used"] if tavily_used else templates["tavily_skipped"] if tavily_skipped or not intent.get("needs_external_review") else templates["tavily_failed"]),
        item("LLM 동선 선택 Tool", llm_route_used, templates["llm_route_used"] if llm_route_used else templates["llm_route_failed"]),
        item("규칙 기반 Fallback Tool", not llm_route_used, templates["fallback_used"] if not llm_route_used else templates["fallback_unused"]),
        item("거리/동선 계산 Tool", True, templates["distance"]),
        item("ODSay 대중교통 Tool", transit_used, templates["odsay_used"] if transit_used else templates["odsay_failed"] if transit_attempted else templates["odsay_skipped"]),
        item("OutputParser", True, templates["output_parser"]),
    ]


def build_rag_sources(route: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []

    for place in route:
        matched_tags = place.get("matched_tags", [])
        tag_text = ", ".join(matched_tags)

        if place.get("llm_reason"):
            reason = str(place["llm_reason"])
        elif tag_text:
            reason = f"사용자 조건({tag_text})과 일치하여 추천 후보로 선택되었습니다."
        else:
            reason = "거리, 예산, 장소 품질 점수를 기준으로 추천 후보로 선택되었습니다."

        sources.append(
            {
                "place_name": place.get("name"),
                "source": place.get("source", "로컬 장소 DB"),
                "category": place.get("category"),
                "role": place.get("role"),
                "matched_tags": matched_tags,
                "quality_score": place.get("quality_score"),
                "review_boost": place.get("review_boost", 0),
                "review_evidence": place.get("review_evidence", []),
                "reason": reason,
            }
        )

    return sources

def build_middleware_decision(state: AgentState) -> list[dict[str, Any]]:
    warnings = state.get("warnings", [])
    memory_context = state.get("memory_context", [])
    templates = OUTPUT_TEMPLATES["middleware_decision"]

    privacy_used = any("개인정보 제거 Middleware" in warning for warning in warnings)
    memory_used = bool(memory_context)

    return [
        {
            "name": "입력 검증 Middleware",
            "used": True,
            "reason": templates["input_validation"],
        },
        {
            "name": "개인정보 제거 Middleware",
            "used": privacy_used,
            "reason": templates["privacy_used"] if privacy_used else templates["privacy_unused"],
        },
        {
            "name": "Session Memory 병합",
            "used": memory_used,
            "reason": (
                templates["memory_used"].format(memory_turns=len(memory_context))
                if memory_used
                else templates["memory_unused"]
            ),
        },
    ]


BASE_AGENT_FLOW = APP_CONFIG["agent_flow"]


def build_agent_flow(intent: dict[str, Any] | None, tool_events: list[str] | None) -> list[str]:
    tool_plan = (intent or {}).get("tool_plan", [])
    plan_line = f"Agent Tool 선택 계획: {' → '.join(tool_plan)}" if tool_plan else "Agent Tool 선택 계획: 로컬 기본 계획"
    return [*BASE_AGENT_FLOW[:7], plan_line, *BASE_AGENT_FLOW[7:11], *(tool_events or []), *BASE_AGENT_FLOW[11:]]


def output_parser(
    state: AgentState,
    route: list[dict[str, Any]],
    legs: list[dict[str, Any]],
    planner_mode: str = "규칙 기반 추천",
    intent: dict[str, Any] | None = None,
    planner_decision: dict[str, Any] | None = None,
    rag_document_count: int = 0,
    tool_events: list[str] | None = None,
) -> dict[str, Any]:
    total_cost = sum(place["cost"] for place in route)
    total_distance = round(sum(leg["distance_km"] for leg in legs), 2)
    total_move_minutes = sum(leg["move_minutes"] for leg in legs)
    finish = "출발지 복귀 또는 성안길 카페 마무리"
    ai_comment, ai_mode = llm_response_tool(
        state,
        route,
        legs,
        total_cost,
        total_distance,
        total_move_minutes,
    )
    route_names = [state["start_name"]] + [place["name"] for place in route]

    raw_result = {
        "summary": {
            "title": "Cheongju Trip Agent 추천 결과",
            "tags": state["tags"],
            "route_text": " → ".join(route_names),
            "total_cost": total_cost,
            "budget": state["budget"],
            "total_distance": total_distance,
            "total_move_minutes": total_move_minutes,
            "finish": finish,
            "weather": state["weather"],
            "transport": state["transport"],
            "ai_comment": ai_comment,
            "ai_mode": ai_mode,
            "planner_mode": planner_mode,
            "intent": intent or {},
            "planner_decision_summary": (planner_decision or {}).get("decision_summary"),
            "session_id": state["session_id"],
            "memory_turns": len(state.get("memory_context", [])),
            "rag_document_count": rag_document_count,
        },
        "schedule": build_schedule(route, legs, state["duration"]),
        "places": [
            {
                "name": place["name"],
                "category": place["category"],
                "score": place["agent_score"],
                "tags": place["tags"],
                "matched_tags": place["matched_tags"],
                "slot_role": place.get("slot_role"),
                "slot_label": place.get("slot_label"),
                "day": place.get("day", 1),
                "cost": place["cost"],
                "indoor": place["indoor"],
                "lat": place["lat"],
                "lng": place["lng"],
                "start_distance_km": place.get("start_distance_km"),
                "address": place.get("address"),
                "phone": place.get("phone"),
                "url": place.get("url"),
                "map_url": place.get("map_url") or place.get("url") or kakao_map_search_url(place["name"], place.get("address", "")),
                "source": place.get("source"),
                "quality_score": place.get("quality_score"),
                "review_boost": place.get("review_boost", 0),
                "review_evidence": place.get("review_evidence", []),
                "llm_reason": place.get("llm_reason"),
            }
            for place in route
        ],
  "legs": legs,
        "warnings": state["warnings"],
        "agent_flow": build_agent_flow(intent, tool_events),
        "tool_decision": build_tool_decision(
            state=state,
            intent=intent,
            planner_mode=planner_mode,
            tool_events=tool_events,
            legs=legs,
        ),
        "rag_sources": build_rag_sources(route),
        "middleware_decision": build_middleware_decision(state),
    }
    try:
        parsed_result = TRIP_OUTPUT_PARSER.parse(json.dumps(raw_result, ensure_ascii=False)) if TRIP_OUTPUT_PARSER else TripPlan.model_validate(raw_result)
        return parsed_result.model_dump()
    except (ValidationError, ValueError) as error:
        raw_result["warnings"].append(f"TripPlan OutputParser 검증 실패: {error}")
        return raw_result


def build_llm_context(
    state: AgentState,
    route: list[dict[str, Any]],
    legs: list[dict[str, Any]],
    total_cost: int,
    total_distance: float,
    total_move_minutes: int,
) -> str:
    places = [
        {
            "name": place["name"],
            "category": place["category"],
            "matched_tags": place["matched_tags"],
            "cost": place["cost"],
            "indoor": place["indoor"],
            "agent_score": place["agent_score"],
            "review_boost": place.get("review_boost", 0),
            "review_evidence": place.get("review_evidence", []),
        }
        for place in route
    ]
    context = {
        "user_request": {
            "session_id": state["session_id"],
            "start": state["start_name"],
            "duration": state["duration"],
            "style_text": state["style_text"],
            "selected_keywords": state.get("selected_keywords", []),
            "tags": state["tags"],
            "transport": state["transport"],
            "budget": state["budget"],
            "weather": state["weather"],
            "memory_context": state.get("memory_context", []),
        },
        "optimized_route": [place["name"] for place in route],
        "places": places,
        "move_legs": legs,
        "total_cost": total_cost,
        "total_distance_km": total_distance,
        "total_move_minutes": total_move_minutes,
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def fast_final_comment(
    state: AgentState,
    route: list[dict[str, Any]],
    total_cost: int,
    total_distance: float,
    total_move_minutes: int,
) -> str:
    route_preview = " → ".join(place["name"] for place in route[:3])
    if len(route) > 3:
        route_preview += " → ..."
    return OUTPUT_TEMPLATES["fast_final_comment"].format(
        transport=state["transport"],
        route_preview=route_preview,
        total_cost=total_cost,
        total_move_minutes=total_move_minutes,
        total_distance=total_distance,
    )


def llm_response_tool(
    state: AgentState,
    route: list[dict[str, Any]],
    legs: list[dict[str, Any]],
    total_cost: int,
    total_distance: float,
    total_move_minutes: int,
) -> tuple[str, str]:
    if not env_flag("ENABLE_FINAL_LLM", default=False):
        return (
            fast_final_comment(state, route, total_cost, total_distance, total_move_minutes),
            "로컬 Agent 응답",
        )

    context = build_llm_context(
        state,
        route,
        legs,
        total_cost,
        total_distance,
        total_move_minutes,
    )
    prompt = FINAL_COMMENT_PROMPT.format(context=context)
    llm_text, mode = model_provider().final_comment(prompt, temperature=0.4, timeout=12)
    if llm_text:
        return llm_text, mode
    if mode.endswith("없음"):
        return (
            f"현재 실행 환경에는 {mode}이라 규칙 기반 Fallback으로 최종 문장을 생성했습니다. "
            "추천 결과는 Tool들이 만든 Context를 기반으로 구성되었습니다.",
            "규칙 기반 Fallback",
        )
    else:
        return (
            f"LLM 호출이 실패해 규칙 기반 Fallback으로 응답했습니다. 실패 사유: {mode}",
            "LLM 실패 Fallback",
        )

def build_plan_with_local_tools(
    state: AgentState,
    payload: dict[str, Any],
    tool_agent_events: list[str] | None = None,
) -> tuple[dict[str, Any], int]:
    try:
        selection = select_trip_candidates_core(state)
    except RuntimeError as error:
        logger.error("Place candidate selection failed: %s", error)
        return {
            "errors": [
                str(error),
                "장소 데이터를 사용하려면 KAKAO_REST_API_KEY를 설정하거나 /api/places/sync로 장소 캐시를 다시 생성하세요.",
            ]
        }, 503

    recommended = selection.get("recommended", [])
    if not recommended:
        return {
            "errors": [
                "예산과 이동 조건 안에서 추천 가능한 장소를 찾지 못했습니다.",
                "예산을 조금 늘리거나 여행 스타일을 넓게 입력해 주세요.",
            ]
        }, 422

    legs = distance_tool(state["start_point"], recommended, state["transport"], state["duration"])
    tool_events = [*(tool_agent_events or []), *selection.get("tool_events", [])]
    result = output_parser(
        state,
        recommended,
        legs,
        planner_mode=selection.get("planner_mode", "규칙 기반 추천"),
        intent=selection.get("intent", {}),
        planner_decision=selection.get("planner_decision", {}),
        rag_document_count=len(selection.get("retrieved_documents", [])),
        tool_events=tool_events,
    )
    remember_turn(state["session_id"], payload, result)
    return result, 200


def run_agent(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    state, errors = validate_and_normalize(payload)
    if errors or state is None:
        return {"errors": errors}, 400

    agent_result, tool_agent_events = run_tool_agent(state, payload)
    if agent_result:
        remember_turn(state["session_id"], payload, agent_result)
        return agent_result, 200

    state["warnings"].extend(tool_agent_events)
    return build_plan_with_local_tools(state, payload, tool_agent_events)


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

PUBLIC_DIR = BASE_DIR / "public"
app = FastAPI(title="Cheongju Trip Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/public", StaticFiles(directory=PUBLIC_DIR), name="public")


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    logger.info("%s %s", request.method, request.url.path)
    return await call_next(request)


@app.get("/")
async def index():
    return FileResponse(PUBLIC_DIR / "index.html")


@app.post("/api/recommend")
async def recommend(payload: dict[str, Any]):
    result, status = run_agent(payload)
    if status >= 400:
        return JSONResponse(result, status_code=status)
    return result


@app.post("/api/places/sync")
async def sync_places_route():
    try:
        places = sync_place_db()
        rag_document_count = rebuild_place_vectorstore()
    except RuntimeError as error:
        return JSONResponse({"errors": [str(error)]}, status_code=503)
    payload = json.loads(PLACE_DB_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"places": payload}
    return {
        "count": len(places),
        "db_path": str(PLACE_DB_PATH),
        "source": payload.get("source"),
        "source_counts": payload.get("source_counts", {}),
        "sync_errors": payload.get("sync_errors", []),
        "places": places,
        "rag_document_count": rag_document_count,
        "vectorstore": "FAISS",
    }


@app.get("/api/health")
async def health():
    places = load_place_db()
    return {"ok": True, "places": len(places), "agent": "openai" if os.getenv("OPENAI_API_KEY") else "local-fallback"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=int(os.getenv("PORT", "5000")), reload=True)
