from __future__ import annotations

import math
import json
import os
import re
import xml.etree.ElementTree as ET
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal, TypedDict

from flask import Flask, jsonify, render_template, request

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    ValidationError = ValueError

    def Field(default: Any = None, default_factory: Any = None, **_: Any) -> Any:
        return default_factory() if default_factory else default

    class BaseModel:
        def __init__(self, **data: Any) -> None:
            annotations = getattr(self, "__annotations__", {})
            for key, default in self.__class__.__dict__.items():
                if key.startswith("_") or callable(default):
                    continue
                if key in annotations and key not in data:
                    setattr(self, key, default)
            for key in annotations:
                if key in data:
                    setattr(self, key, data[key])

        @classmethod
        def model_validate(cls, data: dict[str, Any]) -> Any:
            return cls(**data)

        def model_dump(self) -> dict[str, Any]:
            return dict(self.__dict__)

try:
    from langchain_core.documents import Document
    from langchain_core.output_parsers import PydanticOutputParser
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
except ImportError:
    Document = None
    PydanticOutputParser = None
    MemorySaver = None
    END = "__end__"
    StateGraph = None


app = Flask(__name__)

# Gemini build.
# app.py와 같은 LangGraph/Memory/RAG/OutputParser 구조를 사용하고,
# LLM 호출부만 Gemini API로 교체한 버전입니다.


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLACE_DB_PATH = DATA_DIR / "cheongju_places.json"


def load_env_file() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()


CHUNGBUK_TOUR_API_URL = os.getenv(
    "CHUNGBUK_TOUR_API_URL",
    "https://tour.chungbuk.go.kr/openapi/tourInfo/attr.do",
)
TOUR_API_BASE_URL = "https://apis.data.go.kr/B551011/KorService1/areaBasedList1"
TOUR_API_KEY = os.getenv("TOUR_API_KEY") or os.getenv("TOURAPI_SERVICE_KEY")
TOUR_API_AREA_CODE = os.getenv("TOUR_API_AREA_CODE", "33")
TOUR_API_SIGUNGU_CODE = os.getenv("TOUR_API_SIGUNGU_CODE", "10")
KAKAO_LOCAL_API_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY") or os.getenv("KAKAO_API_KEY")
ODSAY_TRANSIT_API_URL = "https://api.odsay.com/v1/api/searchPubTransPathT"
ODSAY_API_KEY = os.getenv("ODSAY_API_KEY") or os.getenv("ODSAY_KEY")
ODSAY_CACHE_PATH = DATA_DIR / "odsay_transit_cache.json"
WALK_ONLY_MAX_MINUTES = 15
DAY_TRIP_MAX_TRANSIT_LEGS = 2
TRANSIT_OPTION_LIMIT = 4

KAKAO_KEYWORD_SEARCHES = [
    ("청주 성안길 맛집", "meal"),
    ("청주 성안길 카페", "cafe"),
    ("청주 성안길 노래방", "activity"),
    ("청주 운리단길 카페", "cafe"),
    ("청주 수암골 카페", "cafe"),
    ("청주 애견동반 카페", "cafe"),
    ("청주 동물 체험", "activity"),
    ("청주 반려견 놀이터", "activity"),
    ("청주 육거리시장 맛집", "meal"),
    ("청주 청주대 맛집", "meal"),
    ("청주 충북대 맛집", "meal"),
    ("청주 브런치", "meal"),
    ("청주 베이커리 카페", "cafe"),
    ("청주 체험 놀거리", "activity"),
    ("청주 실내 놀거리", "activity"),
    ("청주 호텔", "lodging"),
    ("오송역 호텔", "lodging"),
]

START_POINTS = {
    "청주고속버스터미널": {"lat": 36.6260, "lng": 127.4317},
    "청주역": {"lat": 36.6487, "lng": 127.3927},
    "충북대": {"lat": 36.6283, "lng": 127.4565},
    "오송역": {"lat": 36.6200, "lng": 127.3275},
}

STYLE_KEYWORDS = {
    "카페": ["카페", "커피", "디저트", "감성"],
    "맛집": ["맛집", "밥", "식사", "먹거리", "시장", "로컬"],
    "동물": ["동물", "반려견", "강아지", "고양이", "애견", "펫", "동물원"],
    "동물먹이": ["먹이", "먹여", "먹이주", "사료", "동물 먹"],
    "놀거리": ["놀거리", "노래방", "노래연습장", "노래연습실", "노래궁", "보드게임", "볼링", "방탈출", "오락", "게임"],
    "노래방": ["노래방", "노래연습장", "노래연습실", "노래궁", "코인노래"],
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

CHEONGJU_LAT_RANGE = (36.40, 36.75)
CHEONGJU_LNG_RANGE = (127.25, 127.65)

LOCAL_FALLBACK_PLACES = [
    {
        "name": "청주동물원",
        "category": "관광지",
        "role": "activity",
        "lat": 36.6499,
        "lng": 127.5136,
        "tags": ["동물", "사진", "산책", "자연"],
        "score": 4.45,
        "cost": 1000,
        "indoor": False,
        "stay_minutes": 90,
        "source": "로컬 보강 DB",
        "address": "충북 청주시 상당구 명암로 171",
    },
    {
        "name": "문암생태공원 반려견 놀이터",
        "category": "공원",
        "role": "walk",
        "lat": 36.6764,
        "lng": 127.4473,
        "tags": ["동물", "반려견", "산책", "자연", "사진"],
        "score": 4.35,
        "cost": 0,
        "indoor": False,
        "stay_minutes": 70,
        "source": "로컬 보강 DB",
        "address": "충북 청주시 흥덕구 무심서로 1097",
    },
]

LOW_PRIORITY_REPEATED_CAFE_NAMES = {"목욕탕 카페(카페목간)", "폴앤주비 카페"}
LOW_CONFIDENCE_PLACE_NAMES = {"레인데이"}
GENERIC_PLACE_NAMES = {
    "성안길",
    "수암골 카페거리",
    "운리단길 카페거리",
    "성안길 카페거리",
    "육거리종합시장",
    "서문시장 삼겹살거리",
}
GENERIC_PLACE_SUFFIXES = ("카페거리", "맛집거리", "삼겹살거리")
ANIMAL_EVIDENCE_TERMS = ("동물", "동물원", "반려견", "애견", "펫", "강아지", "고양이", "멍", "댕")
ACTIVITY_EVIDENCE_TERMS = (
    "놀거리",
    "노래방",
    "노래연습장",
    "노래연습실",
    "노래궁",
    "코인노래",
    "볼링",
    "보드게임",
    "방탈출",
    "VR",
    "피시방",
    "PC방",
    "게임",
)
MIN_RECOMMENDATION_QUALITY = 3.0
CAFE_EVIDENCE_TERMS = ("카페", "커피", "디저트", "베이커리", "스타벅스", "메가MGC", "메가커피", "투썸", "컴포즈", "빽다방", "이디야")
MEAL_EVIDENCE_TERMS = (
    "음식점",
    "한식",
    "일식",
    "중식",
    "양식",
    "분식",
    "식당",
    "맛집",
    "국밥",
    "순대",
    "돈까스",
    "돈가스",
    "버거",
    "냉면",
    "칼국수",
    "만두",
    "고기",
    "갈비",
    "치킨",
    "피자",
    "초밥",
    "카츠",
    "장어",
)


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
    if any(word in name_text for word in ["박물관", "미술관", "전시관", "기념관", "체험관", "교육원", "공예관"]):
        return "박물관"
    if any(word in name_text for word in ["카페", "커피"]):
        return "카페거리"
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
        tags.update(["자연", "산책"])
    if category in {"관광지", "동물체험"}:
        tags.update(["산책", "자연"])
    if category == "놀거리":
        tags.update(["실내"])
    if any(word in haystack for word in ["노래방", "노래연습장", "노래연습실", "노래궁", "코인노래"]):
        tags.update(["노래방", "실내"])
    return sorted(tags)


def place_role_for_category(category: str) -> str:
    if category in {"맛집", "식당", "시장"}:
        return "meal"
    if category in {"카페", "카페거리"}:
        return "cafe"
    if category == "숙소":
        return "lodging"
    if category in {"동물체험", "놀거리"}:
        return "activity"
    if category in {"공원"}:
        return "walk"
    return "activity"


def role_label(role: str) -> str:
    return {
        "meal": "밥집",
        "cafe": "카페",
        "activity": "놀거리",
        "walk": "산책/마무리",
        "lodging": "숙소",
    }.get(role, "장소")


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
    return any(
        word in haystack
        for word in ["반려견", "애견", "펫", "강아지", "댕댕", "멍뭉", "퍼피", "유치원", "놀이터", "스테이", "호텔", "미용", "헬스멍"]
    )


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

    score = 0.0
    if source == "카카오 Local API":
        score += 2.0
        if url:
            score += 1.5
        if phone:
            score += 1.5
        if address and "일대" not in address:
            score += 1.0
        if kakao_category:
            score += 1.0
        if role in {"meal", "cafe", "lodging"}:
            score += 0.7
        if role == "activity" and category in {"놀거리", "동물체험"}:
            score += 0.7
    elif source == "충청북도 관광명소정보 API":
        score += 3.0
        if address and "일대" not in address:
            score += 0.8
        if category in {"박물관", "공원", "관광지", "동물체험", "놀거리"}:
            score += 0.7
    elif source == "로컬 보강 DB":
        score += 3.5
        if address:
            score += 0.8
    else:
        score += 1.0

    if name in LOW_CONFIDENCE_PLACE_NAMES:
        score -= 10.0
    return round(max(0.0, score), 2)


def estimate_kakao_cost(role: str, category: str, name: str, query: str, kakao_category: str) -> int:
    haystack = f"{name} {query} {kakao_category} {category}"
    if role == "meal":
        return 12000
    if role == "cafe":
        return 7000
    if role == "lodging":
        return 40000
    if any(word in haystack for word in ["동물", "반려견", "애견", "펫", "강아지", "고양이", "멍", "댕"]):
        if "문암" in haystack and "놀이터" in haystack:
            return 0
        return 12000
    if any(word in haystack for word in ["노래방", "코인노래", "볼링", "보드게임", "방탈출", "VR", "피시방", "PC방", "게임"]):
        return 12000
    if any(word in haystack for word in ["체험", "공방", "클래스"]):
        return 15000
    if category in {"카페거리", "상권", "시장"}:
        return 10000
    return 5000


def pet_care_penalty(place: dict[str, Any], tags: list[str]) -> float:
    if "동물먹이" not in tags:
        return 0.0
    if is_pet_care_place(place):
        return 20.0
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
    elif role == "lodging":
        category = "숙소"
        tags = ["실내", "숙소", "교통"]
        stay_minutes = 0
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
    PLACE_DB_CACHE = sanitize_place_db(payload.get("places", []) + LOCAL_FALLBACK_PLACES)
    return PLACE_DB_CACHE


def sanitize_place_db(places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = []
    for place in places:
        name = str(place.get("name", "")).strip()
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
        if original_role in {"meal", "cafe", "lodging"}:
            role = original_role
            category = {"meal": "맛집", "cafe": "카페", "lodging": "숙소"}[role]
            tags = {
                "meal": ["맛집", "식사", "로컬"],
                "cafe": ["카페", "디저트", "사진"],
                "lodging": ["실내", "숙소", "교통"],
            }[role]
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
            "indoor": bool(place.get("indoor")) or "실내" in tags or category in {"박물관", "카페거리", "카페", "맛집", "상권", "숙소", "놀거리"},
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
        ("충청북도 관광명소정보 API", fetch_chungbuk_places),
        ("한국관광공사 TourAPI", fetch_tour_api_places),
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


def fetch_chungbuk_places() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page_unit = 100
    for page_index in range(1, 6):
        raw = http_get(
            CHUNGBUK_TOUR_API_URL,
            {"pageUnit": str(page_unit), "pageIndex": str(page_index)},
        )
        page_items = deep_find_items(parse_api_payload(raw))
        items.extend(page_items)
        if len(page_items) < page_unit:
            break
    return [place for item in items if (place := normalize_place(item, "충청북도 관광명소정보 API"))]


def fetch_tour_api_places() -> list[dict[str, Any]]:
    if not TOUR_API_KEY:
        return []
    params = {
        "serviceKey": TOUR_API_KEY,
        "MobileOS": "ETC",
        "MobileApp": "CheongjuTripAgent",
        "_type": "json",
        "numOfRows": "100",
        "pageNo": "1",
        "areaCode": TOUR_API_AREA_CODE,
        "sigunguCode": TOUR_API_SIGUNGU_CODE,
        "arrange": "A",
        "contentTypeId": "12",
    }
    raw = http_get(TOUR_API_BASE_URL, params)
    items = deep_find_items(parse_api_payload(raw))
    return [place for item in items if (place := normalize_place(item, "한국관광공사 TourAPI"))]


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
    load_env_file()
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
        return [], str(error)

    if isinstance(payload, dict) and payload.get("error"):
        errors = payload.get("error")
        if isinstance(errors, list) and errors:
            message = errors[0].get("message") if isinstance(errors[0], dict) else str(errors[0])
            code = errors[0].get("code") if isinstance(errors[0], dict) else ""
            return [], f"ODsay API 오류 {code}: {message}".strip()
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
    return [], "ODsay 대중교통 경로 없음"


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


class AgentState(TypedDict):
    session_id: str
    start_name: str
    start_point: dict[str, float]
    duration: str
    style_text: str
    transport: str
    budget: int
    weather: str
    tags: list[str]
    warnings: list[str]
    memory_context: list[dict[str, Any]]


class AgentGraphState(TypedDict, total=False):
    payload: dict[str, Any]
    state: AgentState
    errors: list[str]
    status: int
    tags: list[str]
    intent: dict[str, Any]
    intent_mode: str
    slots: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    retrieved_documents: list[dict[str, Any]]
    recommended: list[dict[str, Any]]
    planner_mode: str
    planner_decision: dict[str, Any]
    accommodation: dict[str, Any] | None
    legs: list[dict[str, Any]]
    result: dict[str, Any]


class TravelIntent(BaseModel):
    tags: list[str] = Field(default_factory=list)
    must_have: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    pace: Literal["느긋", "보통", "빡빡"] = "보통"
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
    lodging_cost: int = 0
    session_id: str
    memory_turns: int = 0
    rag_document_count: int = 0


class TripPlan(BaseModel):
    summary: TripSummary
    schedule: list[dict[str, Any]]
    accommodation: dict[str, Any] | None = None
    places: list[dict[str, Any]]
    legs: list[dict[str, Any]]
    warnings: list[str]
    agent_flow: list[str]


INTENT_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=TravelIntent) if PydanticOutputParser else None
TRIP_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=TripPlan) if PydanticOutputParser else None
GRAPH_MEMORY = MemorySaver() if MemorySaver else None
SESSION_STORE: dict[str, list[dict[str, Any]]] = {}
PLACE_DB_CACHE: list[dict[str, Any]] | None = None


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

    if "자동차" in style_text or "차로" in style_text:
        merged["transport"] = "자동차"
    elif "도보" in style_text or "걸어서" in style_text:
        merged["transport"] = "도보 중심"
    elif "대중교통" in style_text or "버스" in style_text:
        merged["transport"] = "대중교통"

    if "1박" in style_text or "하룻밤" in style_text:
        merged["duration"] = "1박 2일"
    elif "당일" in style_text:
        merged["duration"] = "당일치기"
    merged["session_id"] = session_id
    return merged, history


def remember_turn(session_id: str, payload: dict[str, Any], result: dict[str, Any]) -> None:
    history = SESSION_STORE.setdefault(session_id, [])
    history.append(
        {
            "payload": {
                key: payload.get(key)
                for key in ["session_id", "start_name", "duration", "style_text", "transport", "budget", "weather"]
            },
            "summary": result.get("summary", {}),
        }
    )
    del history[:-6]


def validate_and_normalize(payload: dict[str, Any]) -> tuple[AgentState | None, list[str]]:
    warnings: list[str] = []
    merged_payload, memory_context = merge_memory_payload(payload)
    session_id = normalize_session_id(merged_payload.get("session_id"))
    style_text, privacy_warnings = remove_private_info(str(merged_payload.get("style_text", "")).strip())
    warnings.extend(privacy_warnings)
    if memory_context:
        warnings.append(f"Session Memory: {session_id}의 최근 {len(memory_context)}턴 대화 이력을 참고했습니다.")

    start_name = str(merged_payload.get("start_name", "")).strip()
    if start_name not in START_POINTS:
        warnings.append(
            "Fallback Middleware: 입력한 출발지를 찾을 수 없어 청주고속버스터미널로 계산했습니다."
        )
        start_name = "청주고속버스터미널"

    duration = str(merged_payload.get("duration", "당일치기")).strip()
    transport = str(merged_payload.get("transport", "대중교통")).strip()
    weather = str(merged_payload.get("weather", "맑음")).strip()

    if not style_text:
        return None, ["여행 스타일을 하나 이상 입력해주세요."]

    try:
        budget = int(str(merged_payload.get("budget", "0")).replace(",", "").strip())
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

    state: AgentState = {
        "session_id": session_id,
        "start_name": start_name,
        "start_point": START_POINTS[start_name],
        "duration": duration,
        "style_text": style_text,
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
    return tags or ["카페", "맛집", "사진"]


def extract_json_payload(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def extract_gemini_text(data: dict[str, Any]) -> str:
    text_parts: list[str] = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if part.get("text"):
                text_parts.append(str(part["text"]))
    return "\n".join(text_parts).strip()


def gemini_json_tool(prompt: str, temperature: float = 0.2, timeout: int = 18) -> tuple[dict[str, Any] | None, str]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, "GEMINI_API_KEY 없음"

    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model, safe='-_.')}:generateContent"
        f"?key={urllib.parse.quote(api_key)}"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
        return None, str(error)

    text = extract_gemini_text(data)
    parsed = extract_json_payload(text)
    if parsed is None:
        return None, "LLM JSON 파싱 실패"
    return parsed, "Gemini LLM"


def llm_intent_tool(state: AgentState) -> tuple[list[str], dict[str, Any], str]:
    fallback_tags = style_analysis_tool(state["style_text"])
    format_instructions = (
        INTENT_OUTPUT_PARSER.get_format_instructions()
        if INTENT_OUTPUT_PARSER
        else "JSON 스키마: {\"tags\":[\"...\"],\"must_have\":[\"...\"],\"avoid\":[\"...\"],\"pace\":\"느긋|보통|빡빡\",\"intent_summary\":\"짧은 한국어 요약\"}."
    )
    prompt = (
        "너는 청주 여행 Agent의 의도 해석기다. 사용자의 자연어 입력을 여행 선호 태그와 제약으로 해석해라. "
        "반드시 JSON 객체만 출력해라. tags는 아래 허용 태그 중 필요한 것만 골라라: "
        f"{list(STYLE_KEYWORDS.keys())}. "
        f"{format_instructions}\n\n"
        f"이전 대화 메모리: {json.dumps(state.get('memory_context', []), ensure_ascii=False)}\n"
        f"사용자 입력: {state['style_text']}\n"
        f"기간: {state['duration']}, 이동수단: {state['transport']}, 날씨: {state['weather']}, 예산: {state['budget']}"
    )
    parsed, mode = gemini_json_tool(prompt)
    if not parsed:
        return fallback_tags, {"intent_summary": "규칙 기반 태그 분석", "must_have": [], "avoid": []}, f"의도 해석 Fallback: {mode}"

    try:
        intent_model = INTENT_OUTPUT_PARSER.parse(json.dumps(parsed, ensure_ascii=False)) if INTENT_OUTPUT_PARSER else TravelIntent.model_validate(parsed)
        parsed = intent_model.model_dump()
    except (ValidationError, ValueError) as error:
        return fallback_tags, {"intent_summary": f"Pydantic OutputParser 검증 실패: {error}", "must_have": [], "avoid": []}, "의도 해석 OutputParser Fallback"

    allowed = set(STYLE_KEYWORDS)
    tags = [str(tag).strip() for tag in parsed.get("tags", []) if str(tag).strip() in allowed]
    if not tags:
        tags = fallback_tags
    return tags, parsed, "Gemini LLM 의도 해석"


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


def start_proximity_score(distance_km: float, transport: str) -> float:
    if transport == "도보 중심":
        if distance_km <= 1.5:
            return 4.5
        if distance_km <= 3:
            return 3.0
        if distance_km <= 5:
            return 1.0
        if distance_km <= 8:
            return -2.0
        return max(-10.0, -distance_km * 0.55)

    if transport == "자동차":
        if distance_km <= 5:
            return 0.8
        if distance_km <= 20:
            return 1.4
        if distance_km <= 35:
            return 0.6
        return -1.0

    if distance_km <= 3:
        return 2.2
    if distance_km <= 7:
        return 1.4
    if distance_km <= 12:
        return 0.7
    if distance_km <= 20:
        return 0
    return max(-4.0, -distance_km * 0.12)


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
    if "카페" in tags and category in {"카페", "카페거리", "상권"}:
        score += 3.0
    if "맛집" in tags and category in {"맛집", "시장", "상권", "카페거리"}:
        score += 3.0
    if "쇼핑" in tags and category in {"상권", "시장"}:
        score += 2.6
    if "놀거리" in tags and category in {"놀거리", "동물체험"}:
        score += 4.0
    if "노래방" in tags:
        if "노래방" in place_tags:
            score += 8.0
        elif category == "놀거리":
            score += 1.5
    if "역사" in tags and category in {"박물관", "관광지"}:
        score += 2.6
    if "산책" in tags and category in {"공원", "관광지"}:
        score += 2.2
    if "자연" in tags and category in {"공원", "관광지"}:
        score += 2.2
    return score


def repeated_cafe_penalty(place: dict[str, Any]) -> float:
    if place["name"] in LOW_PRIORITY_REPEATED_CAFE_NAMES:
        return 4.0
    if "목욕탕" in place["name"] and place["category"] in {"카페", "카페거리"}:
        return 3.0
    return 0.0


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
    if {"맛집", "카페", "쇼핑"}.intersection(tags):
        categories.update({"맛집", "카페", "시장", "상권", "카페거리"})
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
        elif transport == "자동차" and distance > 8.0:
            score += 0.6

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
        for max_distance in (4, 6, 8, 12):
            nearby = [place for place in ranked_places if place["start_distance_km"] <= max_distance]
            if len(nearby) >= target_count:
                return nearby
    elif transport == "대중교통":
        for max_distance in (10, 15, 22):
            reachable = [place for place in ranked_places if place["start_distance_km"] <= max_distance]
            if len(reachable) >= target_count:
                return reachable
    return ranked_places


def itinerary_slots(duration: str) -> list[dict[str, Any]]:
    if duration == "1박 2일":
        return [
            {"day": 1, "role": "meal", "label": "점심"},
            {"day": 1, "role": "activity", "label": "놀거리"},
            {"day": 1, "role": "activity", "label": "놀거리"},
            {"day": 1, "role": "cafe", "label": "카페"},
            {"day": 1, "role": "meal", "label": "저녁"},
            {"day": 2, "role": "meal", "label": "아침/브런치"},
            {"day": 2, "role": "activity", "label": "놀거리"},
            {"day": 2, "role": "cafe", "label": "카페"},
        ]
    return [
        {"day": 1, "role": "meal", "label": "밥집"},
        {"day": 1, "role": "activity", "label": "놀거리"},
        {"day": 1, "role": "cafe", "label": "카페"},
        {"day": 1, "role": "activity", "label": "놀거리"},
        {"day": 1, "role": "walk", "label": "산책/마무리"},
    ]


def role_matches(place: dict[str, Any], role: str) -> bool:
    place_role = place.get("role") or place_role_for_category(place["category"])
    if role == "meal":
        return place_role == "meal" or place["category"] in {"맛집", "시장", "상권", "카페거리"}
    if role == "cafe":
        return place_role == "cafe" or place["category"] in {"카페", "카페거리", "상권"}
    if role == "walk":
        return place_role == "walk" or place["category"] in {"공원", "관광지"} or "산책" in place["tags"]
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
    budget_penalty = 1.0 if place["cost"] > per_place_budget and place["cost"] > 0 else 0
    place_role = place.get("role") or place_role_for_category(place["category"])
    if place_role == slot["role"]:
        role_bonus = 14.0
    elif role_matches(place, slot["role"]):
        role_bonus = 2.0
    else:
        role_bonus = -5.0

    return (
        place["score"]
        + role_bonus
        + len(matched_tags) * 1.8
        + weather_filter_score(place, weather)
        + indoor_preference_score(place, tags)
        + category_preference_score(place, tags)
        + start_proximity_score(place.get("start_distance_km", current_distance), transport)
        - route_candidate_cost(current, place, selected, transport) * 0.35
        - duplicate_category_count * 1.2
        - budget_penalty
        - repeated_cafe_penalty(place)
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
        if place["name"] not in selected_names and role_matches(place, slot["role"])
    ]
    if not candidates and slot["role"] == "walk":
        candidates = [
            place
            for place in places
            if place["name"] not in selected_names and role_matches(place, "activity")
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


def recommendation_tool(
    tags: list[str],
    budget: int,
    weather: str,
    duration: str,
    start_point: dict[str, float],
    transport: str,
) -> list[dict[str, Any]]:
    places = []
    slots = itinerary_slots(duration)
    if transport == "도보 중심" and duration == "당일치기":
        slots = slots[:4]
    target_count = len(slots)
    per_place_budget = budget / target_count
    place_db = load_place_db()

    for place in place_db:
        quality = float(place.get("quality_score") or place_quality_score(place))
        if quality < MIN_RECOMMENDATION_QUALITY:
            continue
        matched_tags = sorted(set(tags).intersection(place["tags"]))
        budget_penalty = 1.0 if place["cost"] > per_place_budget and place["cost"] > 0 else 0
        start_distance = haversine_km(start_point, place)
        score = (
            place["score"]
            + min(quality, 6.0) * 0.35
            + len(matched_tags) * 2
            + weather_filter_score(place, weather)
            + indoor_preference_score(place, tags)
            + category_preference_score(place, tags)
            + start_proximity_score(start_distance, transport)
            - budget_penalty
            - repeated_cafe_penalty(place)
            - pet_care_penalty(place, tags)
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

    return enforce_preferred_tag(selected, ranked_places, tags)


def candidate_lookup_tool(
    state: AgentState,
    tags: list[str],
    max_candidates: int = 48,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slots = itinerary_slots(state["duration"])
    if state["transport"] == "도보 중심" and state["duration"] == "당일치기":
        slots = slots[:4]

    places: list[dict[str, Any]] = []
    place_db = load_place_db()
    target_count = len(slots)
    per_place_budget = max(1, state["budget"] / max(1, target_count))

    for place in place_db:
        quality = float(place.get("quality_score") or place_quality_score(place))
        if quality < MIN_RECOMMENDATION_QUALITY:
            continue
        if "동물먹이" in tags and ((place.get("role") == "activity" or place["category"] == "동물체험") and is_pet_care_place(place)):
            continue

        matched_tags = sorted(set(tags).intersection(place["tags"]))
        start_distance = haversine_km(state["start_point"], place)
        budget_penalty = 1.0 if place["cost"] > per_place_budget and place["cost"] > 0 else 0
        score = (
            place["score"]
            + min(quality, 6.0) * 0.35
            + len(matched_tags) * 2
            + weather_filter_score(place, state["weather"])
            + indoor_preference_score(place, tags)
            + category_preference_score(place, tags)
            + start_proximity_score(start_distance, state["transport"])
            - budget_penalty
            - repeated_cafe_penalty(place)
            - pet_care_penalty(place, tags)
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


def place_to_document(place: dict[str, Any]) -> Any:
    page_content = (
        f"{place.get('name', '')}\n"
        f"category={place.get('category', '')}; role={place.get('role', '')}; "
        f"tags={', '.join(place.get('tags', []))}; address={place.get('address', '')}; "
        f"source={place.get('source', '')}; score={place.get('score', '')}"
    )
    metadata = {
        "name": place.get("name"),
        "category": place.get("category"),
        "role": place.get("role"),
        "tags": place.get("tags", []),
        "source": place.get("source"),
        "address": place.get("address"),
    }
    if Document:
        return Document(page_content=page_content, metadata=metadata)
    return {"page_content": page_content, "metadata": metadata}


def document_to_context(document: Any) -> dict[str, Any]:
    if Document and isinstance(document, Document):
        return {"page_content": document.page_content, "metadata": document.metadata}
    return {
        "page_content": document.get("page_content", ""),
        "metadata": document.get("metadata", {}),
    }


def retrieve_place_documents(candidates: list[dict[str, Any]], max_documents: int = 12) -> list[dict[str, Any]]:
    documents = [place_to_document(place) for place in candidates[:max_documents]]
    return [document_to_context(document) for document in documents]


def compact_candidate_for_llm(index: int, place: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": index,
        "name": place["name"],
        "category": place["category"],
        "role": place.get("role"),
        "cost": place["cost"],
        "indoor": place["indoor"],
        "tags": place["tags"],
        "matched_tags": place.get("matched_tags", []),
        "quality_score": place.get("quality_score"),
        "agent_score": place.get("agent_score"),
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

    compact_candidates = [
        compact_candidate_for_llm(index, place)
        for index, place in enumerate(candidates, start=1)
    ]
    prompt = (
        "너는 청주 여행 동선을 직접 판단하는 AI Agent다. 아래 후보 목록만 사용해서 여행 일정을 선택해라. "
        "규칙 점수는 참고자료일 뿐이고, 최종 선택은 네가 사용자 의도, 품질점수, 실제 방문 가능성, "
        "거리, 예산, 날씨, 카테고리 균형을 비교해서 결정한다. "
        "품질점수가 낮거나 지도/전화/주소 근거가 약한 곳, 사용 의도와 카테고리가 맞지 않는 곳은 고르지 마라. "
        "반드시 JSON 객체만 출력해라.\n\n"
        "JSON 스키마: {\"selected\":[{\"candidate_id\":1,\"day\":1,\"slot_role\":\"meal\","
        "\"slot_label\":\"점심\",\"reason\":\"선택 이유\"}],\"decision_summary\":\"판단 요약\","
        "\"rejected_notes\":[\"제외 판단\"]}.\n"
        "selected 개수는 slots 개수를 넘기지 말고, 예산 안에 최대한 맞춰라.\n\n"
        f"사용자 요청: {state['style_text']}\n"
        f"대화 메모리: {json.dumps(state.get('memory_context', []), ensure_ascii=False)}\n"
        f"LLM 의도 해석: {json.dumps(intent, ensure_ascii=False)}\n"
        f"태그: {tags}\n"
        f"출발지: {state['start_name']}, 기간: {state['duration']}, 이동수단: {state['transport']}, "
        f"날씨: {state['weather']}, 예산: {state['budget']}\n"
        f"필요 슬롯: {json.dumps(slots, ensure_ascii=False)}\n"
        f"RAG 검색 문서 Context: {json.dumps(retrieved_documents or [], ensure_ascii=False)}\n"
        f"후보 목록: {json.dumps(compact_candidates, ensure_ascii=False)}"
    )
    parsed, mode = gemini_json_tool(prompt, temperature=0.25, timeout=25)
    if not parsed:
        return [], f"LLM 동선 선택 Fallback: {mode}", {}

    selected_items = parsed.get("selected", [])
    if not isinstance(selected_items, list):
        return [], "LLM 동선 선택 JSON 형식 오류 Fallback", parsed
    route = hydrate_llm_route(selected_items, candidates, slots, tags)
    return route, "Gemini LLM 동선 선택", parsed


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
    if transport == "자동차":
        return distance - (0.25 if route and place["category"] not in {item["category"] for item in route} else 0)
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
    accommodation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    speed = TRANSPORT_SPEED_KMH[transport]
    legs = []
    current_name = "출발지"
    current = start_point
    current_day = 1
    transit_leg_count = 0
    max_transit_legs = DAY_TRIP_MAX_TRANSIT_LEGS if duration == "당일치기" else len(route)

    for place in route:
        place_day = int(place.get("day", 1))
        if accommodation and place_day != current_day:
            current_name = accommodation["name"]
            current = accommodation
            current_day = place_day

        distance = haversine_km(current, place)
        walk_minutes = max(5, round(distance / TRANSPORT_SPEED_KMH["도보 중심"] * 60))
        mode = "walk"
        transit_options: list[dict[str, Any]] = []
        transit_error = None

        if transport == "대중교통" and walk_minutes > WALK_ONLY_MAX_MINUTES and transit_leg_count < max_transit_legs:
            transit_options, transit_error = odsay_transit_options(current, place)
            if transit_options:
                mode = "transit"
                transit_leg_count += 1
                move_minutes = max(5, int(transit_options[0]["total_time"]))
            else:
                move_minutes = walk_minutes
        elif transport == "대중교통":
            move_minutes = walk_minutes
            if walk_minutes > WALK_ONLY_MAX_MINUTES and transit_leg_count >= max_transit_legs:
                transit_error = "당일치기 버스 이용 최대 2회 조건으로 도보 이동 처리"
        else:
            move_minutes = max(5, round(distance / speed * 60))
            mode = "car" if transport == "자동차" else "walk"
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


def accommodation_tool(
    state: AgentState,
    route: list[dict[str, Any]],
    place_cost: int,
) -> dict[str, Any] | None:
    if state["duration"] != "1박 2일":
        return None

    day1_places = [place for place in route if place.get("day", 1) == 1]
    day1_last_place = day1_places[-1] if day1_places else (route[0] if route else state["start_point"])
    remaining_budget = max(0, state["budget"] - place_cost)

    candidates = []
    for accommodation in ACCOMMODATION_DB:
        distance = haversine_km(day1_last_place, accommodation)
        budget_penalty = 3 if accommodation["cost"] > remaining_budget else 0
        transport_bonus = 1.0 if state["transport"] in accommodation["tags"] or "교통" in accommodation["tags"] else 0
        score = accommodation["score"] + transport_bonus - distance * 0.25 - budget_penalty
        candidates.append(
            {
                **accommodation,
                "distance_from_day1_km": round(distance, 2),
                "agent_score": round(score, 2),
                "matched_tags": sorted(set(state["tags"]).intersection(accommodation["tags"])),
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
    previous_day = 1

    for index, place in enumerate(route):
        day = int(place.get("day", 1))
        if duration == "1박 2일" and day != previous_day:
            current_minutes = 10 * 60
            previous_day = day
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

        next_day = int(route[index + 1].get("day", day)) if index + 1 < len(route) else day
        if duration == "1박 2일" and accommodation and day == 1 and next_day == 2:
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
    if place.get("llm_reason"):
        return str(place["llm_reason"])
    matched = ", ".join(place["matched_tags"]) if place["matched_tags"] else "균형 일정"
    slot = place.get("slot_label")
    weather_note = "실내" if place["indoor"] else "야외"
    if slot:
        return f"{slot} 슬롯에 맞춰 선택했고, {matched} 선호와 맞는 {weather_note} 장소입니다."
    return f"{matched} 선호와 맞고, {weather_note} 일정으로 활용하기 좋습니다."


def output_parser(
    state: AgentState,
    route: list[dict[str, Any]],
    legs: list[dict[str, Any]],
    accommodation: dict[str, Any] | None = None,
    planner_mode: str = "규칙 기반 추천",
    intent: dict[str, Any] | None = None,
    planner_decision: dict[str, Any] | None = None,
    rag_document_count: int = 0,
) -> dict[str, Any]:
    place_cost = sum(place["cost"] for place in route)
    accommodation = accommodation if accommodation is not None else accommodation_tool(state, route, place_cost)
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
    route_names = [state["start_name"]] + [place["name"] for place in route]
    if accommodation:
        split_index = next((index for index, place in enumerate(route) if place.get("day", 1) == 2), len(route))
        route_names = (
            [state["start_name"]]
            + [place["name"] for place in route[:split_index]]
            + [f"{accommodation['name']}(숙소)"]
            + [place["name"] for place in route[split_index:]]
        )

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
            "lodging_cost": lodging_cost,
            "session_id": state["session_id"],
            "memory_turns": len(state.get("memory_context", [])),
            "rag_document_count": rag_document_count,
        },
        "schedule": build_schedule(route, legs, state["duration"], accommodation),
        "accommodation": accommodation,
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
                "llm_reason": place.get("llm_reason"),
            }
            for place in route
        ],
        "legs": legs,
        "warnings": state["warnings"],
        "agent_flow": [
            "LangGraph StateGraph: validate_input",
            "입력 검증 Middleware",
            "개인정보 제거 Middleware",
            "Session Memory 병합",
            "Fallback Middleware",
            "LangGraph StateGraph: analyze_intent",
            "LangChain PydanticOutputParser: TravelIntent",
            "LLM 여행 의도 해석 Tool",
            "LangGraph StateGraph: retrieve_places",
            "로컬 JSON 장소 DB → LangChain Document Retriever",
            "검색된 후보를 LLM Context로 제공",
            "품질/실재성 필터 Tool",
            "LangGraph StateGraph: plan_route",
            "LLM 후보 비교/동선 선택 Tool",
            "add_conditional_edges: LLM 실패 시 fallback_route",
            "LangGraph StateGraph: fallback_route",
            "규칙 기반 Fallback Tool",
            "로컬 JSON 장소 DB",
            "날씨 대응 Tool",
            "LangGraph StateGraph: final_response",
            "거리 계산 Tool",
            "동선 최적화 Tool",
            "숙소 추천 Tool",
            "Context 생성",
            "LLM 응답 생성 Tool",
            "LangChain PydanticOutputParser: TripPlan",
        ],
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
            "session_id": state["session_id"],
            "start": state["start_name"],
            "duration": state["duration"],
            "style_text": state["style_text"],
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
        "accommodation": accommodation,
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def fast_final_comment(
    state: AgentState,
    route: list[dict[str, Any]],
    total_cost: int,
    total_distance: float,
    total_move_minutes: int,
    accommodation: dict[str, Any] | None = None,
) -> str:
    route_preview = " → ".join(place["name"] for place in route[:3])
    if len(route) > 3:
        route_preview += " → ..."
    lodging_note = f" 마지막은 {accommodation['name']} 체크인까지 이어집니다." if accommodation else ""
    return (
        f"{state['weather']} 날씨와 {state['transport']} 이동을 기준으로 {route_preview} 순서가 가장 무난합니다. "
        f"예상 비용은 {total_cost:,}원, 이동은 약 {total_move_minutes}분/{total_distance}km입니다."
        f"{lodging_note}"
    )


def llm_response_tool(
    state: AgentState,
    route: list[dict[str, Any]],
    legs: list[dict[str, Any]],
    total_cost: int,
    total_distance: float,
    total_move_minutes: int,
    accommodation: dict[str, Any] | None = None,
) -> tuple[str, str]:
    if not env_flag("ENABLE_FINAL_LLM", default=False):
        return (
            fast_final_comment(state, route, total_cost, total_distance, total_move_minutes, accommodation),
            "빠른 로컬 응답",
        )

    context = build_llm_context(
        state,
        route,
        legs,
        total_cost,
        total_distance,
        total_move_minutes,
        accommodation,
    )
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return (
            "현재 실행 환경에는 GEMINI_API_KEY가 없어 규칙 기반 Fallback으로 최종 문장을 생성했습니다. "
            "추천 결과는 Tool들이 만든 Context를 기반으로 구성되었습니다.",
            "규칙 기반 Fallback",
        )

    prompt = (
        "너는 청주 여행 추천 Agent의 최종 응답 생성기다. "
        "아래 JSON Context만 근거로 사용해서 한국어로 3문장 이내의 짧은 추천 코멘트를 작성해라. "
        "예산, 날씨, 동선 최적화 이유를 자연스럽게 포함해라.\n\n"
        f"{context}"
    )
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
        },
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model, safe='-_.')}:generateContent"
        f"?key={urllib.parse.quote(api_key)}"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            data = json.loads(response.read().decode("utf-8"))
            llm_text = extract_gemini_text(data)
            if llm_text:
                return llm_text, "Gemini LLM"
            return (
                "Gemini 호출은 성공했지만 응답 텍스트가 비어 있어 규칙 기반 Fallback 문장을 사용했습니다.",
                "LLM 응답 없음 Fallback",
            )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
        return (
            f"LLM 호출이 실패해 규칙 기반 Fallback으로 응답했습니다. 실패 사유: {error}",
            "LLM 실패 Fallback",
        )


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
    try:
        slots, candidates = candidate_lookup_tool(state, tags)
    except RuntimeError as error:
        errors = [
            str(error),
            (
                "TourAPI를 사용하려면 TOUR_API_KEY 또는 TOURAPI_SERVICE_KEY 환경변수를 설정한 뒤 "
                "/api/places/sync를 호출하세요."
            ),
        ]
        return {"errors": errors, "status": 503, "result": {"errors": errors}}
    return {
        "slots": slots,
        "candidates": candidates,
        "retrieved_documents": retrieve_place_documents(candidates),
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
    recommended = recommendation_tool(
        graph_state.get("tags", state["tags"]),
        state["budget"],
        state["weather"],
        state["duration"],
        state["start_point"],
        state["transport"],
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

    accommodation = accommodation_tool(state, recommended, sum(place["cost"] for place in recommended))
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


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/recommend")
def recommend():
    result, status = run_agent(request.get_json(force=True))
    return jsonify(result), status


@app.post("/api/places/sync")
def sync_places():
    try:
        places = sync_place_db()
    except RuntimeError as error:
        return jsonify({"errors": [str(error)]}), 503
    payload = json.loads(PLACE_DB_PATH.read_text(encoding="utf-8"))
    return jsonify(
        {
            "count": len(places),
            "db_path": str(PLACE_DB_PATH),
            "source": payload.get("source"),
            "source_counts": payload.get("source_counts", {}),
            "sync_errors": payload.get("sync_errors", []),
            "places": places,
        }
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5001)
