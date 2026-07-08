# Cheongju Trip Agent

청주 여행 요청을 자연어로 입력하면 OpenAI 기반 Agent, 로컬 RAG 데이터, Kakao Local API, TourAPI, ODsay, Tavily를 조합해 여행 코스를 추천하는 FastAPI 앱입니다.

이 프로젝트는 OpenAI 전용입니다. 다른 LLM provider용 실행 파일, 환경변수, provider 분기는 사용하지 않습니다.

## 구조

```text
final-agent/
├─ server.py
├─ prompts.py
├─ public/
│  ├─ index.html
│  └─ style.css
├─ data/
│  ├─ places_cache.json
│  └─ kakao_keywords.json
├─ .env.example
├─ requirements.txt
└─ README.md
```

핵심 코드 파일은 `server.py`, `prompts.py`, `public/index.html`, `public/style.css`입니다. `data/places_cache.json`은 RAG 검색 대상이고, `data/kakao_keywords.json`은 카카오 검색어 데이터입니다.

## 실행

```powershell
pip install -r requirements.txt
python server.py
```

브라우저에서 `http://127.0.0.1:5000`을 열면 됩니다.

## 환경변수

`.env.example`을 참고해 필요한 키만 `.env`에 설정하세요. OpenAI LLM 의도 해석/동선 선택을 사용하려면 `OPENAI_API_KEY`가 필요합니다. 키가 없으면 로컬 `places_cache.json` 기반 fallback 추천은 동작합니다.

```powershell
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=openai:gpt-4o-mini
KAKAO_REST_API_KEY=your_kakao_rest_api_key
TOUR_API_KEY=your_tour_api_service_key
ODSAY_API_KEY=your_odsay_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 구현 방식

- FastAPI 서버, middleware, API route, 정적 파일 서빙은 `server.py`에 있습니다.
- Kakao, TourAPI, ODsay, Tavily 호출은 `server.py`의 adapter 함수로 압축했습니다.
- Agent tool은 `search_places`, `check_route`, `search_reviews` 3개만 노출합니다.
- RAG는 `data/places_cache.json`을 읽고 사용자 질문의 키워드/태그와 매칭합니다. 선택 의존성이 준비된 환경에서는 기존처럼 LangChain Document 계층을 통해 확장할 수 있습니다.
- 세션별 메모리는 `SESSION_STORE`에 최근 6턴을 저장합니다.
- 최종 응답은 `TripPlan` Pydantic 모델과 `PydanticOutputParser`로 검증합니다.
- LangGraph `StateGraph`는 `validate_input → analyze_intent → retrieve_places → plan_route → check_quality → final_response` 흐름이며, 품질이 낮으면 재검색 또는 `fallback_route`로 조건 분기합니다.

OpenAI 호출은 `server.py`의 OpenAI provider로 통합되어 있습니다. 다른 LLM provider 분기는 제거했습니다.
