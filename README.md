# Cheongju Trip Agent

청주 여행 스타일을 입력하면 관광 API와 카카오 Local API에서 수집한 장소 데이터를 로컬 JSON DB로 저장하고, 그 DB를 기반으로 밥집/카페/놀거리 슬롯형 추천 동선을 생성하는 Flask 앱입니다.

## 장소 데이터 수집

기본 수집 경로는 충청북도 관광명소정보 API입니다. 밥집/카페/숙소 같은 세부 장소는 카카오 Local API 키가 있을 때 함께 수집됩니다.

```powershell
python app.py
```

앱 실행 후 장소 DB를 갱신하려면 다음 엔드포인트를 호출합니다.

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:5000/api/places/sync
```

수집된 데이터는 `data/cheongju_places.json`에 저장됩니다. 추천 요청 시 이 파일이 없으면 자동으로 수집을 시도합니다.

`data/cheongju_places.json`은 API 응답으로 생성되는 로컬 캐시라 Git에는 올리지 않습니다. GitHub에는 `.env.example`만 올리고, 실제 `.env`와 발급받은 API 키는 커밋하지 마세요.

## 카카오 Local API 사용

카카오 Developers에서 REST API 키를 발급받아 `.env`에 넣습니다.

```powershell
KAKAO_REST_API_KEY=발급받은_REST_API_KEY
```

수집 키워드는 앱 내부에서 다음 유형으로 자동 검색합니다.

- `청주 성안길 맛집`
- `청주 성안길 카페`
- `청주 운리단길 카페`
- `청주 수암골 카페`
- `청주 육거리시장 맛집`
- `청주 청주대 맛집`
- `청주 충북대 맛집`

키 설정 후 서버를 다시 실행하고 `/api/places/sync`를 호출하면 카카오 장소가 로컬 JSON DB에 합쳐집니다.

## TourAPI 사용

충청북도 API 수집이 실패하거나 TourAPI를 우선 활용하려면 서비스키를 환경변수로 설정합니다.

```powershell
$env:TOUR_API_KEY="발급받은_서비스키"
python app.py
```

청주 시군구 코드가 다르면 아래 값으로 조정할 수 있습니다.

```powershell
$env:TOUR_API_AREA_CODE="33"
$env:TOUR_API_SIGUNGU_CODE="10"
```
