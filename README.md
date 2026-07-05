# Cheongju Trip Agent

청주 여행 스타일을 입력하면 관광 API에서 수집한 장소 데이터를 로컬 JSON DB로 저장하고, 그 DB를 기반으로 추천 동선을 생성하는 Flask 앱입니다.

## 장소 데이터 수집

기본 수집 경로는 충청북도 관광명소정보 API입니다.

```powershell
python app.py
```

앱 실행 후 장소 DB를 갱신하려면 다음 엔드포인트를 호출합니다.

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:5000/api/places/sync
```

수집된 데이터는 `data/cheongju_places.json`에 저장됩니다. 추천 요청 시 이 파일이 없으면 자동으로 수집을 시도합니다.

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
