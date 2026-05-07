# Air Freight Index 예측 웹사이트

`xgboost_2026_jan_feb.py`를 사용하는 Streamlit 웹앱입니다.
사용자가 WTI, Brent, DXY, VIX, GSCPI, BDI, GPR, GPRH, GDP 값을 입력하면 XGBoost 모델이 Air Freight Index를 예측하고 그래프를 출력합니다.

## 실행 방법

```bash
cd air_freight_webapp_v2
pip install -r requirements.txt
streamlit run app.py
```

## 구성

- `app.py`: 웹사이트 실행 파일
- `xgboost_2026_jan_feb.py`: 기존 XGBoost 학습/예측 코드
- `data/raw/`: 모델 학습에 필요한 CSV 데이터
- `result/`: 기존 결과 이미지

## 사용 방법

1. 사이드바에서 GDP 미포함/GDP 포함 모델 선택
2. 2026년 1월 또는 2월 기본값 선택
3. 입력 박스의 피처값 수정
4. 예측 대상 월 입력
5. `예측하기` 클릭
6. 예측값, 그래프, 입력값 테이블 확인

