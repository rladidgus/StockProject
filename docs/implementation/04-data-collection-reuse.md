# data_collection 산출물 재사용 기준

## 목적

`data_collection` 브랜치의 수집 산출물 중 현재 브랜치에서 바로 활용 가능한 데이터를 선별한다.
구조화된 CSV는 가져오고 재현 가능한 수집 로직은 `src/collect.py`에 반영한다.

---

## 재사용한 데이터

아래 데이터는 구조화된 시계열 또는 정제 CSV라서 현재 브랜치에 가져왔다.

- `data/raw/fdr/fred/`
  - `DGS10`, `DGS2`, `T10Y2Y`, `DTWEXBGS`
  - `IPG3344S`, `PCU334413334413P`, `A34SNO`
  - 보조 FRED 지표
- `data/raw/fdr/global_indices/`
  - KOSPI, KOSPI200, KOSDAQ
  - NASDAQ, S&P500, Dow Jones, Russell 2000
  - Shanghai, Hang Seng, Nikkei225, VIX
- `data/raw/fdr/semi_indices/`
  - `^SOX`, `SMH`, `SOXX`
- `data/raw/fdr/rates/`
  - 미국 5년/10년/30년 금리
  - 달러 인덱스
- `data/raw/fdr/fx_rates/`
  - USD/KRW, USD/CNY, USD/JPY, CNY/KRW
- `data/raw/fdr/futures/`
  - WTI, Brent, 천연가스, 금, 은, 구리
- `data/raw/fdr/financials/`
  - 삼성전자, SK하이닉스 NAVER/FDR 재무제표 스냅샷
- `data/raw/external/sec/nvidia_sec_key_financials.csv`
  - NVIDIA SEC Company Facts에서 주요 재무 항목만 추출한 정제 CSV

---

## Archive로만 볼 데이터

아래 데이터는 원문/HTML archive다.
그 자체를 모델 피처로 쓰지 말고, 구조화 수치 시계열이 확보될 때만 모델 후보로 승격한다.

- `data/raw/external/public_pages/sia_market_data_page.html`
- `data/raw/external/public_pages/semi_billings_page.html`
- `data/raw/external/public_pages/trendforce_dram_prices_page.html`
- `data/raw/external/public_pages/nvidia_investor_events_page.html`
- `data/raw/external/public_pages/chips_act_nist_page.html`

`TSMC monthly revenue page`는 `data_collection`에서 403으로 실패했으므로 가져오지 않았다.

---

## 주의할 데이터

- `data/raw/fdr/fx_rates/cny_krw.csv`
  - `data_collection` 기준 row가 1개뿐이다.
  - 수집 파일은 남기지만 모델 피처로 쓰기에는 부적합하다.
- NAVER/FDR 재무제표 스냅샷
  - 과거 시점 재현성이 약할 수 있다.
  - 모델 투입 전 DART 공시일 기준 as-of 처리 여부를 별도로 검토한다.
- 월간·분기·공시 기반 지표
  - `period_end` 기준으로 바로 붙이면 look-ahead가 생길 수 있다.
  - 모델 투입 시 `release_date` 또는 `filing_date` 기반 `effective_from` 정렬이 필요하다.

---

## 현재 브랜치 반영 방식

- `data_collection`의 구조화 raw 산출물을 현재 브랜치의 `data/raw/fdr/`, `data/raw/external/`에 병합했다.
- 대용량 SEC 원본 JSON은 커밋하지 않고, 정제 CSV만 병합했다.
- SEC 원본 JSON은 `src/collect.py`의 재수집 로직으로 재현 가능하게 한다.
- `src/collect.py`는 core 수집 성공 여부를 기존처럼 core 기준으로만 판단한다.
- extended/archive 수집 실패는 baseline 실패로 보지 않는다.

---

## 기간 차이에 따른 처리 원칙

`data_collection` 산출물은 대체로 2011년부터 2026년까지의 넓은 기간을 가진다.
기존 현재 브랜치의 공식 수집 범위였던 2018년부터 2024년까지와 그대로 섞으면,
전처리에서 날짜 범위와 시장 regime이 달라지는 문제가 생긴다.

따라서 현재 브랜치의 수집 기본값도 `2011-01-01`부터 수집 실행일 현재까지로 넓힌다.
기존 core 피처도 같은 기준으로 재수집한다.

다만 모델 학습에 전 기간을 그대로 넣지는 않는다.
권장 방식은 다음과 같다.

- raw 수집과 EDA: 2011년부터 현재까지 사용
- A안 core-only baseline: 2018년부터 2024년까지를 기본 학습/검증 구간으로 사용
- 최근 regime 검증: 2025년부터 2026년 현재까지를 별도 테스트 구간으로 사용
- B안 extended 탐색: 전체 기간 또는 기간별 구간을 나누어 비교하되, look-ahead와 중복 피처를 별도 검증

즉 파일 경로 충돌은 없지만, 전처리에서는 반드시 기준 기간을 명시하고 windowing한 뒤 join한다.
