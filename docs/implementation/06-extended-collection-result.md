# Extended 피처 수집 결과

## 결론

뉴스 감성을 제외한 extended 후보는 현재 수집 가능한 범위에서 모두 수집했다.
공식 수집 run은 다음과 같다.

| 항목 | 값 |
|---|---|
| 공식 run id | `20260520T155315691434Z-82034936` |
| 수집 범위 | `2011-01-01` ~ `2026-05-21` |
| 공식 raw 경로 | `data/raw/runs/20260520T155315691434Z-82034936/` |
| run metadata | `data/metadata/runs/20260520T155315691434Z-82034936/collection_summary.csv` |
| 현재 manifest | `data/metadata/current_raw_manifest.csv` |

`data/metadata/current_raw_manifest.csv`도 위 run을 가리키도록 갱신했다.

---

## 수집 완료 요약

최종 metadata 기준 상태는 다음과 같다.

| category | 상태 |
|---|---|
| `stock_price` | 3개 성공 |
| `macro_series` | 39개 성공 |
| `macro` | `core_macro_features` 성공 |
| `alpha` | 국내 외국인/기관 순매수 2개 성공 |
| `financials` | 5개 성공 |
| `external_archive` | 6개 성공 |
| `pipeline` | coverage warning 1개 |

실패 상태로 남은 피처는 없다.

---

## 수집된 주요 피처군

### 1. 주가

- 삼성전자
- SK하이닉스
- NVIDIA

### 2. Core macro

- NASDAQ
- SOX
- VIX
- FEDFUNDS
- USD/KRW
- `core_macro_features.csv`

### 3. 금리/달러 계열

- `DGS10`
- `DGS2`
- `T10Y2Y`
- `DTWEXBGS`
- FDR 기반 미국 5년/10년/30년 금리
- FDR 기반 dollar index

### 4. 시장지수/반도체 ETF

- KOSPI
- KOSPI200
- KOSDAQ
- S&P500
- Dow Jones
- Russell 2000
- Shanghai Composite
- Hang Seng
- Nikkei225
- SMH
- SOXX

KOSPI, KOSPI200, KOSDAQ은 FDR에서 빈 데이터가 반환되어
yfinance 대체 심볼로 수집하도록 보완했다.

### 5. FRED 업황 지표

- `IPG3344S`: 반도체·전자부품 산업생산
- `PCU334413334413P`: 반도체 관련 PPI
- `A34SNO`: 컴퓨터·전자제품 신규주문
- `INDPRO`
- `VIXCLS`
- `NASDAQCOM`

월간 지표는 모델 투입 전 발표일/as-of 또는 보수적 lag 처리가 필요하다.

### 6. 환율/원자재

- USD/KRW
- USD/CNY
- USD/JPY
- CNY/KRW
- WTI
- Brent
- 천연가스
- 금
- 은
- 구리

단, CNY/KRW는 수집 row가 1개라 모델 피처로 쓰기에는 부적합하다.
파일은 남기되 모델 투입에서는 제외하거나 별도 대체 소스를 확인해야 한다.

### 7. 국내 수급

- 삼성전자 외국인/기관 순매수
- SK하이닉스 외국인/기관 순매수

초기 실행에서는 `KRX_ID`, `KRX_PW` 환경변수가 없어 실패했으나,
KRX 일반 로그인 계정 정보를 환경변수로 설정한 뒤 재실행하여 성공했다.

수집 결과:

| 피처 | rows | 기간 |
|---|---:|---|
| `samsung_electronics_alpha` | 3,778 | `2011-01-03` ~ `2026-05-20` |
| `sk_hynix_alpha` | 3,781 | `2011-01-03` ~ `2026-05-20` |

이 값은 당일 장중/종가 이후 확정되는 성격이 있으므로,
모델에서는 다음 거래일 예측 등에 lag를 둬서 사용해야 한다.

### 8. 재무/공시

- 삼성전자 연간 재무제표 스냅샷
- 삼성전자 분기 재무제표 스냅샷
- SK하이닉스 연간 재무제표 스냅샷
- SK하이닉스 분기 재무제표 스냅샷
- NVIDIA SEC Company Facts 주요 재무 항목 정제 CSV

SEC 원본 JSON은 재현 가능하지만 크기가 커서 저장하지 않고,
정제 CSV만 저장한다.

재무제표 및 SEC 데이터는 period end 기준으로 바로 붙이면 look-ahead가 생길 수 있다.
모델 투입 전 filing date 또는 공시일 기준의 `effective_from` 처리가 필요하다.

### 9. Archive/reference

아래 자료는 모델 입력용 구조화 피처가 아니라 archive/reference로 보관한다.

- SIA market data page
- SEMI billings page
- TrendForce DRAM prices page
- TSMC monthly revenue page
- NVIDIA investor events page
- CHIPS Act NIST page

최신 run에서 일부 HTML 페이지는 403이 발생했으나,
직전 성공 run의 archive 파일을 최신 official run에 보강했다.
metadata에는 `reused-from-run`으로 남겨두었다.

---

## Coverage warning

최종 run에는 실패가 아니라 coverage warning이 1개 남아 있다.

| 항목 | 내용 |
|---|---|
| 삼성전자 주가 | 소스에서 `2014-02-27`부터 제공 |
| SK하이닉스 주가 | 소스에서 `2014-02-27`부터 제공 |
| FEDFUNDS | 최신 관측치가 `2026-04-01`까지 제공 |

이는 수집 실패가 아니라 원천 데이터의 제공 범위 차이다.
전처리/모델링에서는 실제 available period를 기준으로 windowing하면 된다.