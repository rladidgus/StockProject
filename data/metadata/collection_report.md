# 데이터 수집 결과 리포트

수집 실행 시각: 2026-05-09 01:02 KST

## 요약

| 구분 | 성공 | 실패 | 비고 |
|---|---:|---:|---|
| 1차 수집 | 36 | 1 | FDR/FRED/NAVER 중심 |
| 2차 수집 | 4 | 1 | SEC, SIA, SEMI, TrendForce, TSMC 시도 |
| 3차 수집 | 2 | 0 | NVIDIA 이벤트, CHIPS Act 페이지 |
| 전체 | 42 | 2 | 실패 내역은 `failed_collections.csv` 참고 |

## 수집 완료 산출물

- 주가: 삼성전자, SK하이닉스, NVIDIA
- 재무: 삼성전자 FDR/NAVER 연간·분기, SK하이닉스 FDR/NAVER 연간·분기, NVIDIA SEC company facts
- 반도체 종합 지수: SMH, SOXX, `^SOX`
- 글로벌 지수: KOSPI, KOSPI200, KOSDAQ, NASDAQ, S&P500, Dow Jones, Russell 2000, VIX, Shanghai Composite, Hang Seng, Nikkei 225
- 환율/금리/달러: USD/KRW, USD/CNY, USD/JPY, CNY/KRW, 미국 5년·10년·30년 금리, 달러인덱스
- 원자재 선물: WTI, Brent, 천연가스, 금, 은, 구리
- FRED: FEDFUNDS, DGS10, DGS2, T10Y2Y, VIXCLS, NASDAQCOM, DTWEXBGS, INDPRO, IPG3344S, PCU334413334413P, A34SNO
- 외부 반도체 업황 후보: SIA/WSTS 페이지, SEMI billings 페이지, TrendForce 가격 페이지
- 이벤트 후보: NVIDIA investor events 페이지, NIST CHIPS Act 페이지

## 처리 패널

| 파일 | 설명 |
|---|---|
| `data/processed/daily_panel.csv` | 일별 종가 패널, 31개 시계열 |
| `data/processed/daily_panel.parquet` | 일별 종가 패널 Parquet |
| `data/processed/daily_returns.csv` | 일별 수익률 패널 |
| `data/processed/daily_returns.parquet` | 일별 수익률 패널 Parquet |
| `data/processed/monthly_panel.csv` | 월말 기준 시장 데이터 + FRED 패널, 42개 시계열 |
| `data/processed/monthly_panel.parquet` | 월별 패널 Parquet |

## 실패 항목

| 우선순위 | 항목 | 원인 | 대응 |
|---:|---|---|---|
| 1 | `SOX` 직접 심볼 | FDR/Yahoo 응답 파싱 중 `timestamp` 키 없음 | `^SOX` 수집 성공. 분석에는 `phlx_semiconductor_index_yahoo`, `SMH`, `SOXX` 사용 |
| 2 | TSMC monthly revenue page | HTTP 403 Forbidden | TSMC IR 페이지는 차단. 수동 다운로드, TSMC API 대체, 또는 월별 매출 CSV 별도 적재 필요 |

## 주의점

- 삼성전자와 SK하이닉스 주가는 FDR/NAVER 응답 기준 약 3000거래일로 수집되어 2014-02-17부터 시작한다. 최소 10년 조건은 만족하지만, 2011년부터의 국내 주가가 반드시 필요하면 KRX/네이버 보조 수집기를 추가해야 한다.
- FDR/NAVER 재무제표는 장기 전체 재무제표가 아니라 최근 실적과 추정치가 섞인 스냅샷이다. 10년 이상 재무제표가 필요하면 DART API 키 또는 별도 재무 API가 필요하다.
- SIA/WSTS, SEMI, TrendForce는 공개 페이지 접근까지만 성공했다. 구조화된 월별 데이터는 유료/로그인/수동 다운로드 성격이 강하므로, 라이선스 확인 후 CSV 적재 방식으로 보강하는 것이 현실적이다.
