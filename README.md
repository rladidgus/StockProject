# 반도체 주가 영향 요인 분석 프로젝트

## 프로젝트 목표 및 가설

- __목표__<br>
임의의 기간 동안 삼성전자, SK하이닉스, NVIDIA 주가의 상승/하락에 영향을 주는 요인을 통계 분석과 머신러닝 기법으로 파악하여 사용자에게 제공한다. 

---

- __가설__<br>
A지표가 몇일 뒤 B주식의 가격에 영향을 줄 것이다.

---
- __반도체 주식 흐름__<br>


1. 거시 유동성/금리/달러 환경
2. IT·AI·서버·모바일 최종 수요
3. DRAM/NAND/HBM 등 제품 가격과 재고 사이클
4. 설비투자와 장비 발주
5. 기업별 실적, 마진, 밸류에이션
6. 시장 심리와 이벤트

따라서 데이터는 `주가`, `재무`, `반도체 업황`, `거시경제`, `수급/가격`, `이벤트`로 나누어 수집한다.

## 수집 기간 후보

- 10년: 2016년 이후 메모리 슈퍼사이클, 2018년 다운사이클, 코로나 공급망 충격, 2022년 금리 상승, 2023년 이후 AI/HBM 사이클을 포함.
- 15년 내외: 2011년 이후 스마트폰 성장, PC 둔화, 데이터센터 성장, 메모리 가격 사이클 안정적 학습 가능

(코덱스 추천) 실무적으로는 `2011-01-01 ~ 현재`를 원천 저장 구간으로 두고, 모델 학습은 `rolling window` 방식으로 5년/10년/전체 기간을 비교하는 것이 좋다.

## FinanceDataReader 중심 수집 항목

FinanceDataReader<br>
주가, 주요 지수, 환율, 일부 원자재 선물, 미국 국채 수익률, FRED 데이터, 국내 기업 네이버 재무제표 수집

```python
import FinanceDataReader as fdr

START = "2011-01-01"
END = None

prices = {
    "samsung": fdr.DataReader("005930", START, END),
    "sk_hynix": fdr.DataReader("000660", START, END),
    "nvidia": fdr.DataReader("NVDA", START, END),
}
```

### 1. 대상 기업 주가

| 구분 | 심볼 | 수집 방법 | 비고 |
|---|---:|---|---|
| 삼성전자 | `005930` | `fdr.DataReader("005930")` | 원화 기준, 배당/액면분할 조정 여부 확인 필요 |
| SK하이닉스 | `000660` | `fdr.DataReader("000660")` | 메모리 업황 민감도 높음 |
| NVIDIA | `NVDA` | `fdr.DataReader("NVDA")` | AI, GPU, 데이터센터 수요 대리 변수|

파생 변수:

- 일간/주간/월간 수익률
- 5/20/60/120일 모멘텀
- 20/60일 변동성
- 거래량 변화율
- 시장 대비 초과수익률
- 원화 환산 NVIDIA 수익률: `NVDA_USD_return + USD/KRW_return`

### 2. 재무제표 데이터

NAVER 재무제표 스냅샷 활용

```python
samsung_fs_y = fdr.SnapDataReader("NAVER/FINSTATE-Y/005930")
samsung_fs_q = fdr.SnapDataReader("NAVER/FINSTATE-Q/005930")
skhynix_fs_y = fdr.SnapDataReader("NAVER/FINSTATE-Y/000660")
skhynix_fs_q = fdr.SnapDataReader("NAVER/FINSTATE-Q/000660")
```

NVIDIA 재무제표는 FinanceDataReader 대신 외부 API 사용 권장  <mark> (검증 필요) </mark>

- 우선순위 1: SEC EDGAR Company Facts API
- 우선순위 2: Financial Modeling Prep, Alpha Vantage, yfinance 등 API
- 우선순위 3: 수동 CSV 보강

핵심 재무 변수:

- 매출액, 영업이익, 순이익
- 매출총이익률, 영업이익률, 순이익률
- 재고자산, 재고자산 회전율
- 유형자산, CapEx
- 연구개발비
- 부채비율, 현금성 자산
- EPS, BPS, ROE
- 전년동기대비 성장률, 전분기대비 성장률

반도체 기업은 재고와 마진이 특히 중요. 메모리 기업인 삼성전자·SK하이닉스는 `재고 증가 -> 가격 하락 -> 마진 악화 -> 주가 약세` 경로 자주 나타남.<br> NVIDIA는 재고보다 `데이터센터 매출 성장률`, `매출총이익률`, `수주/공급 제약 관련 코멘트`가 더 강한 신호가 될 수 있음.

### 3. 반도체 종합 지수

| 목적 | 추천 지표 | FinanceDataReader 가능성 | 이유 |
|---|---|---|---|
| 미국 반도체 대표 벤치마크 | PHLX Semiconductor Index, `SOX`/`^SOX` | 환경별 확인 필요 | 반도체 주가 사이클의 대표 지수 |
| 글로벌 반도체 벤치마크 | MSCI ACWI Semiconductors & Semiconductor Equipment | 별도 다운로드/API 필요 | 한국, 미국, 대만, 유럽 장비주를 함께 반영 |
| 거래 가능한 대체 지표 | `SMH`, `SOXX` ETF | `fdr.DataReader("SMH")`, `fdr.DataReader("SOXX")` 확인 | SOX 지수 수집이 불안정할 때 대체 가능 |
| 한국 반도체 상대강도 | KOSPI, KOSPI200 대비 삼성전자/SK하이닉스 | FDR 가능 | 한국 시장 전체와 반도체 대형주 차별화 측정 |

권장 방식:

```python
semi_etfs = {
    "smh": fdr.DataReader("SMH", START, END),
    "soxx": fdr.DataReader("SOXX", START, END),
}
```

SOX 지수가 직접 수집되지 않으면 `SMH` 또는 `SOXX`로 사용. ETF는 운용보수와 구성 종목 변경이 있지만, 실제 투자자 수급을 반영한다는 장점도 존재.

### 4. 글로벌 지수 데이터

반도체 주가는 기술주 베타(시장 민감도에 반응), 경기 민감도, 아시아 제조업 사이클을 모두 가진다. 아래 지수를 함께 수집한다.
<mark> 추가 이해 필요 </mark>
<mark>기술주 베타, 경기 민감도, 아시아 제조업 사이클을 반영할 수 있는 지표를 수집해야 함. 아래 지표는 시장 지표이기 때문에 시장 분위기를 반영하지만 앞서 말한 지표만을 반영할 수 없음</mark>
<mark>글로벌 지수 데이터는 주식 시장 흐름이나 추세를 반영하는 목표로 사용하는 건 어떤지 조사</mark>
```python
global_indices = {
    "kospi": fdr.DataReader("KS11", START, END),
    "kosdaq": fdr.DataReader("KQ11", START, END),
    "kospi200": fdr.DataReader("KS200", START, END),
    "nasdaq": fdr.DataReader("IXIC", START, END),
    "sp500": fdr.DataReader("S&P500", START, END),
    "dow": fdr.DataReader("DJI", START, END),
    "russell2000": fdr.DataReader("RUT", START, END),
    "vix": fdr.DataReader("VIX", START, END),
    "shanghai": fdr.DataReader("SSEC", START, END),
    "hang_seng": fdr.DataReader("HSI", START, END),
    "nikkei": fdr.DataReader("N225", START, END),
}
```

특히 중요한 변수: <mark> 추가 이해 필요 </mark>

- `IXIC`: NVIDIA와 성장주 멀티플에 직접 영향
- `S&P500`: 미국 위험자산 공통 요인
- `KS11`, `KS200`: 삼성전자·SK하이닉스의 국내 시장 공통 요인
- `SSEC`, `HSI`: 중국 수요와 아시아 제조업 심리
- `N225`: 일본 장비/소재 밸류체인 프록시
- `VIX`: 위험회피 심리

### 5. 환율, 금리, 달러

삼성전자와 SK하이닉스는 원화 약세가 단기적으로 매출 환산에 유리할 수 있지만, 글로벌 위험회피와 함께 오면 주가에는 악재가 될 수 있다. 환율은 방향 자체보다 `달러 강세가 위험회피인지, 한국 수출기업 마진 개선인지`를 분리해야 한다.
<mark> 추가 이해 필요 환율은 고차원 데이터를 사용하거나 상호 작용을 반영한 피처를 사용하는 것 조사</mark>


```python
fx_rates = {
    "usdkrw": fdr.DataReader("USD/KRW", START, END),
    "usdcny": fdr.DataReader("USD/CNY", START, END),
    "usdjpy": fdr.DataReader("USD/JPY", START, END),
}

rates = {
    "us_5y": fdr.DataReader("US5YT", START, END),
    "us_10y": fdr.DataReader("US10YT", START, END),
    "us_30y": fdr.DataReader("US30YT", START, END),
}

dxy = fdr.DataReader("^NYICDX", START, END)
```

파생 변수:<mark> 추가 이해 필요 </mark>
- 장기금리 변화율
- 10년물 금리 레벨
- 달러인덱스 변화율
- USD/KRW 변화율
- 환율 변동성
- 원화 약세 국면 더미

### 6. 원자재와 선물 가격

FinanceDataReader로 바로 가져오기 좋은 선물:

```python
futures = {
    "wti": fdr.DataReader("CL=F", START, END),
    "brent": fdr.DataReader("BZ=F", START, END),
    "natural_gas": fdr.DataReader("NG=F", START, END),
    "gold": fdr.DataReader("GC=F", START, END),
    "silver": fdr.DataReader("SI=F", START, END),
    "copper": fdr.DataReader("HG=F", START, END),
}
```

반도체 분석에서 원자재는 직접 원가보다 `경기`, `인플레이션`, `제조업 수요`, `전력비` 대리 요인으로 볼 수 있으며 이 방향이 더 나을 수 있음.

- 구리: 글로벌 제조업/전력 인프라/데이터센터 투자 심리
- 천연가스·유가: 전력비, 인플레이션, 금리 기대
- 금: 위험회피 및 실질금리 대리 요인
- 은: 산업재와 귀금속 성격을 동시에 가짐

별도 소스가 필요한 반도체 특화 원재료:

- 실리콘 웨이퍼 가격/출하량: SEMI, SUMCO 자료
- 네온, 크립톤, 제논 등 희귀가스: 산업가스 가격 데이터, 무역 통계
- 포토레지스트/불화수소: 일본/한국 무역 통계
- 반도체 장비: SEMI billings, ASML/Lam/Applied Materials 실적

### 7. FRED 데이터

FinanceDataReader는 `FRED:` 접두어로 FRED 데이터를 불러올 수 있음.

```python
fred = fdr.DataReader(
    "FRED:FEDFUNDS,DGS10,DGS2,T10Y2Y,VIXCLS,NASDAQCOM,DTWEXBGS,INDPRO,IPG3344S,PCU334413334413P,A34SNO",
    START,
    END,
)
```

FRED 데이터:

| 변수 | FRED 코드 | 의미 |
|---|---|---|
| 기준금리 | `FEDFUNDS` | 할인율/성장주 멀티플 압력 |
| 미국 10년 금리 | `DGS10` | 장기 할인율 |
| 미국 2년 금리 | `DGS2` | 정책금리 기대 |
| 장단기 금리차 | `T10Y2Y` | 경기 침체/회복 국면 |
| VIX | `VIXCLS` | 위험회피 심리 |
| 나스닥 | `NASDAQCOM` | 기술주 공통 요인 |
| 달러 Broad Index | `DTWEXBGS` | 달러 강세/약세 |
| 산업생산 | `INDPRO` | 경기 사이클 |
| 반도체·전자부품 산업생산 | `IPG3344S` | 미국 반도체 생산 사이클 |
| 반도체 PPI | `PCU334413334413P` | 반도체 생산자 가격 |
| 컴퓨터·전자제품 신규주문 | `A34SNO` | 전방 IT 수요 대리 요인 |

주의점:

- 일간 주가와 월간 FRED 데이터는 빈도가 다르므로 월말 기준으로 정렬하거나, FRED 발표일 기준 lag를 줘야 한다.
- 모델에 미래 정보가 들어가지 않도록 월간 지표는 발표 후 사용 가능한 날짜로 밀어야 한다.
- `A34SNO`는 반도체 신규주문이 아니라 컴퓨터·전자제품 신규주문 대리 요인. FRED는 반도체 산업 신규주문 데이터 없음 명시.

### 8. 반도체 수급 데이터

이 프로젝트의 성패를 가르는 핵심 데이터다. 주가, 금리, 지수만 넣으면 “기술주 베타 모델”이 되고, 반도체 도메인 분석의 날카로움이 약해진다.

우선순위:

1. WSTS/SIA 월간 반도체 매출: 제품군별, 지역별 매출/출하/ASP
2. DRAMeXchange/TrendForce: DRAM, NAND, HBM 가격
3. SEMI Billings/WWSEMS: 반도체 장비 billings
4. 한국 관세청/무역협회: 반도체 수출액, 메모리 수출액, 국가별 수출
5. 대만 월간 매출: TSMC, UMC, MediaTek 등 파운드리/팹리스 매출
6. 기업별 출하·재고 코멘트: 실적 발표 자료, 컨퍼런스콜

특히 추천하는 변수:

- DRAM spot price, contract price
- NAND spot price, contract price
- HBM 관련 공급 부족/가격 프리미엄 지표
- global semiconductor sales YoY
- memory sales YoY
- semiconductor inventory-to-sales ratio
- semiconductor equipment billings YoY
- Korea semiconductor exports YoY
- Taiwan foundry revenue YoY

삼성전자와 SK하이닉스는 메모리 가격 민감도가 매우 크고, NVIDIA는 AI 가속기 수요와 HBM 공급망의 병목에 민감하다. 따라서 `DRAM/NAND 가격 -> 삼성전자·SK하이닉스`, `HBM/데이터센터 수요 -> NVIDIA·SK하이닉스`의 경로를 따로 설계하는 것이 좋다.

### 9. 반도체가 들어가는 제품 가격/수요 데이터
<mark> 내용 포함 여부 검증 필요 </mark>

반도체는 최종재 수요가 중요하다. 제품 가격 자체보다 출하량, 재고, 판매량, 교체 사이클이 더 강한 설명력을 가질 가능성이 높다.

권장 수집 항목:

- PC 출하량: Gartner, IDC, Canalys
- 스마트폰 출하량: IDC, Counterpoint, Canalys
- 서버 출하량/클라우드 CapEx: Gartner, Synergy Research, hyperscaler 실적
- GPU/AI 서버 수요: NVIDIA 실적, Supermicro 실적, 클라우드 업체 CapEx
- 자동차 생산량/EV 판매량: OICA, 각국 통계, IEA
- SSD/HDD 출하량과 가격
- consumer electronics retail sales

현실적인 대체 프록시:
<mark> 내용 포함 여부 검증 필요 </mark>
- Apple, Microsoft, Amazon, Google, Meta CapEx (반도체 Capex)
- TSMC 월간 매출
- ASML 수주잔고와 매출
- Micron 실적과 재고
- Dell/Supermicro 서버 관련 매출

## 모델링 관점의 데이터 설계

### 타깃 변수

분석 목적에 따라 타깃을 여러 개 둔다.

- 다음 1일/5일/20일 수익률
- 다음 20일 상승 여부
- 시장 대비 초과수익률
- 큰 하락 이벤트 여부: 예) 20영업일 수익률 -10% 이하
- 변동성 상승 여부

### 빈도 설계

| 빈도 | 사용 데이터 | 목적 |
|---|---|---|
| 일간 | 주가, 지수, 환율, 금리, 선물, VIX | 단기 가격 움직임 |
| 주간 | 주가/지수 리샘플링, 가격 모멘텀 | 노이즈 완화 |
| 월간 | FRED, 수출, 반도체 매출, 장비 billings, 제품 가격 | 업황 사이클 |
| 분기 | 재무제표, 실적, 재고, CapEx | 기업 펀더멘털 |

추천은 `월간 업황 모델 + 일간 시장 모델`을 분리한 뒤, 사용자 화면에서 같이 해석하는 것이다.

- 월간 모델: 반도체 사이클과 펀더멘털 설명
- 일간 모델: 금리, 환율, 시장 심리, 기술주 베타 설명

### 피처 엔지니어링

강한 후보:

- YoY, MoM, QoQ 변화율
- 3개월/6개월 이동평균
- 3개월 변화율의 변화: 업황 개선 속도
- 가격 레벨보다 가격 변화율
- 지표별 z-score
- 반도체 지수 대비 개별 종목 상대강도
- 한국 주식은 USD/KRW와 KOSPI를 통제한 초과수익률
- NVIDIA는 NASDAQ, SOX/SMH, 미국 10년 금리를 통제한 초과수익률
- lag feature: 1개월, 3개월, 6개월 지연값

창의적으로 넣어볼 만한 피처:

- `memory_price_momentum - inventory_growth`: 가격은 오르는데 재고도 줄면 강한 업황 개선 신호
- `semi_equipment_billings_yoy - semi_sales_yoy`: 설비투자가 매출보다 과하게 빠르면 공급과잉 위험
- `USD/KRW 상승 + KOSPI 하락`: 위험회피성 원화 약세 더미<mark> 추가 이해 필요 </mark>
- `NASDAQ 상승 + 금리 상승`: AI/성장 기대가 할인율 부담을 이기는 국면
- `SMH 대비 NVDA 상대강도`: AI 가속기 특수 요인
- `SK하이닉스 대비 삼성전자 상대강도`: HBM/메모리 순수 노출도 차이

### 추천 분석 방법

1. EDA: 상관관계, lead-lag correlation, 국면별 수익률
2. 통계: OLS, Lasso/Ridge, VAR, Granger causality
3. 머신러닝: RandomForest, XGBoost/LightGBM, CatBoost
4. 해석: SHAP, permutation importance
5. 검증: walk-forward validation, time series split

주의: 랜덤 셔플 train/test split은 사용하지 않는다. 시계열에서는 미래 정보가 섞여 성능이 과대평가된다.

## 수집 우선순위

### 1차: FinanceDataReader로 즉시 수집

- 삼성전자, SK하이닉스, NVIDIA 주가
- KOSPI, KOSPI200, NASDAQ, S&P500, VIX
- SMH, SOXX ETF
- USD/KRW, USD/CNY, USD/JPY
- 미국 5년/10년/30년 금리
- 달러인덱스
- WTI, Brent, 천연가스, 금, 은, 구리
- FRED macro/semiconductor series
- 삼성전자, SK하이닉스 재무제표

### 2차: 도메인 핵심 외부 데이터

- WSTS/SIA 월간 반도체 매출
- DRAM/NAND/HBM 가격
- SEMI equipment billings
- 한국 반도체 수출 데이터
- TSMC/ASML/Micron 월간·분기 지표

### 3차: 고급 이벤트 데이터

- 실적 발표일
- 컨퍼런스콜 텍스트
- 수출 규제, CHIPS Act, 대중 제재
- NVIDIA GPU 출시 이벤트
- HBM 공급 계약 뉴스
- 주요 고객사 CapEx 가이던스

## 데이터 저장 구조 제안

```text
data/
  raw/
    fdr/
      prices/
      indices/
      fx/
      futures/
      fred/
      financials/
    external/
      wsts/
      dram_nand_prices/
      semi_billings/
      exports/
  processed/
    daily_panel.parquet
    monthly_panel.parquet
    quarterly_fundamentals.parquet
  metadata/
    source_catalog.csv
    symbol_map.csv
```

`source_catalog.csv`에는 데이터명, 출처, 수집 방식, 빈도, 업데이트 주기, 라이선스, 결측 처리 규칙을 기록한다.

## 초기 구현 예시

```python
import FinanceDataReader as fdr
import pandas as pd

START = "2011-01-01"
END = None

FDR_SERIES = {
    "005930": "samsung",
    "000660": "sk_hynix",
    "NVDA": "nvidia",
    "KS11": "kospi",
    "KS200": "kospi200",
    "IXIC": "nasdaq",
    "S&P500": "sp500",
    "VIX": "vix",
    "SMH": "smh",
    "SOXX": "soxx",
    "USD/KRW": "usdkrw",
    "USD/CNY": "usdcny",
    "US10YT": "us10y",
    "^NYICDX": "dxy",
    "CL=F": "wti",
    "NG=F": "natural_gas",
    "HG=F": "copper",
}

def read_close(symbol: str, name: str) -> pd.Series:
    df = fdr.DataReader(symbol, START, END)
    col = "Close" if "Close" in df.columns else df.columns[0]
    return df[col].rename(name)

daily_panel = pd.concat(
    [read_close(symbol, name) for symbol, name in FDR_SERIES.items()],
    axis=1,
).sort_index()

daily_returns = daily_panel.pct_change()

fred = fdr.DataReader(
    "FRED:FEDFUNDS,DGS10,DGS2,T10Y2Y,VIXCLS,NASDAQCOM,DTWEXBGS,INDPRO,IPG3344S,PCU334413334413P,A34SNO",
    START,
    END,
)
```

## 참고 출처

- FinanceDataReader GitHub: https://github.com/financedata/financedatareader
- FRED Semiconductor Industrial Production `IPG3344S`: https://fred.stlouisfed.org/series/IPG3344S
- FRED Semiconductor PPI `PCU334413334413P`: https://fred.stlouisfed.org/series/PCU334413334413P
- FRED Computers and Electronic Products New Orders `A34SNO`: https://fred.stlouisfed.org/series/A34SNO
- SIA/WSTS Semiconductor Market Data: https://www.semiconductors.org/data-resources/market-data/
- SEMI Billings Report: https://www.semi.org/en/products-services/market-data/equipment/billings-report
- MSCI ACWI Semiconductors & Semiconductor Equipment Index: https://www.msci.com/indexes/index/738842
- Nasdaq PHLX Semiconductor Sector Index methodology note: https://ir.nasdaq.com/news-releases/news-release-details/methodology-change-phlx-semiconductor-sector-index


