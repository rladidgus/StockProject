# 반도체 핵심주 주가 분석 코드 해설

## 문서 목적

이 문서는 지금까지 삼성전자(`005930`)와 SK하이닉스(`000660`) 주가 분석을 위해 작성한 코드가 어떤 역할을 하는지 설명한다.
단순히 실행 순서를 적는 문서가 아니라, 각 파일과 함수가 어떤 데이터를 만들고, 어떤 컬럼이 무엇을 의미하며, AUC와 SHAP이 어느 코드에서 계산되는지 해석하기 위한 문서다.

## 전체 흐름

```text
src/db.py
→ PostgreSQL 테이블 생성

src/collect.py
→ 삼성전자 주가, 거시 지표, 외국인/기관 수급 데이터 수집
→ DB 저장

src/preprocess.py
→ DB에서 데이터 로드
→ 주가 + 거시 + 수급 데이터 결합
→ shift(1), lag feature, 변화율, rolling feature 생성
→ 5거래일 후 상승 여부 target 생성
→ data/processed/samsung_preprocessed.csv 저장
→ data/processed/sk_hynix_preprocessed.csv 저장

src/model.py
→ 전처리 CSV 로드
→ Feature/Target 분리
→ Train/Validation/Test 시간순 분할
→ XGBoost 학습
→ Validation AUC 기준 Optuna 튜닝
→ proposed, proposed_plus, macro_alpha feature set 비교
→ Test 성능 평가
→ outputs/results/model_comparison.csv 저장

src/shap_analysis.py
→ 튜닝된 XGBoost 모델 재학습
→ Test set 기준 SHAP 분석
→ 그래프 저장
→ shap_results 테이블에 SHAP 수치 저장

src/interpret_period.py
→ shap_results에서 사용자가 지정한 날짜 구간 조회
→ 구간별 평균 절대 SHAP 기준 지표 영향력 계산
→ outputs/results/period_feature_importance_*.csv 저장
```

## 1. `src/db.py`: DB 연결과 테이블 생성

### DB 연결 코드

```python
load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
```

이 코드는 `.env` 파일에 있는 DB 접속 정보를 읽어서 PostgreSQL 연결 주소를 만든다.

사용하는 환경변수:

- `DB_USER`: PostgreSQL 사용자명
- `DB_PASSWORD`: PostgreSQL 비밀번호
- `DB_HOST`: DB 서버 주소
- `DB_PORT`: PostgreSQL 포트, 보통 `5432`
- `DB_NAME`: DB 이름, 현재 프로젝트에서는 `semiconductor`

### `get_engine()`

```python
def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)
```

SQLAlchemy 엔진을 생성한다.
다른 파일에서 DB에 접속할 때 공통으로 사용하는 함수다.

### `stock_prices` 테이블

삼성전자, SK하이닉스, NVDA 같은 종목의 일별 주가를 저장하는 테이블이다.

| 컬럼 | 의미 |
|---|---|
| `date` | 거래일 |
| `ticker` | 종목 코드. 삼성전자는 `005930` |
| `open` | 시가 |
| `high` | 고가 |
| `low` | 저가 |
| `close` | 종가 |
| `volume` | 거래량 |
| `return_pct` | 전일 대비 일별 수익률, 퍼센트 단위 |

`PRIMARY KEY (date, ticker)`가 설정되어 있어 같은 날짜, 같은 종목 데이터가 중복 저장되지 않는다.

### `macro_features` 테이블

거시경제 지표를 날짜별로 저장하는 테이블이다.

| 컬럼 | 의미 |
|---|---|
| `date` | 기준 날짜 |
| `nasdaq` | 나스닥 지수 |
| `sox` | 반도체 지수 또는 반도체 관련 지표 |
| `vix` | 변동성 지수 |
| `us_rate` | 미국 기준금리 |
| `usd_krw` | 원/달러 환율 |
| `kospi` | 코스피 지수 |
| `kospi200` | 코스피200 지수 |
| `nasdaq_lag1`, `nasdaq_lag3`, `nasdaq_lag5` | 나스닥 1일, 3일, 5일 전 값 |
| `vix_lag1`, `vix_lag3`, `vix_lag5` | VIX 1일, 3일, 5일 전 값 |
| `us_rate_lag1` | 미국 기준금리 1일 전 값 |
| `usd_krw_lag1` | 환율 1일 전 값 |

현재 실제 전처리에서는 lag feature를 CSV 생성 단계에서 다시 만든다.
DB 테이블에도 일부 lag 컬럼이 있지만, 핵심 lag 생성 로직은 `src/preprocess.py`에 있다.

### `alpha_features` 테이블

수급 데이터를 저장하는 테이블이다.

| 컬럼 | 의미 |
|---|---|
| `date` | 거래일 |
| `ticker` | 종목 코드 |
| `foreign_net_buy` | 외국인 순매수 금액 |
| `institution_net_buy` | 기관 순매수 금액 |
| `put_call_ratio` | Put/Call Ratio. NVDA용 예비 컬럼 |
| `short_interest` | 공매도 잔고 비율. NVDA용 예비 컬럼 |

삼성전자에서는 `foreign_net_buy`, `institution_net_buy`를 사용한다.

### `shap_results` 테이블

SHAP 분석 결과를 날짜, 종목, feature set, 피처별로 저장하는 테이블이다.

| 컬럼 | 의미 |
|---|---|
| `id` | 자동 증가 ID |
| `date` | SHAP을 계산한 Test set 날짜 |
| `ticker` | 종목 코드 |
| `feature_set` | SHAP을 계산한 feature set. 예: `proposed`, `macro_alpha` |
| `feature_name` | 피처 이름 |
| `shap_value` | 해당 피처가 예측값에 기여한 정도 |
| `feature_value` | 그 날짜의 실제 피처 값 |

예를 들어 `feature_name = nasdaq_lag5`이고 `shap_value`가 양수라면, 해당 날짜의 5일 전 나스닥 값이 모델의 상승 예측 확률을 높이는 방향으로 작용했다는 뜻이다.

## 2. `src/collect.py`: 데이터 수집 코드

이 파일은 삼성전자 분석에 필요한 원천 데이터를 수집하고 DB에 저장한다.

### 기본 기간과 티커

```python
START = "2018-01-01"
END = "2024-12-31"
SAMSUNG_TICKER = "005930"
```

분석 기간은 2018년 1월 1일부터 2024년 12월 31일까지다.
삼성전자 티커는 `005930`이다.

### `save_to_db()`

```python
stmt = pg_insert(table).values(records)
stmt = stmt.on_conflict_do_nothing(index_elements=index_col)
```

수집한 DataFrame을 PostgreSQL 테이블에 저장하는 함수다.
`on_conflict_do_nothing`을 사용하기 때문에 primary key가 이미 있는 데이터는 중복 저장하지 않는다.

예를 들어 삼성전자 주가의 primary key는 `date`, `ticker`이므로 같은 날짜의 삼성전자 데이터가 이미 있으면 건너뛴다.

### `collect_stock_price()`

```python
df = fdr.DataReader(ticker, START, END)[["Open", "High", "Low", "Close", "Volume"]]
df.columns = df.columns.str.lower()
df["ticker"] = ticker
df["return_pct"] = df["close"].pct_change() * 100
```

이 코드는 FinanceDataReader로 주가 데이터를 가져온다.

생성되는 컬럼:

- `open`: 시가
- `high`: 고가
- `low`: 저가
- `close`: 종가
- `volume`: 거래량
- `ticker`: 종목 코드
- `return_pct`: 전일 대비 수익률

`return_pct`는 `close.pct_change() * 100`으로 계산한다.
즉 전날 종가 대비 오늘 종가가 몇 퍼센트 변했는지를 나타낸다.

### `collect_samsung_stock_price()`

```python
def collect_samsung_stock_price() -> pd.DataFrame:
    return collect_stock_price(SAMSUNG_TICKER)
```

삼성전자 주가만 수집하는 함수다.
내부적으로 `collect_stock_price("005930")`을 실행한다.

### `collect_macro()`

```python
nasdaq  = fdr.DataReader("IXIC", START, END)["Close"].rename("nasdaq")
sox     = fdr.DataReader("SOXX", START, END)["Close"].rename("sox")
vix     = fdr.DataReader("VIX", START, END)["Close"].rename("vix")
usd_krw = fdr.DataReader("USD/KRW", START, END)["Close"].rename("usd_krw")
us_rate = fdr.DataReader("FRED:FEDFUNDS", START, END)["FEDFUNDS"].rename("us_rate")
```

거시 지표를 수집하는 함수다.

수집 지표:

- `nasdaq`: 나스닥 지수
- `sox`: 반도체 관련 지표
- `vix`: 변동성 지수
- `usd_krw`: 원/달러 환율
- `us_rate`: 미국 기준금리
- `kospi`: 코스피 지수
- `kospi200`: 코스피200 지수

### `collect_alpha_kr()`

```python
df = stock.get_market_trading_value_by_date(
    START.replace("-", ""), END.replace("-", ""), ticker
)
```

PyKRX를 사용해서 국내 종목의 투자자별 거래대금을 가져온다.
삼성전자에서는 외국인과 기관 순매수 데이터를 사용한다.

```python
df = df[["외국인합계", "기관합계"]].rename(
    columns={"외국인합계": "foreign_net_buy", "기관합계": "institution_net_buy"}
)
```

컬럼명 변환:

- `외국인합계` → `foreign_net_buy`
- `기관합계` → `institution_net_buy`

### `collect_samsung_to_db()`

```python
collect_samsung_to_db(include_alpha=True)
```

삼성전자 1차 데이터 수집을 한 번에 실행하는 함수다.

실행 순서:

1. 삼성전자 주가 수집 후 `stock_prices` 저장
2. 거시 지표 수집 후 `macro_features` 저장
3. 삼성전자 수급 수집 후 `alpha_features` 저장

`--skip-alpha` 옵션을 주면 수급 데이터 수집은 건너뛴다.

## 3. `src/preprocess.py`: 전처리와 feature 생성

이 파일은 DB에 저장된 데이터를 불러와 모델 학습용 CSV를 만든다.

최종 산출물:

```text
data/processed/samsung_preprocessed.csv
```

### 주요 상수

```python
SAMSUNG_TICKER = "005930"
TARGET_HORIZON_DAYS = 5
```

현재 전처리 대상은 삼성전자이고, 예측 목표는 5거래일 후 상승 여부다.

### `sync_timeseries()`

```python
if target_ticker in ["005930", "000660"]:
    macro_df = macro_df.shift(1)
```

국내 종목인 삼성전자와 SK하이닉스는 미국 장 마감 이후 한국장이 열리는 시차가 있다.
따라서 미국 거시 지표를 그대로 당일 값으로 쓰면 미래 정보를 사용하는 문제가 생길 수 있다.

이를 막기 위해 국내 종목에서는 거시 지표를 하루 밀어서 `shift(1)` 적용한다.
즉 삼성전자 2024년 1월 5일 예측에는 2024년 1월 4일까지 확인 가능한 미국 지표를 사용한다.

### `make_lag_features()`

```python
df[f"{col}_lag{lag}"] = df[col].shift(lag)
```

거시 지표의 과거 값을 피처로 만든다.

대상 컬럼:

- `nasdaq`
- `vix`
- `us_rate`
- `usd_krw`
- `sox`

lag 종류:

- `lag1`: 1거래일 전 값
- `lag3`: 3거래일 전 값
- `lag5`: 5거래일 전 값

예를 들어 `nasdaq_lag5`는 5거래일 전 나스닥 지수다.
`kospi_lag5`는 5거래일 전 코스피 지수이고, `kospi200_lag5`는 5거래일 전 코스피200 지수다.

### `add_technical_features()`

삼성전자 주가 자체에서 기술적 지표를 만든다.

| 컬럼 | 의미 |
|---|---|
| `ma5` | 5일 이동평균 |
| `ma10` | 10일 이동평균 |
| `ma20` | 20일 이동평균 |
| `rsi` | 14일 기준 RSI |
| `bb_upper` | 볼린저 밴드 상단 |
| `bb_lower` | 볼린저 밴드 하단 |

이 feature들은 주가의 단기 추세, 과열/침체, 변동 범위를 반영한다.

### `add_macro_change_features()`

거시 지표의 절대 수준뿐 아니라 변화율도 만든다.

| 컬럼 | 의미 |
|---|---|
| `nasdaq_return_1d` | 나스닥 1일 변화율 |
| `nasdaq_return_3d` | 나스닥 3일 변화율 |
| `sox_return_1d` | 반도체 지표 1일 변화율 |
| `sox_return_3d` | 반도체 지표 3일 변화율 |
| `vix_change_1d` | VIX 1일 변화량 |
| `vix_change_3d` | VIX 3일 변화량 |
| `usd_krw_return_1d` | 원/달러 환율 1일 변화율 |
| `usd_krw_return_3d` | 원/달러 환율 3일 변화율 |
| `kospi_return_1d` | 코스피 1일 변화율 |
| `kospi_return_3d` | 코스피 3일 변화율 |
| `kospi200_return_1d` | 코스피200 1일 변화율 |
| `kospi200_return_3d` | 코스피200 3일 변화율 |
| `stock_vs_kospi_return_1d` | 대상 종목 일별 수익률에서 코스피 일별 수익률을 뺀 값 |
| `stock_vs_kospi200_return_1d` | 대상 종목 일별 수익률에서 코스피200 일별 수익률을 뺀 값 |

예를 들어 `sox_return_3d`는 최근 3일간 반도체 지표가 얼마나 올랐는지 나타낸다.
반도체 종목은 업황 지수 변화에 민감할 수 있으므로 중요한 후보 피처다.

### `add_alpha_rolling_features()`

외국인과 기관 수급을 누적 관점으로 만든다.

| 컬럼 | 의미 |
|---|---|
| `foreign_net_buy_5d_sum` | 최근 5거래일 외국인 순매수 합계 |
| `foreign_net_buy_20d_sum` | 최근 20거래일 외국인 순매수 합계 |
| `institution_net_buy_5d_sum` | 최근 5거래일 기관 순매수 합계 |
| `institution_net_buy_20d_sum` | 최근 20거래일 기관 순매수 합계 |
| `trading_value` | 종가 × 거래량으로 계산한 추정 거래대금 |
| `foreign_net_buy_to_volume` | 외국인 순매수를 거래량으로 나눈 값 |
| `institution_net_buy_to_volume` | 기관 순매수를 거래량으로 나눈 값 |
| `foreign_net_buy_to_trading_value` | 외국인 순매수를 추정 거래대금으로 나눈 값 |
| `institution_net_buy_to_trading_value` | 기관 순매수를 추정 거래대금으로 나눈 값 |
| `foreign_net_buy_5d_to_trading_value_5d` | 최근 5거래일 외국인 순매수 합계를 최근 5거래일 거래대금 합계로 나눈 값 |
| `institution_net_buy_20d_to_trading_value_20d` | 최근 20거래일 기관 순매수 합계를 최근 20거래일 거래대금 합계로 나눈 값 |

단일 날짜의 순매수보다 최근 며칠간의 누적 수급이 주가 방향에 더 의미 있을 수 있어서 추가한 feature다.

### `load_stock_prices()`

```sql
SELECT date, open, high, low, close, volume, return_pct
FROM stock_prices
WHERE ticker = %(ticker)s
ORDER BY date
```

`stock_prices` 테이블에서 삼성전자 주가를 날짜순으로 불러온다.

### `load_macro_features()`

```sql
SELECT date, nasdaq, sox, vix, us_rate, usd_krw
FROM macro_features
ORDER BY date
```

거시 지표 테이블에서 모델에 사용할 기본 거시 컬럼을 불러온다.

### `load_alpha_features()`

```sql
SELECT date, foreign_net_buy, institution_net_buy
FROM alpha_features
WHERE ticker = %(ticker)s
ORDER BY date
```

삼성전자의 외국인/기관 수급 데이터를 불러온다.

### `build_samsung_dataset()`

전처리의 핵심 함수다.

실행 순서:

1. `stock_prices`에서 삼성전자 주가 로드
2. `macro_features`에서 거시 지표 로드
3. `alpha_features`에서 수급 지표 로드
4. 국내 종목 기준으로 거시 지표 `shift(1)`
5. 거시 지표 lag feature 생성
6. 삼성전자 거래일 기준으로 거시 지표 날짜 정렬
7. 결측치는 앞 날짜 값으로 채움, 즉 forward-fill
8. 거시 변화율 feature 생성
9. 주가와 거시 지표 결합
10. 기술적 지표 생성
11. 수급 지표 결합
12. 수급 rolling feature 생성
13. 5거래일 후 종가 기준 target 생성
14. 결측치 제거
15. 최종 DataFrame 반환

### Target 생성 코드

```python
future_close = feature_df["close"].shift(-TARGET_HORIZON_DAYS)
feature_df["target_up"] = (future_close > feature_df["close"]).astype("Int64")
feature_df["next_return_pct"] = (future_close / feature_df["close"] - 1) * 100
```

이 부분이 예측 대상인 `target_up`을 만드는 코드다.

- `future_close`: 5거래일 후 종가
- `target_up`: 5거래일 후 종가가 현재 종가보다 높으면 `1`, 아니면 `0`
- `next_return_pct`: 현재 종가 대비 5거래일 후 수익률

예를 들어 오늘 종가가 70,000원이고 5거래일 후 종가가 72,000원이면:

- `target_up = 1`
- `next_return_pct = 약 2.86`

### `save_preprocessed_dataset()`

```python
df.to_csv(output_path, index_label="date")
```

전처리된 데이터를 CSV로 저장한다.
현재 저장 위치는 `data/processed/samsung_preprocessed.csv`다.

## 4. `data/processed/samsung_preprocessed.csv`: 최종 학습 데이터

이 CSV는 모델 학습에 직접 사용되는 최종 데이터다.

주요 컬럼 그룹:

### 주가 컬럼

| 컬럼 | 의미 |
|---|---|
| `open` | 시가 |
| `high` | 고가 |
| `low` | 저가 |
| `close` | 종가 |
| `volume` | 거래량 |
| `return_pct` | 일별 수익률 |

### 거시 지표 컬럼

| 컬럼 | 의미 |
|---|---|
| `nasdaq` | shift(1) 적용 후 나스닥 지수 |
| `sox` | shift(1) 적용 후 반도체 지표 |
| `vix` | shift(1) 적용 후 변동성 지수 |
| `us_rate` | shift(1) 적용 후 미국 기준금리 |
| `usd_krw` | shift(1) 적용 후 원/달러 환율 |

### lag 컬럼

예시:

- `nasdaq_lag1`: 1거래일 전 나스닥
- `nasdaq_lag3`: 3거래일 전 나스닥
- `nasdaq_lag5`: 5거래일 전 나스닥
- `vix_lag5`: 5거래일 전 VIX
- `sox_lag5`: 5거래일 전 반도체 지표

### 수급 컬럼

| 컬럼 | 의미 |
|---|---|
| `foreign_net_buy` | 외국인 순매수 |
| `institution_net_buy` | 기관 순매수 |
| `foreign_net_buy_5d_sum` | 최근 5거래일 외국인 순매수 합계 |
| `foreign_net_buy_20d_sum` | 최근 20거래일 외국인 순매수 합계 |
| `institution_net_buy_5d_sum` | 최근 5거래일 기관 순매수 합계 |
| `institution_net_buy_20d_sum` | 최근 20거래일 기관 순매수 합계 |

### 타겟 컬럼

| 컬럼 | 의미 |
|---|---|
| `target_up` | 5거래일 후 상승 여부. 상승이면 `1`, 아니면 `0` |
| `next_return_pct` | 5거래일 후 수익률 |

`target_up`은 모델이 맞혀야 하는 정답이고, `next_return_pct`는 수익률 확인용 보조 컬럼이다.
모델 feature에서는 둘 다 제외한다.

## 5. `src/model.py`: 모델 학습과 AUC 계산

이 파일은 XGBoost 모델을 학습하고 성능을 평가한다.

### Feature set 정의

```python
BASELINE_FEATURES = ["ma5", "ma10", "ma20", "rsi", "bb_upper", "bb_lower"]
```

`baseline`은 기술적 지표만 사용하는 모델이다.

```python
PROPOSED_FEATURES = [
    "open", "high", "low", "close", "volume", "return_pct",
    "nasdaq", "sox", "vix", "us_rate", "usd_krw",
    ...
]
```

`proposed`는 주가, 거시 지표, 수급 지표를 포함하는 모델이다.
이 프로젝트의 핵심 목적인 거시/수급 지표 영향력 분석에 사용하는 후보 모델이다.

```python
MACRO_ALPHA_FEATURES = [
    feature
    for feature in PROPOSED_FEATURES
    if feature not in {"open", "high", "low", "close", "trading_value"}
]
```

`macro_alpha`는 가격 레벨 피처를 제외하고 거시/수급 중심으로 해석하기 위한 feature set이다.
현재 프로젝트의 핵심 질문인 “거시경제 지표와 수급 지표가 종목별로 어떻게 다르게 작용하는가”를 보기에는 `proposed`보다 `macro_alpha`가 더 적합하다.

제외한 컬럼:

- `open`: 시가
- `high`: 고가
- `low`: 저가
- `close`: 종가
- `trading_value`: 종가 × 거래량으로 만든 추정 거래대금

```python
PROPOSED_PLUS_FEATURES = PROPOSED_FEATURES + BASELINE_FEATURES
```

`proposed_plus`는 proposed feature에 기술적 지표까지 추가한 모델이다.

### `load_preprocessed_data()`

```python
df = pd.read_csv(path, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
```

전처리 CSV를 불러오고 날짜순으로 정렬한다.
시계열 분석에서는 날짜 순서가 중요하므로 정렬을 먼저 한다.

### `split_features_target()`

```python
feature_df = df[selected_columns]
target = df[target_col].astype(int)
```

모델 입력값 `X`와 정답값 `y`를 분리한다.

- `X`: 주가, 거시, 수급 등 feature
- `y`: `target_up`

제외 컬럼:

- `date`: 날짜는 직접 숫자 feature로 사용하지 않음
- `target_up`: 정답 컬럼이므로 feature에서 제외
- `next_return_pct`: 미래 수익률이라 feature에 넣으면 데이터 누수 발생

### `split_train_val_test()`

```python
train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))
```

데이터를 시간 순서대로 나눈다.

- Train: 앞 70%
- Validation: 다음 15%
- Test: 마지막 15%

랜덤 셔플을 하지 않는 이유는 주가 데이터가 시계열이기 때문이다.
미래 데이터를 학습에 섞으면 실제 예측 상황과 달라진다.

### `make_default_xgb_model()`

기본 XGBoost 모델을 만든다.

```python
XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    ...
)
```

주요 파라미터:

- `n_estimators`: 트리 개수
- `max_depth`: 트리 깊이
- `learning_rate`: 학습률
- `subsample`: 학습할 때 사용할 행 비율
- `colsample_bytree`: 학습할 때 사용할 컬럼 비율
- `scale_pos_weight`: 상승/하락 클래스 불균형 보정
- `eval_metric`: 모델 내부 평가 지표

### `evaluate_classifier()`: 성능 지표 계산 코드

```python
proba = model.predict_proba(X)[:, 1]
pred = (proba >= threshold).astype(int)
```

모델은 각 날짜에 대해 상승 확률을 계산한다.
`threshold`보다 확률이 높으면 상승 `1`, 낮으면 하락 `0`으로 분류한다.

```python
"accuracy": accuracy_score(y, pred)
"precision": precision_score(y, pred)
"recall": recall_score(y, pred)
"f1": f1_score(y, pred)
"auc": roc_auc_score(y, proba)
```

여기서 AUC가 계산된다.

각 지표 의미:

| 지표 | 의미 |
|---|---|
| `accuracy` | 전체 예측 중 맞힌 비율 |
| `precision` | 상승이라고 예측한 것 중 실제 상승한 비율 |
| `recall` | 실제 상승한 날 중 모델이 상승이라고 잡아낸 비율 |
| `f1` | precision과 recall의 균형 지표 |
| `auc` | 상승 확률 순위가 실제 상승/하락을 얼마나 잘 구분하는지 나타내는 지표 |
| `threshold` | 상승/하락을 나누는 확률 기준 |

AUC는 threshold와 직접적인 관련이 적다.
즉 threshold를 0.10으로 낮춰 F1이나 Recall이 바뀌어도, AUC는 상승 확률의 순위 자체를 평가한다.

### `find_best_threshold()`

```python
candidates = [i / 100 for i in range(10, 91)]
scores = [(threshold, f1_score(y_val, proba >= threshold)) for threshold in candidates]
```

Validation set에서 F1-score가 가장 높은 threshold를 찾는다.

현재 구조는 다음처럼 나뉜다.

- 하이퍼파라미터 튜닝 기준: AUC
- 최종 분류 threshold 선택 기준: F1-score

이렇게 한 이유는 AUC로 확률 순위를 개선하고, 실제 상승/하락 라벨을 찍을 때는 F1-score를 고려하기 위해서다.

### `build_objective()`: AUC 기준 Optuna 튜닝 코드

```python
from sklearn.metrics import roc_auc_score
```

이 함수가 AUC 기준 튜닝의 핵심이다.

```python
val_proba = model.predict_proba(X_val)[:, 1]
return roc_auc_score(y_val, val_proba)
```

Optuna가 여러 파라미터 조합을 시도할 때, Validation AUC가 가장 높은 조합을 선택한다.

즉 이 코드는 “AUC를 만든 코드”라기보다 정확히는 “AUC를 기준으로 좋은 XGBoost 파라미터를 찾는 코드”다.

### `train_tuned_xgboost()`

```python
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(build_objective(...), n_trials=n_trials)
```

Optuna 튜닝을 실행한다.
`direction="maximize"`이므로 AUC가 클수록 좋은 모델로 판단한다.

```python
best_params = study.best_params | {
    "scale_pos_weight": scale_weight,
    "random_state": 42,
    "eval_metric": "auc",
}
```

가장 좋은 파라미터 조합에 클래스 불균형 보정값과 평가 지표를 추가한다.

### `run_tuned_pipeline()`

모델링 전체 과정을 한 번에 실행하는 함수다.

실행 순서:

1. feature set 선택
2. Train/Validation/Test 분리
3. Train과 Validation으로 Optuna 튜닝
4. Validation F1 기준 threshold 선택
5. Train + Validation 데이터로 최종 모델 재학습
6. Test set 성능 평가
7. 결과 CSV 저장

### `model_comparison.csv`

모델 성능 결과가 저장되는 파일이다.

현재 주요 컬럼:

| 컬럼 | 의미 |
|---|---|
| `ticker` | 종목 코드 |
| `model` | 모델 이름 |
| `accuracy` | 정확도 |
| `precision` | 정밀도 |
| `recall` | 재현율 |
| `f1` | F1-score |
| `auc` | ROC-AUC |
| `threshold` | 분류 기준값 |

현재 결과 예시:

```csv
005930,xgboost_tuned_train_val_macro_alpha,0.4392156862745098,0.4351464435146444,0.9285714285714286,0.5925925925925926,0.4749000999000999,0.18
000660,xgboost_tuned_train_val_macro_alpha,0.5450980392156862,0.5531914893617021,0.9219858156028369,0.6914893617021277,0.5417429898100079,0.13
```

해석:

- 삼성전자 `macro_alpha` AUC는 `0.4749`로 예측력은 약하다.
- SK하이닉스 `macro_alpha` AUC는 `0.5417`로 삼성전자보다 높지만, 강한 예측 모델이라고 보기는 아직 어렵다.
- 두 모델 모두 Recall이 높고 Precision이 낮아 상승 신호를 넓게 잡는 편이다.
- 이 모델은 예측 성능만 보면 강하지 않지만, 거시/수급 지표 영향력 분석용 SHAP 모델 후보로 사용한다.

### `run_statistical_validation()`: 보조 검증 코드

XGBoost는 비선형 관계와 feature 간 상호작용을 잘 잡을 수 있지만, 해석이 복잡하다.
그래서 보조 검증용으로 Logistic Regression과 feature-target 상관분석을 추가했다.

```python
run_statistical_validation(data, feature_set="proposed")
```

이 함수의 목적은 다음과 같다.

1. XGBoost + SHAP에서 중요하게 나온 피처가 단순 선형 모델에서도 의미가 있는지 확인한다.
2. 각 피처와 `target_up` 사이의 단순 상관관계를 확인한다.
3. XGBoost 결과만 보고 과도하게 해석하지 않도록 보조 근거를 만든다.

### `save_feature_target_correlation()`

```python
corr = X_train[feature].corr(y_train)
```

이 코드는 Train 구간에서 각 feature와 `target_up`의 피어슨 상관계수를 계산한다.
미래 Test 데이터를 해석에 섞지 않기 위해 Train 데이터만 사용한다.

저장 파일:

```text
outputs/results/feature_target_correlation.csv
```

주요 컬럼:

| 컬럼 | 의미 |
|---|---|
| `ticker` | 종목 코드 |
| `feature_set` | 사용한 feature set |
| `feature` | 피처 이름 |
| `pearson_corr_train` | Train 구간에서 target과의 상관계수 |
| `abs_pearson_corr_train` | 상관계수 절댓값 |

현재 상관분석 결과는 `feature_set`과 `ticker`별로 저장된다.
예를 들어 `macro_alpha` 기준에서는 삼성전자와 SK하이닉스의 상위 피처를 따로 비교할 수 있다.

### `train_logistic_regression()`

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )),
])
```

Logistic Regression은 선형 분류 모델이다.
XGBoost보다 단순하지만, 각 feature의 계수 방향과 크기를 보기 쉽다.

- `StandardScaler`: feature 단위를 표준화한다.
- `class_weight="balanced"`: 상승/하락 클래스 불균형을 보정한다.
- `coef_`: 각 feature가 상승 확률에 주는 선형 영향 방향을 나타낸다.

저장 파일:

```text
outputs/results/logistic_coefficients.csv
```

현재 Logistic Regression 계수 절댓값 상위 피처는 `close`, `sox_lag5`, `nasdaq_lag1`, `us_rate_lag1`, `us_rate_lag3` 등이다.

### 보조 검증 결과 해석

현재 Logistic Regression `macro_alpha` 모델의 Test 결과 예시:

| 지표 | 값 |
|---|---|
| 삼성전자 ROC-AUC | `0.4750` |
| 삼성전자 F1-score | `0.6154` |
| SK하이닉스 ROC-AUC | `0.5286` |
| SK하이닉스 F1-score | `0.6745` |

Logistic Regression도 Test AUC가 높지는 않다.
따라서 현재 데이터에서는 “모델 종류를 XGBoost에서 선형 모델로 바꾸면 예측력이 해결된다”고 보기는 어렵다.
이 결과는 예측 성능 자체보다 지표 영향력 탐색에 초점을 맞춰야 한다는 근거가 된다.

## 6. `src/shap_analysis.py`: SHAP 영향력 분석

이 파일은 최종 모델을 다시 학습한 뒤 SHAP 값을 계산한다.

### `BEST_PARAMS_BY_TICKER_FEATURE_SET`

```python
BEST_PARAMS_BY_TICKER_FEATURE_SET = {
    ("005930", "proposed"): {...},
    ("000660", "proposed"): {...},
    ("005930", "macro_alpha"): {...},
    ("000660", "macro_alpha"): {...},
}
```

티커와 feature set별로 AUC 기준 Optuna 튜닝에서 선택된 파라미터를 저장한다.
SHAP 분석에서는 해당 조합의 파라미터로 모델을 다시 학습한다.

### `prepare_final_model_and_test_data()`

```python
X, y = split_features_target(data, feature_set=feature_set)
splits = split_train_val_test(X, y, data["date"])
X_train_val, y_train_val = combine_train_validation(splits)
model = train_xgboost_with_params(X_train_val, y_train_val, params)
```

SHAP 분석용 최종 모델을 준비한다.

특징:

- feature set은 실행 옵션으로 선택한다. 현재 `proposed`, `macro_alpha`를 사용할 수 있다.
- Train + Validation 데이터를 합쳐 최종 모델을 학습한다.
- Test set은 SHAP 값을 계산하는 해석 대상이다.

### `global_shap()`

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

XGBoost 모델을 SHAP으로 해석하는 코드다.

```python
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
```

전체 Test set에서 평균적으로 가장 영향이 큰 feature를 bar plot으로 만든다.
저장 파일:

```text
outputs/figures/samsung_global_bar.png
```

```python
shap.summary_plot(shap_values, X_test, show=False)
```

각 feature가 예측에 어떤 방향으로 영향을 주는지 beeswarm plot으로 만든다.
저장 파일:

```text
outputs/figures/samsung_global_beeswarm.png
```

### `local_shap()`

특정 날짜 하나에 대해 어떤 feature가 상승/하락 예측에 영향을 줬는지 보여준다.

```python
shap.waterfall_plot(...)
```

현재 케이스 스터디:

- `2022-06-15`: 금리 급등기. 다만 현재 Test set에 없어서 건너뜀
- `2024-03-08`: AI 반도체 랠리 시점

저장 파일 예시:

```text
outputs/figures/samsung_local_ai_rally_2024.png
```

### `sanity_check()`

```python
corr = pd.Series(shap_values[:, X_test.columns.get_loc(feature_col)]).corr(X_test[feature_col])
```

특정 feature 값과 SHAP 값의 상관관계를 확인한다.
경제 상식과 방향이 맞는지 점검하기 위한 코드다.

현재 점검:

- `us_rate`: 금리가 높을수록 주가에는 부정적일 것으로 예상
- `vix`: 변동성이 높을수록 주가에는 부정적일 것으로 예상

결과가 negative면 예상 방향과 일치한다.

### `save_shap_to_db()`

```python
rows = [
    {
        "date": str(date_index[i]),
        "ticker": ticker,
        "feature_set": feature_set,
        "feature_name": feature,
        "shap_value": float(shap_values[i, j]),
        "feature_value": float(X_test.iloc[i, j]),
    }
    for i, date in enumerate(X_test.index)
    for j, feature in enumerate(X_test.columns)
]
```

각 날짜와 각 feature별 SHAP 값을 row 형태로 만든다.

예를 들어 Test set 날짜가 255일이고 feature가 63개이면:

```text
255일 × 63개 feature = 16,065행
```

현재 `proposed`는 63개 feature라서 종목별 16,065행이고, `macro_alpha`는 58개 feature라서 종목별 14,790행이다.

```python
DELETE FROM shap_results
WHERE ticker = :ticker AND feature_set = :feature_set
```

같은 종목과 같은 feature set의 기존 SHAP 결과를 삭제한 뒤 새 결과를 저장한다.
그래서 `proposed`와 `macro_alpha` 결과가 서로 덮어쓰이지 않는다.

## 7. 현재 `macro_alpha` SHAP 해석

가격 레벨 피처를 제외한 `macro_alpha` 기준으로 보면 종목별 상위 영향 지표가 다르게 나타난다.

### 삼성전자

현재 Test set 기준 평균 절대 SHAP 상위 피처:

| 순위 | 피처 | 의미 |
|---|---|---|
| 1 | `nasdaq_lag5` | 5거래일 전 나스닥 지수 |
| 2 | `nasdaq_lag3` | 3거래일 전 나스닥 지수 |
| 3 | `foreign_net_buy_5d_sum` | 최근 5거래일 외국인 순매수 누적 |
| 4 | `us_rate_lag5` | 5거래일 전 미국 기준금리 |
| 5 | `institution_net_buy_20d_to_trading_value_20d` | 최근 20거래일 기관 순매수 강도 |

해석:

- 삼성전자 모델에서는 미국 기술주 흐름을 나타내는 나스닥 lag feature, 미국 금리 lag, 외국인 누적 수급, 기관 수급 강도가 크게 사용됐다.
- 기관과 외국인 수급 누적값도 주요 피처로 나타났다.
- 즉 현재 모델은 삼성전자 5거래일 후 방향성을 판단할 때 거시 지표 중 나스닥 흐름, 수급 지표 중 기관/외국인 매매 흐름을 많이 참고한다.

### SK하이닉스

현재 Test set 기준 평균 절대 SHAP 상위 피처:

| 순위 | 피처 | 의미 |
|---|---|---|
| 1 | `usd_krw_lag5` | 5거래일 전 원/달러 환율 |
| 2 | `nasdaq_lag1` | 1거래일 전 나스닥 지수 |
| 3 | `kospi200_return_3d` | 코스피200 3일 수익률 |
| 4 | `sox` | 반도체 지표 |
| 5 | `foreign_net_buy_5d_sum` | 최근 5거래일 외국인 순매수 누적 |

해석:

- SK하이닉스는 삼성전자보다 환율, 반도체 지수, 국내 시장 흐름의 영향이 더 크게 잡혔다.
- `foreign_net_buy_5d_sum`은 삼성전자와 SK하이닉스 모두에서 상위권이므로, 외국인 단기 수급은 두 국내 반도체 종목의 공통 중요 피처로 볼 수 있다.

주의:

- SHAP 값이 크다는 것은 모델이 그 피처를 많이 사용했다는 뜻이다.
- 하지만 Test AUC가 낮기 때문에 “이 지표가 실제 예측을 매우 잘한다”고 단정하면 안 된다.
- 현재는 “예측력이 강한 모델”이라기보다 “거시/수급 지표가 모델 내부에서 어떤 영향력을 가졌는지 확인하는 모델”에 가깝다.

## 8. `src/interpret_period.py`: 날짜 구간별 영향력 조회

이 파일은 이미 DB에 저장된 `shap_results`를 사용해 사용자가 지정한 날짜 구간의 지표 영향력을 계산한다.

### 실행 예시

```bash
python src/interpret_period.py \
  --ticker 005930 \
  --feature-set macro_alpha \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --top-n 10
```

### 핵심 SQL

```sql
SELECT
    feature_name,
    COUNT(*) AS sample_count,
    AVG(ABS(shap_value)) AS mean_abs_shap,
    AVG(shap_value) AS mean_shap,
    AVG(feature_value) AS mean_feature_value
FROM shap_results
WHERE ticker = :ticker
  AND feature_set = :feature_set
  AND date BETWEEN :start_date AND :end_date
GROUP BY feature_name
ORDER BY mean_abs_shap DESC
```

이 쿼리는 지정한 구간 안에서 feature별 평균 절대 SHAP을 계산한다.
`mean_abs_shap`이 클수록 해당 기간에 모델이 그 지표를 크게 사용했다는 뜻이다.

### 출력 컬럼

| 컬럼 | 의미 |
|---|---|
| `feature_name` | 지표 이름 |
| `sample_count` | 해당 구간에서 계산된 날짜 수 |
| `mean_abs_shap` | 평균 절대 SHAP. 영향력 크기 |
| `mean_shap` | 평균 SHAP. 양수면 상승 예측 방향, 음수면 하락 예측 방향 |
| `mean_feature_value` | 해당 구간의 평균 feature 값 |
| `positive_shap_ratio` | 해당 feature의 SHAP 값이 양수였던 비율 |
| `negative_shap_ratio` | 해당 feature의 SHAP 값이 음수였던 비율 |
| `dominant_direction` | 평균 SHAP 기준 방향. `positive`, `negative`, `neutral` |

### 2024년 구간 예시

삼성전자 2024년 `macro_alpha` 상위 피처:

- `nasdaq_lag5`
- `nasdaq_lag3`
- `foreign_net_buy_5d_sum`
- `us_rate_lag5`
- `institution_net_buy_20d_to_trading_value_20d`

SK하이닉스 2024년 `macro_alpha` 상위 피처:

- `usd_krw_lag5`
- `nasdaq_lag1`
- `sox`
- `kospi200_return_3d`
- `foreign_net_buy_5d_sum`

저장 파일:

```text
outputs/results/period_feature_importance_samsung_macro_alpha_2024-01-01_2024-12-31.csv
outputs/results/period_feature_importance_sk_hynix_macro_alpha_2024-01-01_2024-12-31.csv
```

## 9. 현재 코드 기준 한계

### 1. 국내 종목 Test AUC가 높지 않음

삼성전자 `macro_alpha` Test AUC는 `0.4749`, SK하이닉스 `macro_alpha` Test AUC는 `0.5417`이다.
따라서 상승/하락 구분 능력이 강한 예측 모델이라고 보기는 어렵다.

### 2. 거시/수급 영향력 분석과 예측 성능은 다름

SHAP 상위 피처가 의미 있어 보여도, 모델의 Test AUC가 낮으면 예측 모델로 강하다고 말하기 어렵다.
따라서 결과 해석 시 “영향력 분석”과 “예측 성능”을 구분해야 한다.

### 3. 국내 2종목만으로 결론 내리기 어려움

프로젝트 목표는 삼성전자, SK하이닉스, NVDA 비교다.
삼성전자와 SK하이닉스만 보고 반도체 핵심주 전체 특성을 일반화하면 안 된다.

## 10. 다음 코드 확장 방향

### SK하이닉스 확장 완료

SK하이닉스는 삼성전자와 같은 국내 종목이므로 동일한 구조를 적용했다.

완료된 작업:

- `000660` 주가 수집
- `000660` 수급 수집
- 동일한 `shift(1)` 적용
- 동일한 feature 생성
- 동일한 XGBoost 모델링
- 동일한 SHAP 분석
- 동일한 날짜 구간별 지표 영향력 조회

### NVDA 확장

NVDA는 미국 종목이므로 삼성전자와 다르다.

차이점:

- 미국 거시 지표 `shift(1)` 방식이 다를 수 있다.
- KRX 외국인/기관 수급 데이터는 사용할 수 없다.
- 대신 Put/Call Ratio, Short Interest 같은 미국 알파 피처가 필요하다.

### 예측 성능 개선 후보

- KRX 반도체 지수 추가
- 외국인/기관 순매수를 거래대금 대비 비율로 변환
- 중립 구간 제거 타겟 적용
- 10거래일 후 상승 타겟 실험
- walk-forward validation 적용
- 종목별 모델과 통합 모델 비교
