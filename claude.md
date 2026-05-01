# 거시경제 지표 기반 반도체 핵심주 요인 영향력 분석

## 프로젝트 파이프라인 가이드 (claude.md)

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 분석 대상 | 삼성전자(005930), SK하이닉스(000660), 엔비디아(NVDA) |
| 분석 기간 | 약 7년 (2018-01-01 ~ 2024-12-31) |
| 핵심 목표 | 거시경제 지표가 각 종목 주가에 미치는 영향력 수치화 (SHAP) |
| 사용 언어 | Python 3.10+ |
| DB | PostgreSQL (DB명: `semiconductor`) |

---

## 🗂️ 프로젝트 폴더 구조

```
project/
├── claude.md                  # 이 파일 (파이프라인 전체 가이드)
├── .env                       # DB 연결 정보 (Git 제외 필수)
├── .env.example               # 연결 정보 템플릿 (Git 포함)
├── .gitignore                 # .env 반드시 추가
├── data/
│   └── raw/                   # 수집된 원본 CSV 백업 (선택)
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_shap_analysis.ipynb
├── src/
│   ├── db.py                  # DB 연결 및 테이블 생성
│   ├── collect.py             # 데이터 수집
│   ├── preprocess.py          # 전처리
│   ├── model.py               # XGBoost 모델링
│   └── shap_analysis.py       # SHAP 분석 및 시각화
├── outputs/
│   ├── figures/               # SHAP 시각화 저장
│   └── results/               # 분석 결과 CSV 저장
└── requirements.txt
```

---

## 🔐 환경변수 설정

### `.env` 파일 (로컬에만 보관, Git에 올리면 안 됨)

```
DB_USER=postgres
DB_PASSWORD=yourpassword
DB_HOST=localhost
DB_PORT=5432
DB_NAME=semiconductor
DART_API_KEY=your_dart_api_key   # OpenDartReader API 키
```

### `.env.example` 파일 (Git에 포함 — 팀원 공유용 템플릿)

```
DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=5432
DB_NAME=semiconductor
DART_API_KEY=
```

### `.gitignore` 에 반드시 추가

```
.env
__pycache__/
*.pyc
outputs/figures/
```

---

## 🛢️ DB 설계 (PostgreSQL)

### DB를 쓰는 이유

- 일별 시계열 데이터 중복 수집 방지 (PRIMARY KEY로 날짜+종목 관리)
- SHAP 결과값을 저장해 재분석 없이 빠른 조회 가능
- CSV 파일 난립 방지 및 데이터 일관성 유지
- 팀원 3명이 동일 서버 DB에 동시 접속 가능 (SQLite와의 핵심 차이)

### PostgreSQL 설치 및 DB 생성

```bash
# macOS
brew install postgresql
brew services start postgresql

# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql

# DB 및 유저 생성
psql -U postgres
CREATE DATABASE semiconductor;
CREATE USER your_user WITH PASSWORD 'yourpassword';
GRANT ALL PRIVILEGES ON DATABASE semiconductor TO your_user;
\q
```

### 테이블 구조

```sql
-- 1. 주가 원본 테이블
CREATE TABLE IF NOT EXISTS stock_prices (
    date        DATE          NOT NULL,
    ticker      VARCHAR(10)   NOT NULL,   -- '005930', '000660', 'NVDA'
    open        NUMERIC(12,4),
    high        NUMERIC(12,4),
    low         NUMERIC(12,4),
    close       NUMERIC(12,4),
    volume      BIGINT,
    return_pct  NUMERIC(8,4),             -- 일별 등락률 (%)
    PRIMARY KEY (date, ticker)
);

-- 2. 거시경제 지표 테이블
CREATE TABLE IF NOT EXISTS macro_features (
    date            DATE          PRIMARY KEY,
    nasdaq          NUMERIC(12,4),        -- 나스닥 지수
    sox             NUMERIC(12,4),        -- 필라델피아 반도체 지수
    vix             NUMERIC(8,4),         -- 변동성 지수
    us_rate         NUMERIC(6,4),         -- 미국 기준금리
    usd_krw         NUMERIC(10,4),        -- 환율 (USD/KRW)
    -- Lag features (전처리 단계에서 자동 생성)
    nasdaq_lag1     NUMERIC(12,4),
    nasdaq_lag3     NUMERIC(12,4),
    nasdaq_lag5     NUMERIC(12,4),
    vix_lag1        NUMERIC(8,4),
    vix_lag3        NUMERIC(8,4),
    vix_lag5        NUMERIC(8,4),
    us_rate_lag1    NUMERIC(6,4),
    usd_krw_lag1    NUMERIC(10,4)
);

-- 3. 미시 피처 테이블 (분기별 실적)
CREATE TABLE IF NOT EXISTS micro_features (
    date             DATE          NOT NULL,
    ticker           VARCHAR(10)   NOT NULL,
    revenue          NUMERIC(20,2),        -- 매출
    operating_profit NUMERIC(20,2),        -- 영업이익
    eps              NUMERIC(12,4),        -- 주당순이익
    PRIMARY KEY (date, ticker)
);

-- 4. 알파 피처 테이블 (수급)
CREATE TABLE IF NOT EXISTS alpha_features (
    date                DATE          NOT NULL,
    ticker              VARCHAR(10)   NOT NULL,
    foreign_net_buy     NUMERIC(20,2),     -- 외인 순매수 (국내 종목만)
    institution_net_buy NUMERIC(20,2),     -- 기관 순매수 (국내 종목만)
    put_call_ratio      NUMERIC(8,4),      -- Put/Call Ratio (NVDA만)
    short_interest      NUMERIC(8,4),      -- 공매도 잔고 비율 (NVDA만)
    PRIMARY KEY (date, ticker)
);

-- 5. SHAP 결과 저장 테이블
CREATE TABLE IF NOT EXISTS shap_results (
    id              SERIAL        PRIMARY KEY,   -- PostgreSQL: SERIAL (SQLite AUTOINCREMENT 대체)
    date            DATE          NOT NULL,
    ticker          VARCHAR(10)   NOT NULL,
    feature_name    VARCHAR(50)   NOT NULL,
    shap_value      NUMERIC(12,6),
    feature_value   NUMERIC(12,6)
);

-- 조회 성능을 위한 인덱스
CREATE INDEX IF NOT EXISTS idx_shap_ticker_date ON shap_results (ticker, date);
CREATE INDEX IF NOT EXISTS idx_stock_ticker ON stock_prices (ticker);
```

### DB 초기화 코드 (`src/db.py`)

```python
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

def get_engine():
    """SQLAlchemy 엔진 반환"""
    return create_engine(DB_URL, pool_pre_ping=True)

def get_conn():
    """DB 연결 반환 (with 문으로 사용 권장)"""
    return get_engine().connect()

def init_db():
    """테이블 초기 생성 — 최초 1회 실행"""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                date DATE NOT NULL, ticker VARCHAR(10) NOT NULL,
                open NUMERIC(12,4), high NUMERIC(12,4),
                low NUMERIC(12,4), close NUMERIC(12,4),
                volume BIGINT, return_pct NUMERIC(8,4),
                PRIMARY KEY (date, ticker)
            );
            CREATE TABLE IF NOT EXISTS macro_features (
                date DATE PRIMARY KEY,
                nasdaq NUMERIC(12,4), sox NUMERIC(12,4), vix NUMERIC(8,4),
                us_rate NUMERIC(6,4), usd_krw NUMERIC(10,4),
                nasdaq_lag1 NUMERIC(12,4), nasdaq_lag3 NUMERIC(12,4),
                nasdaq_lag5 NUMERIC(12,4), vix_lag1 NUMERIC(8,4),
                vix_lag3 NUMERIC(8,4), vix_lag5 NUMERIC(8,4),
                us_rate_lag1 NUMERIC(6,4), usd_krw_lag1 NUMERIC(10,4)
            );
            CREATE TABLE IF NOT EXISTS micro_features (
                date DATE NOT NULL, ticker VARCHAR(10) NOT NULL,
                revenue NUMERIC(20,2), operating_profit NUMERIC(20,2),
                eps NUMERIC(12,4),
                PRIMARY KEY (date, ticker)
            );
            CREATE TABLE IF NOT EXISTS alpha_features (
                date DATE NOT NULL, ticker VARCHAR(10) NOT NULL,
                foreign_net_buy NUMERIC(20,2), institution_net_buy NUMERIC(20,2),
                put_call_ratio NUMERIC(8,4), short_interest NUMERIC(8,4),
                PRIMARY KEY (date, ticker)
            );
            CREATE TABLE IF NOT EXISTS shap_results (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL, ticker VARCHAR(10) NOT NULL,
                feature_name VARCHAR(50) NOT NULL,
                shap_value NUMERIC(12,6), feature_value NUMERIC(12,6)
            );
            CREATE INDEX IF NOT EXISTS idx_shap_ticker_date
                ON shap_results (ticker, date);
            CREATE INDEX IF NOT EXISTS idx_stock_ticker
                ON stock_prices (ticker);
        """))
        conn.commit()
    print("PostgreSQL DB 초기화 완료")

if __name__ == "__main__":
    init_db()
```

---

## ⚙️ 전체 파이프라인 단계

```
[STEP 0] 환경 설정
    ↓ .env 작성 / PostgreSQL DB 생성 / init_db() 실행
[STEP 1] 데이터 수집
    ↓ FDR / yfinance / PyKRX / OpenDartReader
[STEP 2] DB 저장
    ↓ PostgreSQL INSERT ON CONFLICT DO NOTHING (중복 방지)
[STEP 3] 데이터 전처리
    ↓ 시계열 동기화 / Lag Feature / 결측치 처리 / 불균형 처리
[STEP 4] XGBoost 모델링
    ↓ Train(70%) / Validation(15%) / Test(15%) 분할
[STEP 5] SHAP 분석
    ↓ 글로벌 해석 + 로컬 케이스 스터디
[STEP 6] 시각화 & 결과 저장
    ↓ outputs/figures/ + shap_results 테이블
```

---

## STEP 0. 환경 설정

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. .env 파일 작성 (.env.example 복사 후 값 입력)
cp .env.example .env

# 3. PostgreSQL DB 생성 (최초 1회)
psql -U postgres -c "CREATE DATABASE semiconductor;"

# 4. 테이블 초기화 (최초 1회)
python src/db.py
```

---

## STEP 1. 데이터 수집 (`src/collect.py`)

### 수집 대상 및 도구

| 분류 | 세부 항목 | 수집 도구 | 비고 |
|------|-----------|-----------|------|
| 주가 | 삼성전자, SK하이닉스, NVDA | FDR, yfinance | 일별 종가·등락률 |
| 거시 | 나스닥, SOX, VIX, 기준금리, 환율 | FDR | 일별 |
| 미시 | 매출, 영업이익, EPS | OpenDartReader(국내) / yfinance(NVDA) | 분기→일별 forward-fill |
| 알파(국내) | 외인·기관 순매수 | PyKRX | 삼성전자·SK하이닉스만 |
| 알파(NVDA) | Put/Call Ratio, Short Interest | yfinance | 엔비디아만 |

### 핵심 코드 구조

```python
import FinanceDataReader as fdr
import yfinance as yf
from pykrx import stock
import OpenDartReader
import os
from dotenv import load_dotenv

load_dotenv()

START = "2018-01-01"
END   = "2024-12-31"

KR_TICKERS = {"삼성전자": "005930", "SK하이닉스": "000660"}
US_TICKERS = {"NVDA": "NVDA"}

def collect_stock_prices():
    """주가 수집 및 등락률 계산"""
    frames = []
    for name, ticker in KR_TICKERS.items():
        df = fdr.DataReader(ticker, START, END)[["Open","High","Low","Close","Volume"]]
        df["ticker"] = ticker
        df["return_pct"] = df["Close"].pct_change() * 100
        frames.append(df)
    df_nvda = yf.download("NVDA", start=START, end=END)[["Open","High","Low","Close","Volume"]]
    df_nvda["ticker"] = "NVDA"
    df_nvda["return_pct"] = df_nvda["Close"].pct_change() * 100
    frames.append(df_nvda)
    return frames

def collect_macro():
    """거시경제 지표 수집"""
    nasdaq  = fdr.DataReader("IXIC", START, END)["Close"].rename("nasdaq")
    sox     = fdr.DataReader("SOXX", START, END)["Close"].rename("sox")
    vix     = fdr.DataReader("VIX",  START, END)["Close"].rename("vix")
    usd_krw = fdr.DataReader("USD/KRW", START, END)["Close"].rename("usd_krw")
    us_rate = fdr.DataReader("FRED:FEDFUNDS", START, END)["FEDFUNDS"].rename("us_rate")
    return nasdaq, sox, vix, usd_krw, us_rate

def collect_alpha_kr(ticker: str):
    """외인·기관 순매수 (국내 종목 전용)"""
    df = stock.get_market_trading_value_by_date(
        START.replace("-",""), END.replace("-",""), ticker
    )
    return df[["외국인합계","기관합계"]].rename(
        columns={"외국인합계": "foreign_net_buy", "기관합계": "institution_net_buy"}
    )
```

---

## STEP 2. DB 저장

PostgreSQL은 `INSERT ON CONFLICT DO NOTHING` 으로 중복을 방지합니다.
SQLite의 `INSERT OR IGNORE` 와 동일한 역할이며, PostgreSQL 전용 문법입니다.

```python
import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import Table, MetaData
from src.db import get_engine

def save_to_db(df: pd.DataFrame, table_name: str, index_col: list):
    """
    중복 무시하고 PostgreSQL에 저장
    index_col: PRIMARY KEY 컬럼 리스트 (예: ["date", "ticker"])
    """
    engine  = get_engine()
    meta    = MetaData()
    meta.reflect(bind=engine)
    table   = meta.tables[table_name]
    records = df.reset_index().to_dict(orient="records")

    with engine.connect() as conn:
        stmt = pg_insert(table).values(records)
        stmt = stmt.on_conflict_do_nothing(index_elements=index_col)
        conn.execute(stmt)
        conn.commit()
    print(f"[저장 완료] {table_name}: {len(records)}행 처리")


# 사용 예시
# save_to_db(stock_df,  "stock_prices",   ["date", "ticker"])
# save_to_db(macro_df,  "macro_features",  ["date"])
# save_to_db(micro_df,  "micro_features",  ["date", "ticker"])
# save_to_db(alpha_df,  "alpha_features",  ["date", "ticker"])
```

---

## STEP 3. 데이터 전처리 (`src/preprocess.py`)

### ① 시계열 동기화 (종목별 분리 적용)

```python
def sync_timeseries(macro_df, target_ticker):
    """
    국내 종목: 미국 거시 지표를 +1 거래일 시프트
    NVDA: 시프트 없이 당일 데이터 사용
    """
    if target_ticker in ["005930", "000660"]:
        macro_df = macro_df.shift(1)
    return macro_df
```

### ② Look-ahead Bias 방지

```python
# 반드시 t-1 피처만 사용 (shift(1) 적용)
feature_df = feature_df.shift(1)

# 단위 테스트: 오늘 날짜의 피처가 어제 값인지 확인
assert feature_df.loc["2022-01-05", "vix"] == raw_vix.loc["2022-01-04", "vix"]
```

### ③ Lag Feature 생성

```python
def make_lag_features(df, cols, lags=[1, 3, 5]):
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

LAG_COLS = ["nasdaq", "vix", "us_rate", "usd_krw", "sox"]
macro_df = make_lag_features(macro_df, LAG_COLS)
```

### ④ 중복 지표 관리 (다중공선성 제거)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def remove_high_vif(df, threshold=10):
    while True:
        vif = pd.DataFrame({
            "feature": df.columns,
            "VIF": [variance_inflation_factor(df.values, i)
                    for i in range(df.shape[1])]
        })
        max_vif = vif["VIF"].max()
        if max_vif < threshold:
            break
        drop_col = vif.loc[vif["VIF"].idxmax(), "feature"]
        print(f"제거: {drop_col} (VIF={max_vif:.1f})")
        df = df.drop(columns=[drop_col])
    return df
```

### ⑤ 클래스 불균형 처리

```python
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 방법 1: SMOTE 오버샘플링
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 방법 2: XGBoost scale_pos_weight
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_weight = neg / pos

model = XGBClassifier(scale_pos_weight=scale_weight, ...)

# 두 방법 성능 비교 후 선택 (Validation F1-score 기준)
```

---

## STEP 4. XGBoost 모델링 (`src/model.py`)

### 시간 기반 분할

```python
def time_split(df, train_ratio=0.70, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return (
        df.iloc[:train_end],
        df.iloc[train_end:val_end],
        df.iloc[val_end:]
    )
```

### 하이퍼파라미터 튜닝 (Optuna)

```python
import optuna
from sklearn.metrics import f1_score

def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": scale_weight,
        "random_state": 42,
        "eval_metric": "logloss"
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=30,
              verbose=False)
    return f1_score(y_val, model.predict(X_val))

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

### Baseline vs Proposed 비교

```python
BASELINE_FEATURES = ["ma5", "ma10", "ma20", "rsi", "bb_upper", "bb_lower"]
PROPOSED_FEATURES = ["nasdaq", "vix", "us_rate", "usd_krw", "sox",
                     "nasdaq_lag1", "vix_lag1", "us_rate_lag1", "usd_krw_lag1",
                     "revenue", "eps", "foreign_net_buy"]
```

---

## STEP 5. SHAP 분석 (`src/shap_analysis.py`)

### 글로벌 해석

```python
import shap, matplotlib.pyplot as plt

def global_shap(model, X_test, save_path="outputs/figures/"):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(f"{save_path}global_bar.png", bbox_inches="tight", dpi=150)
    plt.close()

    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(f"{save_path}global_beeswarm.png", bbox_inches="tight", dpi=150)
    plt.close()

    return shap_values
```

### 로컬 해석 (케이스 스터디)

```python
def local_shap(model, X_test, date_index, label, save_path="outputs/figures/"):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    idx = X_test.index.get_loc(date_index)

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[idx],
            feature_names=X_test.columns.tolist()
        ), show=False
    )
    plt.savefig(f"{save_path}local_{label}.png", bbox_inches="tight", dpi=150)
    plt.close()

CASE_STUDIES = [
    {"date": "2022-06-15", "label": "rate_hike_2022", "desc": "2022년 금리 급등기"},
    {"date": "2024-03-08", "label": "ai_rally_2024",  "desc": "2024년 AI 반도체 랠리"},
]
```

### 해석 신뢰성 검증

```python
def sanity_check(shap_values, X_test, feature_col, expected_direction):
    corr = pd.Series(shap_values[:, X_test.columns.get_loc(feature_col)]).corr(
        X_test[feature_col]
    )
    actual_direction = "positive" if corr > 0 else "negative"
    match = (actual_direction == expected_direction)
    print(f"[{'✅' if match else '⚠️ 경고'}] {feature_col}: 예상={expected_direction}, 실제={actual_direction}")
    return match

sanity_check(shap_values, X_test, "us_rate", expected_direction="negative")
sanity_check(shap_values, X_test, "vix",     expected_direction="negative")
```

---

## STEP 6. 시각화 & 결과 저장

```python
def save_shap_to_db(shap_values, X_test, ticker):
    """SHAP 결과를 PostgreSQL에 저장"""
    engine = get_engine()
    rows = []
    for i, date in enumerate(X_test.index):
        for j, feature in enumerate(X_test.columns):
            rows.append({
                "date": str(date),
                "ticker": ticker,
                "feature_name": feature,
                "shap_value": float(shap_values[i, j]),
                "feature_value": float(X_test.iloc[i, j])
            })
    pd.DataFrame(rows).to_sql(
        "shap_results", engine,
        if_exists="append", index=False,
        method="multi", chunksize=1000
    )
    print(f"[SHAP 저장 완료] {ticker}: {len(rows)}행")
```

---

## 📦 requirements.txt

```
# 데이터 수집
FinanceDataReader>=0.9.50
yfinance>=0.2.40
pykrx>=1.0.45
OpenDartReader>=0.7.1

# 데이터 처리
pandas>=2.0.0
numpy>=1.24.0

# 모델링
scikit-learn>=1.4.0
xgboost>=2.0.0
imbalanced-learn>=0.12.0

# XAI
shap>=0.45.0

# 하이퍼파라미터 튜닝
optuna>=3.6.0

# 시각화
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.20.0

# 통계 검정
statsmodels>=0.14.0

# PostgreSQL 연결
psycopg2-binary>=2.9.9
sqlalchemy>=2.0.0
python-dotenv>=1.0.0
```

---

## ⚠️ 주의사항 & 체크리스트

### 환경 설정 전

- [ ] `.env` 파일 작성 완료 (DB 연결 정보, DART API 키)
- [ ] `.gitignore`에 `.env` 추가 확인 (비밀번호 노출 방지)
- [ ] PostgreSQL 서버 실행 중인지 확인
- [ ] `python src/db.py` 로 테이블 초기화 완료

### 데이터 수집 전

- [ ] PyKRX는 엔비디아에 사용 불가 → NVDA 알파 피처는 Put/Call Ratio로 대체
- [ ] OpenDartReader API 키 발급 필요 (dart.fss.or.kr)
- [ ] 수집 기간: 2018-01-01 ~ 2024-12-31 통일

### 전처리 전

- [ ] 국내 종목과 NVDA의 시계열 동기화 방식이 다름 (shift 여부 확인)
- [ ] shift(1) 적용 후 단위 테스트 반드시 실행
- [ ] VIF 분석으로 다중공선성 제거 후 피처 확정

### 모델링 전

- [ ] Train/Val/Test 분할 시 랜덤 셔플 금지 (시간 순서 유지)
- [ ] Test set은 최종 평가 시 단 1회만 사용
- [ ] SMOTE vs scale_pos_weight 성능 비교 후 선택

### SHAP 분석 전

- [ ] 글로벌 해석 → 로컬 해석 순서로 진행
- [ ] sanity_check()로 경제 상식과 방향 일치 확인
- [ ] 상식에 반하는 결과 → 데이터 누수(Leakage) 의심 및 역추적

---

## 📊 최종 산출물 목록

| 산출물 | 저장 위치 | 설명 |
|--------|-----------|------|
| 글로벌 SHAP Bar Plot | outputs/figures/global_bar.png | 전체 피처 평균 기여도 순위 |
| 글로벌 SHAP Beeswarm | outputs/figures/global_beeswarm.png | 방향 + 크기 동시 시각화 |
| 로컬 SHAP Waterfall | outputs/figures/local_*.png | 특정 시점 케이스 스터디 |
| Baseline vs Proposed | outputs/results/model_comparison.csv | F1, Recall, AUC 비교표 |
| SHAP 수치 결과 | PostgreSQL > shap_results 테이블 | 날짜·종목·피처별 기여도 |
