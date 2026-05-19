import os
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import MetaData
from sqlalchemy.dialects.postgresql import insert as pg_insert

try:
    from src.db import get_engine
except ModuleNotFoundError:
    from db import get_engine

load_dotenv()

START = "2018-01-01"
END   = "2024-12-31"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

KR_TICKERS = {"삼성전자": "005930", "SK하이닉스": "000660"}
US_TICKERS = {"NVDA": "NVDA"}
SAMSUNG_TICKER = "005930"
SK_HYNIX_TICKER = "000660"
ALPHA_REQUIRED_COLUMNS = ["외국인합계", "기관합계"]


@dataclass(frozen=True)
class SeriesSpec:
    feature_group: str
    feature_name: str
    symbol: str
    source: str
    frequency: str
    save_dir: str
    description: str


EXTENDED_FDR_SPECS = [
    SeriesSpec("stock_price", "samsung_electronics", "005930", "FDR", "daily", "prices", "Samsung Electronics stock price"),
    SeriesSpec("stock_price", "sk_hynix", "000660", "FDR", "daily", "prices", "SK Hynix stock price"),
    SeriesSpec("stock_price", "nvidia", "NVDA", "FDR", "daily", "prices", "NVIDIA stock price"),
    SeriesSpec("semi_index", "vanEck_semiconductor_etf_smh", "SMH", "FDR", "daily", "semi_indices", "Semiconductor ETF proxy for SOX/global semi cycle"),
    SeriesSpec("semi_index", "ishares_semiconductor_etf_soxx", "SOXX", "FDR", "daily", "semi_indices", "Semiconductor ETF proxy for SOX"),
    SeriesSpec("semi_index", "phlx_semiconductor_index_sox", "SOX", "FDR", "daily", "semi_indices", "PHLX Semiconductor Index direct symbol trial"),
    SeriesSpec("semi_index", "phlx_semiconductor_index_yahoo", "^SOX", "FDR", "daily", "semi_indices", "PHLX Semiconductor Index Yahoo style symbol trial"),
    SeriesSpec("global_index", "kospi", "KS11", "FDR", "daily", "global_indices", "Korea market common factor"),
    SeriesSpec("global_index", "kospi200", "KS200", "FDR", "daily", "global_indices", "Korea large-cap market factor"),
    SeriesSpec("global_index", "kosdaq", "KQ11", "FDR", "daily", "global_indices", "Korea growth/tech sentiment"),
    SeriesSpec("global_index", "nasdaq_composite", "IXIC", "FDR", "daily", "global_indices", "US technology beta"),
    SeriesSpec("global_index", "sp500", "S&P500", "FDR", "daily", "global_indices", "US risk asset common factor"),
    SeriesSpec("global_index", "dow_jones", "DJI", "FDR", "daily", "global_indices", "US cyclical large-cap sentiment"),
    SeriesSpec("global_index", "russell2000", "RUT", "FDR", "daily", "global_indices", "US small-cap risk appetite"),
    SeriesSpec("global_index", "vix", "VIX", "FDR", "daily", "global_indices", "Risk aversion and volatility"),
    SeriesSpec("global_index", "shanghai_composite", "SSEC", "FDR", "daily", "global_indices", "China demand and Asia manufacturing proxy"),
    SeriesSpec("global_index", "hang_seng", "HSI", "FDR", "daily", "global_indices", "China/Hong Kong risk sentiment"),
    SeriesSpec("global_index", "nikkei225", "N225", "FDR", "daily", "global_indices", "Japan equipment/material supply-chain proxy"),
    SeriesSpec("fx", "usd_krw", "USD/KRW", "FDR", "daily", "fx_rates", "KRW FX factor"),
    SeriesSpec("fx", "usd_cny", "USD/CNY", "FDR", "daily", "fx_rates", "China FX and demand stress proxy"),
    SeriesSpec("fx", "usd_jpy", "USD/JPY", "FDR", "daily", "fx_rates", "Japan FX and carry/risk proxy"),
    SeriesSpec("fx", "cny_krw", "CNY/KRW", "FDR", "daily", "fx_rates", "Korea-China FX relationship"),
    SeriesSpec("rate_dollar", "us_5y_treasury", "US5YT", "FDR", "daily", "rates", "US 5-year treasury yield"),
    SeriesSpec("rate_dollar", "us_10y_treasury", "US10YT", "FDR", "daily", "rates", "US 10-year treasury yield"),
    SeriesSpec("rate_dollar", "us_30y_treasury", "US30YT", "FDR", "daily", "rates", "US 30-year treasury yield"),
    SeriesSpec("rate_dollar", "dollar_index", "^NYICDX", "FDR", "daily", "rates", "US dollar index"),
    SeriesSpec("future", "wti_crude_oil", "CL=F", "FDR", "daily", "futures", "WTI crude oil futures"),
    SeriesSpec("future", "brent_crude_oil", "BZ=F", "FDR", "daily", "futures", "Brent crude oil futures"),
    SeriesSpec("future", "natural_gas", "NG=F", "FDR", "daily", "futures", "Natural gas futures"),
    SeriesSpec("future", "gold", "GC=F", "FDR", "daily", "futures", "Gold futures"),
    SeriesSpec("future", "silver", "SI=F", "FDR", "daily", "futures", "Silver futures"),
    SeriesSpec("future", "copper", "HG=F", "FDR", "daily", "futures", "Copper futures"),
]

FRED_SYMBOLS = {
    "FEDFUNDS": "federal_funds_rate",
    "DGS10": "us_10y_treasury_fred",
    "DGS2": "us_2y_treasury_fred",
    "T10Y2Y": "us_10y_2y_spread",
    "VIXCLS": "vix_fred",
    "NASDAQCOM": "nasdaq_composite_fred",
    "DTWEXBGS": "broad_dollar_index",
    "INDPRO": "industrial_production",
    "IPG3344S": "semiconductor_electronic_component_industrial_production",
    "PCU334413334413P": "semiconductor_related_device_ppi",
    "A34SNO": "computers_electronic_products_new_orders",
}


# DataFrame 인덱스를 date 컬럼 기준 DatetimeIndex로 정리한다.
def _with_date_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


# 인덱스가 날짜인 DataFrame을 DB insert용 record 목록으로 변환한다.
def _records_from_indexed_df(df: pd.DataFrame) -> list[dict]:
    records_df = df.reset_index()
    if "date" not in records_df.columns:
        records_df = records_df.rename(columns={records_df.columns[0]: "date"})
    records_df["date"] = pd.to_datetime(records_df["date"]).dt.date
    records_df = records_df.where(pd.notnull(records_df), None)
    return records_df.to_dict(orient="records")


def _ensure_data_dirs() -> None:
    folders = [
        DATA_ROOT / "raw" / "fdr" / folder
        for folder in [
            "prices",
            "semi_indices",
            "global_indices",
            "fx_rates",
            "rates",
            "futures",
            "fred",
        ]
    ]
    folders += [DATA_ROOT / "metadata"]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def _save_raw_dataframe(df: pd.DataFrame, save_dir: str, feature_name: str) -> Path:
    path = DATA_ROOT / "raw" / "fdr" / save_dir / f"{feature_name}.csv"
    df.to_csv(path, index=True, encoding="utf-8-sig")
    return path


def _series_for_market_feature(df: pd.DataFrame, feature_name: str) -> pd.Series:
    if "Close" in df.columns:
        series = df["Close"]
    else:
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty:
            raise ValueError("numeric column not found")
        series = numeric_df.iloc[:, 0]
    series = pd.to_numeric(series, errors="coerce").rename(feature_name).dropna()
    series.index = pd.to_datetime(series.index)
    return series


def _records_from_market_series(
    series: pd.Series,
    feature_group: str,
    source: str,
) -> list[dict]:
    return [
        {
            "date": date.date(),
            "feature_name": series.name,
            "feature_group": feature_group,
            "source": source,
            "value": float(value),
        }
        for date, value in series.items()
        if pd.notna(value)
    ]


# PostgreSQL 테이블에 DataFrame을 저장하고 primary key 충돌 시 무시 또는 갱신한다.
def save_to_db(
    df: pd.DataFrame,
    table_name: str,
    index_col: list[str],
    update_on_conflict: bool = False,
) -> int:
    engine = get_engine()
    records = _records_from_indexed_df(df)
    if not records:
        print(f"[저장 건너뜀] {table_name}: 저장할 데이터가 없습니다")
        return 0

    meta = MetaData()
    meta.reflect(bind=engine, only=[table_name])
    table = meta.tables[table_name]
    table_columns = set(table.columns.keys())
    records = [
        {key: value for key, value in record.items() if key in table_columns}
        for record in records
    ]

    with engine.connect() as conn:
        stmt = pg_insert(table).values(records)
        if update_on_conflict:
            update_columns = {
                col: getattr(stmt.excluded, col)
                for col in table_columns
                if col not in index_col
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=index_col,
                set_=update_columns,
            )
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=index_col)
        result = conn.execute(stmt)
        conn.commit()

    inserted = result.rowcount or 0
    print(f"[저장 완료] {table_name}: {inserted}/{len(records)}행 신규 저장")
    return inserted


def save_market_series_to_db(
    series: pd.Series,
    feature_group: str,
    source: str,
) -> int:
    records = _records_from_market_series(series, feature_group, source)
    if not records:
        print(f"[저장 건너뜀] market_features: {series.name} 저장할 데이터가 없습니다")
        return 0

    engine = get_engine()
    meta = MetaData()
    meta.reflect(bind=engine, only=["market_features"])
    table = meta.tables["market_features"]

    with engine.connect() as conn:
        stmt = pg_insert(table).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["date", "feature_name"],
            set_={
                "feature_group": stmt.excluded.feature_group,
                "source": stmt.excluded.source,
                "value": stmt.excluded.value,
            },
        )
        result = conn.execute(stmt)
        conn.commit()

    saved = result.rowcount or 0
    print(f"[저장 완료] market_features.{series.name}: {saved}/{len(records)}행 처리")
    return saved


def _record_collection_status(
    records: list[dict],
    spec: SeriesSpec,
    status: str,
    rows: int = 0,
    file_path: Path | None = None,
    error: Exception | None = None,
) -> None:
    records.append({
        "feature_group": spec.feature_group,
        "feature_name": spec.feature_name,
        "symbol": spec.symbol,
        "source": spec.source,
        "frequency": spec.frequency,
        "description": spec.description,
        "rows": rows,
        "status": status,
        "file_path": "" if file_path is None else str(file_path.relative_to(PROJECT_ROOT)),
        "error_type": "" if error is None else type(error).__name__,
        "error_message": "" if error is None else str(error),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    })


def _write_extended_collection_summary(records: list[dict]) -> None:
    if not records:
        return
    metadata_dir = DATA_ROOT / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / "extended_collection_summary.csv"
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[메타데이터 저장 완료] {path}")


def collect_extended_fdr_market_features() -> list[dict]:
    _ensure_data_dirs()
    records: list[dict] = []
    for spec in EXTENDED_FDR_SPECS:
        try:
            df = fdr.DataReader(spec.symbol, START, END)
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            df.index.name = "date"
            raw_path = _save_raw_dataframe(df, spec.save_dir, spec.feature_name)
            series = _series_for_market_feature(df, spec.feature_name)
            save_market_series_to_db(series, spec.feature_group, spec.source)
            _record_collection_status(records, spec, "success", len(series), raw_path)
            print(f"[OK] 확장 FDR: {spec.feature_name}")
        except Exception as exc:
            _record_collection_status(records, spec, "failed", error=exc)
            print(f"[FAIL] 확장 FDR: {spec.feature_name} - {exc}")
    return records


def collect_fred_market_features() -> list[dict]:
    _ensure_data_dirs()
    records: list[dict] = []
    spec_symbol = "FRED:" + ",".join(FRED_SYMBOLS.keys())
    try:
        df = fdr.DataReader(spec_symbol, START, END)
        if df is None or df.empty:
            raise ValueError("empty dataframe")
        df = df.rename(columns=FRED_SYMBOLS)
        df.index.name = "date"
        raw_path = _save_raw_dataframe(df, "fred", "fred_macro_semiconductor_panel")

        for feature_name in FRED_SYMBOLS.values():
            spec = SeriesSpec(
                "fred",
                feature_name,
                feature_name,
                "FDR/FRED",
                "mixed",
                "fred",
                "FRED macro/semiconductor indicator",
            )
            series = pd.to_numeric(df[feature_name], errors="coerce").rename(feature_name).dropna()
            series.index = pd.to_datetime(series.index)
            save_market_series_to_db(series, "fred", "FDR/FRED")
            _record_collection_status(records, spec, "success", len(series), raw_path)
        print("[OK] 확장 FRED 패널")
    except Exception as exc:
        for code, feature_name in FRED_SYMBOLS.items():
            spec = SeriesSpec("fred", feature_name, f"FRED:{code}", "FDR/FRED", "mixed", "fred", "FRED macro/semiconductor indicator")
            _record_collection_status(records, spec, "failed", error=exc)
        print(f"[FAIL] 확장 FRED 패널 - {exc}")
    return records


def collect_extended_market_to_db() -> None:
    records = collect_extended_fdr_market_features()
    records.extend(collect_fred_market_features())
    _write_extended_collection_summary(records)
    success_count = sum(record["status"] == "success" for record in records)
    failed_count = sum(record["status"] == "failed" for record in records)
    print(f"[확장 수집 완료] success={success_count}, failed={failed_count}")


# 프로젝트 대상 전체 종목의 주가 데이터를 수집한다.
def collect_stock_prices() -> list[pd.DataFrame]:
    frames = []
    for name, ticker in KR_TICKERS.items():
        df = fdr.DataReader(ticker, START, END)[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = df.columns.str.lower()
        df["ticker"] = ticker
        df["return_pct"] = df["close"].pct_change() * 100
        frames.append(df)

    import yfinance as yf

    df_nvda = yf.download("NVDA", start=START, end=END)[["Open", "High", "Low", "Close", "Volume"]]
    df_nvda.columns = df_nvda.columns.str.lower()
    df_nvda["ticker"] = "NVDA"
    df_nvda["return_pct"] = df_nvda["close"].pct_change() * 100
    frames.append(df_nvda)
    return frames


# 단일 티커의 일별 OHLCV와 수익률 데이터를 수집한다.
def collect_stock_price(ticker: str) -> pd.DataFrame:
    df = fdr.DataReader(ticker, START, END)[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = df.columns.str.lower()
    df["ticker"] = ticker
    df["return_pct"] = df["close"].pct_change() * 100
    return _with_date_index(df)


# 삼성전자 주가 데이터를 수집한다.
def collect_samsung_stock_price() -> pd.DataFrame:
    return collect_stock_price(SAMSUNG_TICKER)


# 거시 지표와 국내 시장 지표를 수집해 하나의 DataFrame으로 결합한다.
def collect_macro() -> pd.DataFrame:
    nasdaq  = fdr.DataReader("IXIC",        START, END)["Close"].rename("nasdaq")
    sox     = fdr.DataReader("SOXX",        START, END)["Close"].rename("sox")
    vix     = fdr.DataReader("VIX",         START, END)["Close"].rename("vix")
    usd_krw = fdr.DataReader("USD/KRW",     START, END)["Close"].rename("usd_krw")
    us_rate = fdr.DataReader("FRED:FEDFUNDS", START, END)["FEDFUNDS"].rename("us_rate")
    kospi   = fdr.DataReader("KS11",        START, END)["Close"].rename("kospi")
    kospi200 = fdr.DataReader("KS200",      START, END)["Close"].rename("kospi200")
    return _with_date_index(pd.concat([nasdaq, sox, vix, usd_krw, us_rate, kospi, kospi200], axis=1))


# 국내 종목의 외국인/기관 순매수 데이터를 PyKRX로 수집한다.
def collect_alpha_kr(ticker: str) -> pd.DataFrame:
    if not os.getenv("KRX_ID") or not os.getenv("KRX_PW"):
        raise RuntimeError(
            "KRX 수급 데이터 수집에는 KRX_ID, KRX_PW 환경변수가 필요합니다. "
            ".env에 KRX_ID와 KRX_PW를 추가하거나 --skip-alpha 옵션으로 수급 수집을 건너뛰세요."
        )

    from pykrx import stock

    df = stock.get_market_trading_value_by_date(
        START.replace("-", ""), END.replace("-", ""), ticker
    )

    missing_columns = [col for col in ALPHA_REQUIRED_COLUMNS if col not in df.columns]
    if df.empty or missing_columns:
        raise RuntimeError(
            "PyKRX 수급 데이터 응답이 비어 있거나 예상 컬럼이 없습니다. "
            f"ticker={ticker}, rows={len(df)}, columns={list(df.columns)}"
        )

    df = df[ALPHA_REQUIRED_COLUMNS].rename(
        columns={"외국인합계": "foreign_net_buy", "기관합계": "institution_net_buy"}
    )
    df["ticker"] = ticker
    return _with_date_index(df)


# 삼성전자 외국인/기관 수급 데이터를 수집한다.
def collect_samsung_alpha() -> pd.DataFrame:
    return collect_alpha_kr(SAMSUNG_TICKER)


# 지정한 국내 티커의 주가, 거시 지표, 수급 데이터를 DB에 저장한다.
def collect_ticker_to_db(ticker: str, include_alpha: bool = True) -> None:
    print(f"[수집 시작] 주가: {ticker}")
    stock_df = collect_stock_price(ticker)
    save_to_db(stock_df, "stock_prices", ["date", "ticker"])

    print("[수집 시작] 거시 지표")
    macro = collect_macro()
    save_to_db(macro, "macro_features", ["date"], update_on_conflict=True)

    if include_alpha:
        print(f"[수집 시작] 수급: {ticker}")
        alpha_df = collect_alpha_kr(ticker)
        save_to_db(alpha_df, "alpha_features", ["date", "ticker"])
    else:
        print(f"[수집 건너뜀] 수급: {ticker} (--skip-alpha)")

    print(f"[수집 완료] 1차 데이터: {ticker}")


# 삼성전자 전체 수집 파이프라인을 실행한다.
def collect_samsung_to_db(include_alpha: bool = True) -> None:
    collect_ticker_to_db(SAMSUNG_TICKER, include_alpha=include_alpha)


# SK하이닉스 전체 수집 파이프라인을 실행한다.
def collect_sk_hynix_to_db(include_alpha: bool = True) -> None:
    collect_ticker_to_db(SK_HYNIX_TICKER, include_alpha=include_alpha)


# 국내 종목 분기 실적 수집을 위한 자리이며 아직 구현되지 않았다.
def collect_micro_kr(ticker: str) -> pd.DataFrame:
    import OpenDartReader

    dart = OpenDartReader.OpenDartReader(os.getenv("DART_API_KEY"))
    # TODO: OpenDartReader로 분기별 실적 수집 후 forward-fill 적용
    raise NotImplementedError


# NVDA 미시/실적 피처 수집을 위한 자리이며 아직 구현되지 않았다.
def collect_micro_nvda() -> pd.DataFrame:
    import yfinance as yf

    nvda = yf.Ticker("NVDA")
    # TODO: yfinance quarterly financials → forward-fill
    raise NotImplementedError


# 커맨드라인 실행 옵션을 파싱한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="국내 반도체 종목 1차 데이터 수집")
    parser.add_argument(
        "--ticker",
        default=SAMSUNG_TICKER,
        choices=[SAMSUNG_TICKER, SK_HYNIX_TICKER],
        help="수집할 국내 종목 티커입니다. 삼성전자=005930, SK하이닉스=000660",
    )
    parser.add_argument(
        "--skip-alpha",
        action="store_true",
        help="KRX 외국인/기관 수급 데이터 수집을 건너뜁니다.",
    )
    parser.add_argument(
        "--extended-market",
        action="store_true",
        help="zip 수집 항목을 반영한 확장 시장/거시/FRED 데이터를 market_features 테이블과 raw CSV에 저장합니다.",
    )
    parser.add_argument(
        "--skip-core",
        action="store_true",
        help="기존 주가/거시/수급 수집은 건너뛰고 확장 시장 데이터만 수집합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_core:
        collect_ticker_to_db(args.ticker, include_alpha=not args.skip_alpha)
    if args.extended_market:
        collect_extended_market_to_db()
