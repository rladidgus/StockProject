import FinanceDataReader as fdr
import pandas as pd
import os
import argparse
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

KR_TICKERS = {"삼성전자": "005930", "SK하이닉스": "000660"}
US_TICKERS = {"NVDA": "NVDA"}
SAMSUNG_TICKER = "005930"
SK_HYNIX_TICKER = "000660"
ALPHA_REQUIRED_COLUMNS = ["외국인합계", "기관합계"]


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_ticker_to_db(args.ticker, include_alpha=not args.skip_alpha)
