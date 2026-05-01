import FinanceDataReader as fdr
import yfinance as yf
from pykrx import stock
import OpenDartReader
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

START = "2018-01-01"
END   = "2024-12-31"

KR_TICKERS = {"삼성전자": "005930", "SK하이닉스": "000660"}
US_TICKERS = {"NVDA": "NVDA"}


def collect_stock_prices() -> list[pd.DataFrame]:
    frames = []
    for name, ticker in KR_TICKERS.items():
        df = fdr.DataReader(ticker, START, END)[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = df.columns.str.lower()
        df["ticker"] = ticker
        df["return_pct"] = df["close"].pct_change() * 100
        frames.append(df)

    df_nvda = yf.download("NVDA", start=START, end=END)[["Open", "High", "Low", "Close", "Volume"]]
    df_nvda.columns = df_nvda.columns.str.lower()
    df_nvda["ticker"] = "NVDA"
    df_nvda["return_pct"] = df_nvda["close"].pct_change() * 100
    frames.append(df_nvda)
    return frames


def collect_macro() -> pd.DataFrame:
    nasdaq  = fdr.DataReader("IXIC",        START, END)["Close"].rename("nasdaq")
    sox     = fdr.DataReader("SOXX",        START, END)["Close"].rename("sox")
    vix     = fdr.DataReader("VIX",         START, END)["Close"].rename("vix")
    usd_krw = fdr.DataReader("USD/KRW",     START, END)["Close"].rename("usd_krw")
    us_rate = fdr.DataReader("FRED:FEDFUNDS", START, END)["FEDFUNDS"].rename("us_rate")
    return pd.concat([nasdaq, sox, vix, usd_krw, us_rate], axis=1)


def collect_alpha_kr(ticker: str) -> pd.DataFrame:
    df = stock.get_market_trading_value_by_date(
        START.replace("-", ""), END.replace("-", ""), ticker
    )
    return df[["외국인합계", "기관합계"]].rename(
        columns={"외국인합계": "foreign_net_buy", "기관합계": "institution_net_buy"}
    )


def collect_micro_kr(ticker: str) -> pd.DataFrame:
    dart = OpenDartReader.OpenDartReader(os.getenv("DART_API_KEY"))
    # TODO: OpenDartReader로 분기별 실적 수집 후 forward-fill 적용
    raise NotImplementedError


def collect_micro_nvda() -> pd.DataFrame:
    nvda = yf.Ticker("NVDA")
    # TODO: yfinance quarterly financials → forward-fill
    raise NotImplementedError
