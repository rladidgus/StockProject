import pandas as pd
import argparse
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

try:
    from src.db import get_engine
except ModuleNotFoundError:
    from db import get_engine


SAMSUNG_TICKER = "005930"
SK_HYNIX_TICKER = "000660"
DEFAULT_OUTPUT_PATHS = {
    SAMSUNG_TICKER: Path("data/processed/samsung_preprocessed.csv"),
    SK_HYNIX_TICKER: Path("data/processed/sk_hynix_preprocessed.csv"),
}
STOCK_COLUMNS = ["open", "high", "low", "close", "volume", "return_pct"]
ALPHA_COLUMNS = ["foreign_net_buy", "institution_net_buy"]
TARGET_COLUMNS = ["target_up", "next_return_pct"]
TECHNICAL_COLUMNS = ["ma5", "ma10", "ma20", "rsi", "bb_upper", "bb_lower"]
MACRO_CHANGE_COLUMNS = [
    "nasdaq_return_1d", "nasdaq_return_3d",
    "sox_return_1d", "sox_return_3d",
    "vix_change_1d", "vix_change_3d",
    "usd_krw_return_1d", "usd_krw_return_3d",
]
MARKET_FEATURE_COLUMNS = [
    "kospi_return_1d",
    "kospi_return_3d",
    "kospi200_return_1d",
    "kospi200_return_3d",
    "stock_vs_kospi_return_1d",
    "stock_vs_kospi200_return_1d",
]
ALPHA_ROLLING_COLUMNS = [
    "foreign_net_buy_5d_sum",
    "foreign_net_buy_20d_sum",
    "institution_net_buy_5d_sum",
    "institution_net_buy_20d_sum",
]
ALPHA_INTENSITY_COLUMNS = [
    "trading_value",
    "foreign_net_buy_to_volume",
    "institution_net_buy_to_volume",
    "foreign_net_buy_to_trading_value",
    "institution_net_buy_to_trading_value",
    "foreign_net_buy_5d_to_trading_value_5d",
    "foreign_net_buy_20d_to_trading_value_20d",
    "institution_net_buy_5d_to_trading_value_5d",
    "institution_net_buy_20d_to_trading_value_20d",
]
TARGET_HORIZON_DAYS = 5
US_MACRO_COLUMNS = ["nasdaq", "vix", "us_rate", "usd_krw", "sox"]
MARKET_INDEX_COLUMNS = ["kospi", "kospi200"]
LAG_COLS = US_MACRO_COLUMNS + MARKET_INDEX_COLUMNS


# 국내 종목 기준으로 미국 거시 지표만 하루 밀어 시계열을 동기화한다.
def sync_timeseries(macro_df: pd.DataFrame, target_ticker: str) -> pd.DataFrame:
    macro_df = macro_df.copy()
    if target_ticker in ["005930", "000660"]:
        shift_columns = [col for col in US_MACRO_COLUMNS if col in macro_df.columns]
        macro_df[shift_columns] = macro_df[shift_columns].shift(1)
    return macro_df


# 지정한 컬럼들에 대해 1일, 3일, 5일 등 lag feature를 생성한다.
def make_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int] = [1, 3, 5]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


# lag feature 컬럼명을 리스트로 생성한다.
def lag_feature_columns(cols: list[str], lags: list[int] = [1, 3, 5]) -> list[str]:
    return [f"{col}_lag{lag}" for col in cols for lag in lags]


# 종가 기반 이동평균, RSI, 볼린저 밴드 기술적 지표를 추가한다.
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, pd.NA)
    df["rsi"] = 100 - (100 / (1 + rs))

    rolling_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["ma20"] + (2 * rolling_std)
    df["bb_lower"] = df["ma20"] - (2 * rolling_std)
    return df


# 나스닥, SOX, VIX, 환율 같은 거시 지표의 변화율/변화량을 추가한다.
def add_macro_change_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nasdaq_return_1d"] = df["nasdaq"].pct_change(1) * 100
    df["nasdaq_return_3d"] = df["nasdaq"].pct_change(3) * 100
    df["sox_return_1d"] = df["sox"].pct_change(1) * 100
    df["sox_return_3d"] = df["sox"].pct_change(3) * 100
    df["vix_change_1d"] = df["vix"].diff(1)
    df["vix_change_3d"] = df["vix"].diff(3)
    df["usd_krw_return_1d"] = df["usd_krw"].pct_change(1) * 100
    df["usd_krw_return_3d"] = df["usd_krw"].pct_change(3) * 100
    return df


# KOSPI/KOSPI200 변화율과 종목의 시장 대비 상대수익률을 추가한다.
def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kospi_return_1d"] = df["kospi"].pct_change(1) * 100
    df["kospi_return_3d"] = df["kospi"].pct_change(3) * 100
    df["kospi200_return_1d"] = df["kospi200"].pct_change(1) * 100
    df["kospi200_return_3d"] = df["kospi200"].pct_change(3) * 100
    df["stock_vs_kospi_return_1d"] = df["return_pct"] - df["kospi_return_1d"]
    df["stock_vs_kospi200_return_1d"] = df["return_pct"] - df["kospi200_return_1d"]
    return df


# 외국인/기관 순매수의 5일, 20일 rolling 합계를 추가한다.
def add_alpha_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["foreign_net_buy_5d_sum"] = df["foreign_net_buy"].rolling(window=5).sum()
    df["foreign_net_buy_20d_sum"] = df["foreign_net_buy"].rolling(window=20).sum()
    df["institution_net_buy_5d_sum"] = df["institution_net_buy"].rolling(window=5).sum()
    df["institution_net_buy_20d_sum"] = df["institution_net_buy"].rolling(window=20).sum()
    return df


# 외국인/기관 순매수를 거래량과 거래대금 대비 강도 지표로 변환한다.
def add_alpha_intensity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trading_value"] = df["close"] * df["volume"]
    safe_volume = df["volume"].replace(0, pd.NA)
    safe_trading_value = df["trading_value"].replace(0, pd.NA)
    trading_value_5d = df["trading_value"].rolling(window=5).sum().replace(0, pd.NA)
    trading_value_20d = df["trading_value"].rolling(window=20).sum().replace(0, pd.NA)

    df["foreign_net_buy_to_volume"] = df["foreign_net_buy"] / safe_volume
    df["institution_net_buy_to_volume"] = df["institution_net_buy"] / safe_volume
    df["foreign_net_buy_to_trading_value"] = df["foreign_net_buy"] / safe_trading_value
    df["institution_net_buy_to_trading_value"] = df["institution_net_buy"] / safe_trading_value
    df["foreign_net_buy_5d_to_trading_value_5d"] = df["foreign_net_buy_5d_sum"] / trading_value_5d
    df["foreign_net_buy_20d_to_trading_value_20d"] = df["foreign_net_buy_20d_sum"] / trading_value_20d
    df["institution_net_buy_5d_to_trading_value_5d"] = df["institution_net_buy_5d_sum"] / trading_value_5d
    df["institution_net_buy_20d_to_trading_value_20d"] = df["institution_net_buy_20d_sum"] / trading_value_20d
    return df


# stock_prices 테이블에서 지정한 티커의 일별 주가 데이터를 불러온다.
def load_stock_prices(ticker: str = SAMSUNG_TICKER) -> pd.DataFrame:
    query = """
        SELECT date, open, high, low, close, volume, return_pct
        FROM stock_prices
        WHERE ticker = %(ticker)s
        ORDER BY date
    """
    df = pd.read_sql(query, get_engine(), params={"ticker": ticker}, parse_dates=["date"])
    return df.set_index("date")


# macro_features 테이블에서 거시 지표와 국내 시장 지표를 불러온다.
def load_macro_features() -> pd.DataFrame:
    query = """
        SELECT date, nasdaq, sox, vix, us_rate, usd_krw, kospi, kospi200
        FROM macro_features
        ORDER BY date
    """
    df = pd.read_sql(query, get_engine(), parse_dates=["date"])
    return df.set_index("date")


# alpha_features 테이블에서 지정한 티커의 외국인/기관 수급 데이터를 불러온다.
def load_alpha_features(ticker: str = SAMSUNG_TICKER) -> pd.DataFrame:
    query = """
        SELECT date, foreign_net_buy, institution_net_buy
        FROM alpha_features
        WHERE ticker = %(ticker)s
        ORDER BY date
    """
    df = pd.read_sql(query, get_engine(), params={"ticker": ticker}, parse_dates=["date"])
    return df.set_index("date")


# 주가, 거시, 시장, 수급 데이터를 결합해 모델 학습용 데이터셋을 만든다.
def build_ticker_dataset(ticker: str = SAMSUNG_TICKER) -> pd.DataFrame:
    stock_df = load_stock_prices(ticker)
    macro_df = load_macro_features()
    alpha_df = load_alpha_features(ticker)

    if stock_df.empty:
        raise ValueError(f"stock_prices에 {ticker} 데이터가 없습니다.")
    if macro_df.empty:
        raise ValueError("macro_features 데이터가 없습니다.")

    macro_df = sync_timeseries(macro_df, ticker)
    macro_df = make_lag_features(macro_df, LAG_COLS)
    macro_df = macro_df.reindex(stock_df.index).ffill()
    macro_df = add_macro_change_features(macro_df)

    feature_df = stock_df.join(macro_df, how="left")
    feature_df = add_market_features(feature_df)
    feature_df = add_technical_features(feature_df)
    if not alpha_df.empty:
        alpha_df = alpha_df.reindex(stock_df.index).ffill()
        feature_df = feature_df.join(alpha_df, how="left")
        feature_df[ALPHA_COLUMNS] = feature_df[ALPHA_COLUMNS].ffill()
        feature_df = add_alpha_rolling_features(feature_df)
        feature_df = add_alpha_intensity_features(feature_df)
    else:
        feature_df[ALPHA_COLUMNS] = pd.NA
        for col in ALPHA_ROLLING_COLUMNS:
            feature_df[col] = pd.NA
        for col in ALPHA_INTENSITY_COLUMNS:
            feature_df[col] = pd.NA

    future_close = feature_df["close"].shift(-TARGET_HORIZON_DAYS)
    feature_df["target_up"] = (future_close > feature_df["close"]).astype("Int64")
    feature_df["next_return_pct"] = (future_close / feature_df["close"] - 1) * 100
    required_columns = (
        STOCK_COLUMNS
        + TECHNICAL_COLUMNS
        + LAG_COLS
        + lag_feature_columns(LAG_COLS)
        + MACRO_CHANGE_COLUMNS
        + MARKET_FEATURE_COLUMNS
        + ALPHA_ROLLING_COLUMNS
        + ALPHA_INTENSITY_COLUMNS
        + TARGET_COLUMNS
    )
    feature_df = feature_df.dropna(subset=required_columns)
    feature_df["target_up"] = feature_df["target_up"].astype(int)
    return feature_df


# 삼성전자 모델 학습용 데이터셋을 만든다.
def build_samsung_dataset() -> pd.DataFrame:
    return build_ticker_dataset(SAMSUNG_TICKER)


# 전처리된 모델 학습용 데이터셋을 CSV 파일로 저장한다.
def save_preprocessed_dataset(
    df: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATHS[SAMSUNG_TICKER],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index_label="date")
    print(f"[전처리 저장 완료] {output_path}: {len(df)}행, {len(df.columns)}컬럼")


# VIF 기준으로 다중공선성이 높은 컬럼을 반복 제거한다.
def remove_high_vif(df: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    df = df.copy()
    while True:
        vif = pd.DataFrame({
            "feature": df.columns,
            "VIF": [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        })
        max_vif = vif["VIF"].max()
        if max_vif < threshold:
            break
        drop_col = vif.loc[vif["VIF"].idxmax(), "feature"]
        print(f"제거: {drop_col} (VIF={max_vif:.1f})")
        df = df.drop(columns=[drop_col])
    return df


# 커맨드라인 실행 옵션을 파싱한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="국내 반도체 종목 전처리")
    parser.add_argument(
        "--ticker",
        default=SAMSUNG_TICKER,
        choices=[SAMSUNG_TICKER, SK_HYNIX_TICKER],
        help="전처리할 국내 종목 티커입니다. 삼성전자=005930, SK하이닉스=000660",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="전처리 CSV 저장 경로입니다. 생략하면 티커별 기본 경로를 사용합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output or DEFAULT_OUTPUT_PATHS[args.ticker]
    dataset = build_ticker_dataset(args.ticker)
    save_preprocessed_dataset(dataset, output_path)
    print(dataset.tail())
