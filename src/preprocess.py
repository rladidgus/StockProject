import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def sync_timeseries(macro_df: pd.DataFrame, target_ticker: str) -> pd.DataFrame:
    if target_ticker in ["005930", "000660"]:
        macro_df = macro_df.shift(1)
    return macro_df


def make_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int] = [1, 3, 5]) -> pd.DataFrame:
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


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


LAG_COLS = ["nasdaq", "vix", "us_rate", "usd_krw", "sox"]
