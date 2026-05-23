from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BDay, MonthEnd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
METADATA_ROOT = DATA_ROOT / "metadata"
PROCESSED_ROOT = DATA_ROOT / "processed"

MODEL_START = "2018-01-01"
MODEL_END = "2024-12-31"
RECENT_START = "2025-01-01"

CORE_MACRO_COLUMNS = ["nasdaq", "sox", "vix", "us_rate", "usd_krw"]
CORE_MACRO_LAGS = [1, 3, 5]
TARGET_THRESHOLD = 0.01
SAME_DAY_PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "return_pct"]


@dataclass(frozen=True)
class TickerSpec:
    ticker: str
    name: str
    price_file: str
    market: str


TICKER_SPECS = [
    TickerSpec("005930", "samsung_electronics", "samsung_electronics.csv", "KR"),
    TickerSpec("000660", "sk_hynix", "sk_hynix.csv", "KR"),
    TickerSpec("NVDA", "nvidia", "nvidia.csv", "US"),
]


def read_current_raw_manifest(path: Path = METADATA_ROOT / "current_raw_manifest.csv") -> pd.Series:
    manifest = pd.read_csv(path, encoding="utf-8-sig")
    if manifest.empty:
        raise ValueError(f"empty raw manifest: {path}")
    required = {"run_id", "raw_run_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    return manifest.iloc[-1]


def read_date_indexed_csv(path: Path, *, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig", dtype=dtype)
    if "date" not in data.columns:
        raise ValueError(f"missing date column: {path}")
    data["date"] = pd.to_datetime(data["date"])
    return data.set_index("date").sort_index()


def load_core_macro(raw_run_path: Path) -> pd.DataFrame:
    macro_path = raw_run_path / "macro" / "core_macro_features.csv"
    macro = read_date_indexed_csv(macro_path)
    missing = [column for column in CORE_MACRO_COLUMNS if column not in macro.columns]
    if missing:
        raise ValueError(f"core macro missing columns: {missing}")
    return macro[CORE_MACRO_COLUMNS].copy()


def build_macro_features_for_dates(macro: pd.DataFrame, prediction_dates: pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.DataFrame({"date": pd.to_datetime(prediction_dates).sort_values()})
    features = dates.copy()

    for lag in CORE_MACRO_LAGS:
        cutoff_column = f"cutoff_t_minus_{lag}"
        features[cutoff_column] = features["date"] - pd.Timedelta(days=lag)
        left = features[["date", cutoff_column]].sort_values(cutoff_column)

        for column in CORE_MACRO_COLUMNS:
            series = macro[[column]].dropna().reset_index().rename(
                columns={"date": "source_date", column: f"{column}_t_minus_{lag}"}
            )
            if column == "us_rate":
                # FEDFUNDS raw rows are monthly period observations dated on the
                # first day of the month, not publication dates. Use a conservative
                # known-after date to avoid making the full-month value available
                # before the month has finished and data could plausibly be released.
                series["known_date"] = series["source_date"] + MonthEnd(1) + BDay(5)
            else:
                series["known_date"] = series["source_date"]
            series = series.sort_values("known_date")
            merged = pd.merge_asof(
                left,
                series,
                left_on=cutoff_column,
                right_on="known_date",
                direction="backward",
            )
            features = features.merge(
                merged[
                    [
                        "date",
                        f"{column}_t_minus_{lag}",
                        "source_date",
                        "known_date",
                    ]
                ].rename(
                    columns={
                        "source_date": f"{column}_t_minus_{lag}_source_date",
                        "known_date": f"{column}_t_minus_{lag}_known_date",
                    }
                ),
                on="date",
                how="left",
            )
        features = features.drop(columns=[cutoff_column])

    return features.set_index("date").sort_index()


def load_price_frame(raw_run_path: Path, spec: TickerSpec) -> pd.DataFrame:
    price_path = raw_run_path / "prices" / spec.price_file
    prices = read_date_indexed_csv(price_path, dtype={"ticker": "string"})
    required = {"open", "high", "low", "close", "volume", "ticker"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"{spec.name} price data missing columns: {sorted(missing)}")
    return prices[["open", "high", "low", "close", "volume", "ticker"]].copy()


def is_model_feature_column(column: str) -> bool:
    return (
        "_t_minus_" in column
        and not column.endswith("_source_date")
        and not column.endswith("_known_date")
    )


def add_return_and_labels(data: pd.DataFrame, threshold: float = TARGET_THRESHOLD) -> pd.DataFrame:
    result = data.copy()
    result["return_pct"] = result["close"].pct_change()
    result["target_up_1pct"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["target_down_1pct"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    known_return = result["return_pct"].notna()
    result.loc[known_return, "target_up_1pct"] = (
        result.loc[known_return, "return_pct"] >= threshold
    ).astype("int64")
    result.loc[known_return, "target_down_1pct"] = (
        result.loc[known_return, "return_pct"] <= -threshold
    ).astype("int64")
    result["target_direction_1pct"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[result["return_pct"] >= threshold, "target_direction_1pct"] = 1
    result.loc[result["return_pct"] <= -threshold, "target_direction_1pct"] = 0
    return result


def assign_period(date_index: pd.DatetimeIndex) -> pd.Series:
    periods = pd.Series("raw_eda", index=date_index, dtype="string")
    periods.loc[(date_index >= MODEL_START) & (date_index <= MODEL_END)] = "model_2018_2024"
    periods.loc[date_index >= RECENT_START] = "recent_regime_2025_plus"
    return periods


def assign_time_split(data: pd.DataFrame) -> pd.Series:
    split = pd.Series(pd.NA, index=data.index, dtype="string")
    model_mask = data["period_bucket"] == "model_2018_2024"
    model_index = data.index[model_mask]
    model_count = len(model_index)
    if model_count == 0:
        return split

    train_end = int(model_count * 0.70)
    validation_end = int(model_count * 0.85)
    split.loc[model_index[:train_end]] = "train"
    split.loc[model_index[train_end:validation_end]] = "validation"
    split.loc[model_index[validation_end:]] = "test"
    split.loc[data["period_bucket"] == "recent_regime_2025_plus"] = "recent_regime"
    return split


def build_core_dataset(raw_run_path: Path, spec: TickerSpec, macro: pd.DataFrame) -> pd.DataFrame:
    prices = load_price_frame(raw_run_path, spec)
    macro_features = build_macro_features_for_dates(macro, prices.index)
    dataset = prices.join(macro_features, how="left")
    dataset = add_return_and_labels(dataset)
    dataset["market"] = spec.market
    dataset["period_bucket"] = assign_period(dataset.index)
    dataset["time_split"] = assign_time_split(dataset)
    dataset = dataset.reset_index()
    feature_columns = [
        column
        for column in dataset.columns
        if "_t_minus_" in column
    ]
    output_columns = [
        "date",
        "ticker",
        "market",
        "period_bucket",
        "time_split",
        "target_up_1pct",
        "target_down_1pct",
        "target_direction_1pct",
    ] + feature_columns
    return dataset[output_columns]


def summarize_dataset(dataset: pd.DataFrame, spec: TickerSpec, run_id: str) -> dict[str, object]:
    feature_columns = [
        column
        for column in dataset.columns
        if is_model_feature_column(column)
    ]
    model_window = dataset[dataset["period_bucket"] == "model_2018_2024"]
    recent_window = dataset[dataset["period_bucket"] == "recent_regime_2025_plus"]
    split_counts = dataset["time_split"].value_counts(dropna=False).to_dict()
    model_feature_null_rate = model_window[feature_columns].isna().mean().max()
    return {
        "run_id": run_id,
        "ticker": spec.ticker,
        "name": spec.name,
        "market": spec.market,
        "rows": len(dataset),
        "data_start": dataset["date"].min().date().isoformat(),
        "data_end": dataset["date"].max().date().isoformat(),
        "model_rows_2018_2024": len(model_window),
        "recent_rows_2025_plus": len(recent_window),
        "train_rows": int(split_counts.get("train", 0)),
        "validation_rows": int(split_counts.get("validation", 0)),
        "test_rows": int(split_counts.get("test", 0)),
        "recent_regime_rows": int(split_counts.get("recent_regime", 0)),
        "neutral_rows_1pct": int(dataset["target_direction_1pct"].isna().sum()),
        "up_rows_1pct": int((dataset["target_direction_1pct"] == 1).sum()),
        "down_rows_1pct": int((dataset["target_direction_1pct"] == 0).sum()),
        "feature_columns": ",".join(feature_columns),
        "excluded_from_model_columns": ",".join(SAME_DAY_PRICE_COLUMNS),
        "max_core_feature_null_rate": dataset[feature_columns].isna().mean().max(),
        "max_model_feature_null_rate": model_feature_null_rate,
    }


def write_core_datasets(output_dir: Path = PROCESSED_ROOT / "core") -> None:
    manifest_row = read_current_raw_manifest()
    run_id = str(manifest_row["run_id"])
    raw_run_path = PROJECT_ROOT / str(manifest_row["raw_run_path"])
    if not raw_run_path.exists():
        raise FileNotFoundError(f"raw run path does not exist: {raw_run_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in [
        "*_core_features.csv",
        "core_dataset_summary.csv",
        "feature_coverage.csv",
        "label_distribution_by_split.csv",
        "leakage_checks.csv",
    ]:
        for stale_path in output_dir.glob(pattern):
            stale_path.unlink()

    macro = load_core_macro(raw_run_path)
    summary_rows = []

    for spec in TICKER_SPECS:
        dataset = build_core_dataset(raw_run_path, spec, macro)
        output_path = output_dir / f"{spec.name}_core_features.csv"
        dataset.to_csv(output_path, index=False, encoding="utf-8-sig")
        summary_rows.append(summarize_dataset(dataset, spec, run_id))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "core_dataset_summary.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build core model-ready datasets from the adopted raw run.")
    parser.add_argument(
        "--output-dir",
        default=str(PROCESSED_ROOT / "core"),
        help="Directory for generated core datasets.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_core_datasets(Path(args.output_dir))
