from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROCESSED_CORE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "core"


def _dataset_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*_core_features.csv"))


def _model_feature_columns(columns: pd.Index) -> list[str]:
    return [
        column
        for column in columns
        if "_t_minus_" in column
        and not column.endswith("_source_date")
        and not column.endswith("_known_date")
    ]


def _audit_columns(feature_column: str) -> tuple[str, str]:
    return f"{feature_column}_source_date", f"{feature_column}_known_date"


def build_feature_coverage(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in _dataset_files(input_dir):
        data = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"], dtype={"ticker": "string"})
        ticker = str(data["ticker"].iloc[0])
        name = path.name.removesuffix("_core_features.csv")
        for column in _model_feature_columns(data.columns):
            source_date_column, known_date_column = _audit_columns(column)
            rows.append(
                {
                    "name": name,
                    "ticker": ticker,
                    "feature": column,
                    "rows": len(data),
                    "non_null_rows": int(data[column].notna().sum()),
                    "null_rate": float(data[column].isna().mean()),
                    "known_date_present": known_date_column in data.columns,
                    "source_date_present": source_date_column in data.columns,
                    "min_known_date": (
                        pd.to_datetime(data[known_date_column]).min().date().isoformat()
                        if known_date_column in data.columns and data[known_date_column].notna().any()
                        else ""
                    ),
                    "max_known_date": (
                        pd.to_datetime(data[known_date_column]).max().date().isoformat()
                        if known_date_column in data.columns and data[known_date_column].notna().any()
                        else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_label_distribution(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in _dataset_files(input_dir):
        data = pd.read_csv(path, encoding="utf-8-sig", dtype={"ticker": "string"})
        ticker = str(data["ticker"].iloc[0])
        name = path.name.removesuffix("_core_features.csv")
        grouped = data.groupby(["period_bucket", "time_split"], dropna=False)
        for (period_bucket, time_split), group in grouped:
            rows.append(
                {
                    "name": name,
                    "ticker": ticker,
                    "period_bucket": period_bucket,
                    "time_split": time_split if pd.notna(time_split) else "",
                    "rows": len(group),
                    "up_rows_1pct": int((group["target_direction_1pct"] == 1).sum()),
                    "down_rows_1pct": int((group["target_direction_1pct"] == 0).sum()),
                    "neutral_rows_1pct": int(group["target_direction_1pct"].isna().sum()),
                }
            )
    return pd.DataFrame(rows)


def build_leakage_checks(input_dir: Path) -> pd.DataFrame:
    forbidden_columns = {"open", "high", "low", "close", "volume", "return_pct"}
    rows = []
    for path in _dataset_files(input_dir):
        data = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"], dtype={"ticker": "string"})
        ticker = str(data["ticker"].iloc[0])
        name = path.name.removesuffix("_core_features.csv")
        forbidden_present = sorted(forbidden_columns & set(data.columns))
        known_date_violations = 0
        missing_audit_columns = []

        for feature_column in _model_feature_columns(data.columns):
            source_date_column, known_date_column = _audit_columns(feature_column)
            if source_date_column not in data.columns:
                missing_audit_columns.append(source_date_column)
            if known_date_column not in data.columns:
                missing_audit_columns.append(known_date_column)
            if known_date_column not in data.columns:
                continue

            lag_text = feature_column.rsplit("_t_minus_", maxsplit=1)[-1]
            lag = int(lag_text)
            cutoff = data["date"] - pd.Timedelta(days=lag)
            known_dates = pd.to_datetime(data[known_date_column])
            known_date_violations += int((known_dates.notna() & (known_dates > cutoff)).sum())

        rows.append(
            {
                "name": name,
                "ticker": ticker,
                "forbidden_same_day_columns": ",".join(forbidden_present),
                "missing_audit_columns": ",".join(sorted(missing_audit_columns)),
                "known_date_after_lag_cutoff_rows": known_date_violations,
                "passed": not forbidden_present and not missing_audit_columns and known_date_violations == 0,
            }
        )
    return pd.DataFrame(rows)


def write_validation_outputs(input_dir: Path = PROCESSED_CORE_DIR) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    feature_coverage = build_feature_coverage(input_dir)
    label_distribution = build_label_distribution(input_dir)
    leakage_checks = build_leakage_checks(input_dir)

    feature_coverage.to_csv(input_dir / "feature_coverage.csv", index=False, encoding="utf-8-sig")
    label_distribution.to_csv(input_dir / "label_distribution_by_split.csv", index=False, encoding="utf-8-sig")
    leakage_checks.to_csv(input_dir / "leakage_checks.csv", index=False, encoding="utf-8-sig")

    if not leakage_checks["passed"].all():
        raise SystemExit("preprocess leakage checks failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated core preprocessing datasets.")
    parser.add_argument(
        "--input-dir",
        default=str(PROCESSED_CORE_DIR),
        help="Directory containing generated *_core_features.csv files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_validation_outputs(Path(args.input_dir))
