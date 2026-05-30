from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay, MonthEnd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.preprocess import METADATA_ROOT, PROJECT_ROOT, TICKER_SPECS, read_current_raw_manifest

PROCESSED_CORE_DIR = PROJECT_ROOT / "data" / "processed" / "core"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "proposed_xgboost"

TARGET_COLUMN = "target_5d_3class"
RETURN_COLUMN = "future_5d_return"
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = {
    0: "down",
    1: "neutral",
    2: "up",
}
RANDOM_STATE = 42
LAGS = [1, 3, 5]


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    source_type: str = "daily"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    feature_family: str
    macro_series: tuple[SeriesSpec, ...] = ()
    use_domestic_alpha: bool = False


EXPERIMENTS = [
    ExperimentSpec(
        name="core_plus_rates",
        feature_family="core_macro+rates",
        macro_series=(
            SeriesSpec("us_10y_treasury_fred"),
            SeriesSpec("us_2y_treasury_fred"),
            SeriesSpec("us_10y_2y_spread"),
        ),
    ),
    ExperimentSpec(
        name="core_plus_broad_dollar",
        feature_family="core_macro+dollar",
        macro_series=(SeriesSpec("broad_dollar_index"),),
    ),
    ExperimentSpec(
        name="core_plus_semi_etf",
        feature_family="core_macro+semi_etf",
        macro_series=(SeriesSpec("smh"), SeriesSpec("soxx")),
    ),
    ExperimentSpec(
        name="core_plus_monthly_industry",
        feature_family="core_macro+monthly_industry",
        macro_series=(
            SeriesSpec("semiconductor_electronic_component_industrial_production", "monthly"),
            SeriesSpec("semiconductor_related_device_ppi", "monthly"),
            SeriesSpec("computers_electronic_products_new_orders", "monthly"),
        ),
    ),
    ExperimentSpec(
        name="core_plus_domestic_alpha",
        feature_family="core_macro+domestic_alpha",
        use_domestic_alpha=True,
    ),
    ExperimentSpec(
        name="proposed_daily_extended",
        feature_family="core_macro+rates+dollar+semi_etf",
        macro_series=(
            SeriesSpec("us_10y_treasury_fred"),
            SeriesSpec("us_2y_treasury_fred"),
            SeriesSpec("us_10y_2y_spread"),
            SeriesSpec("broad_dollar_index"),
            SeriesSpec("smh"),
            SeriesSpec("soxx"),
        ),
    ),
]


def dataset_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*_core_features.csv"))


def raw_run_path() -> Path:
    manifest_row = read_current_raw_manifest(METADATA_ROOT / "current_raw_manifest.csv")
    path = PROJECT_ROOT / str(manifest_row["raw_run_path"])
    if not path.exists():
        raise FileNotFoundError(f"raw run path does not exist: {path}")
    return path


def core_feature_columns(columns: pd.Index) -> list[str]:
    return [
        column
        for column in columns
        if "_t_minus_" in column
        and not column.endswith("_source_date")
        and not column.endswith("_known_date")
    ]


def read_series(path: Path, value_columns: list[str] | None = None) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])
    if "date" not in data.columns:
        raise ValueError(f"missing date column: {path}")
    if value_columns is None:
        value_columns = [column for column in data.columns if column != "date"]
    missing = set(value_columns) - set(data.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return data[["date", *value_columns]].sort_values("date")


def known_dates(source_dates: pd.Series, source_type: str) -> pd.Series:
    if source_type == "monthly":
        return source_dates + MonthEnd(1) + BDay(5)
    if source_type == "daily":
        return source_dates
    raise ValueError(f"unsupported source_type: {source_type}")


def make_asof_lag_features(
    base_dates: pd.Series,
    series: pd.DataFrame,
    *,
    value_columns: list[str],
    source_type: str,
) -> pd.DataFrame:
    features = pd.DataFrame({"date": pd.to_datetime(base_dates).sort_values()})

    for lag in LAGS:
        cutoff_column = f"cutoff_t_minus_{lag}"
        left = features[["date"]].copy()
        left[cutoff_column] = left["date"] - pd.Timedelta(days=lag)
        left = left.sort_values(cutoff_column)

        right = series[["date", *value_columns]].dropna(how="all", subset=value_columns).copy()
        right = right.rename(columns={"date": "source_date"})
        right["known_date"] = known_dates(right["source_date"], source_type)
        right = right.sort_values("known_date")

        merged = pd.merge_asof(
            left,
            right,
            left_on=cutoff_column,
            right_on="known_date",
            direction="backward",
        )
        renamed = merged[["date", "source_date", "known_date", *value_columns]].copy()
        rename_map = {
            column: f"{column}_t_minus_{lag}"
            for column in value_columns
        }
        renamed = renamed.rename(columns=rename_map)
        for column in value_columns:
            renamed[f"{column}_t_minus_{lag}_source_date"] = merged["source_date"]
            renamed[f"{column}_t_minus_{lag}_known_date"] = merged["known_date"]
        renamed = renamed.drop(columns=["source_date", "known_date"])
        features = features.merge(renamed, on="date", how="left")

    return features


def load_macro_features(raw_path: Path, spec: SeriesSpec, base_dates: pd.Series) -> pd.DataFrame:
    path = raw_path / "macro" / f"{spec.name}.csv"
    series = read_series(path, [spec.name])
    return make_asof_lag_features(
        base_dates,
        series,
        value_columns=[spec.name],
        source_type=spec.source_type,
    )


def alpha_file_name(ticker: str) -> str | None:
    if ticker == "005930":
        return "samsung_electronics_alpha.csv"
    if ticker == "000660":
        return "sk_hynix_alpha.csv"
    return None


def load_alpha_features(raw_path: Path, ticker: str, base_dates: pd.Series) -> pd.DataFrame | None:
    file_name = alpha_file_name(ticker)
    if file_name is None:
        return None
    path = raw_path / "alpha" / file_name
    alpha = read_series(path, ["foreign_net_buy_value", "institution_net_buy_value"])
    alpha = alpha.rename(
        columns={
            "foreign_net_buy_value": "alpha_foreign_net_buy_value",
            "institution_net_buy_value": "alpha_institution_net_buy_value",
        }
    )
    return make_asof_lag_features(
        base_dates,
        alpha,
        value_columns=["alpha_foreign_net_buy_value", "alpha_institution_net_buy_value"],
        source_type="daily",
    )


def extended_feature_columns(columns: pd.Index) -> list[str]:
    return [
        column
        for column in columns
        if "_t_minus_" in column
        and not column.endswith("_source_date")
        and not column.endswith("_known_date")
        and column not in {
            "nasdaq_t_minus_1",
            "sox_t_minus_1",
            "vix_t_minus_1",
            "us_rate_t_minus_1",
            "usd_krw_t_minus_1",
            "nasdaq_t_minus_3",
            "sox_t_minus_3",
            "vix_t_minus_3",
            "us_rate_t_minus_3",
            "usd_krw_t_minus_3",
            "nasdaq_t_minus_5",
            "sox_t_minus_5",
            "vix_t_minus_5",
            "us_rate_t_minus_5",
            "usd_krw_t_minus_5",
        }
    ]


def build_experiment_frame(data: pd.DataFrame, raw_path: Path, experiment: ExperimentSpec) -> pd.DataFrame:
    result = data.copy()
    for series_spec in experiment.macro_series:
        features = load_macro_features(raw_path, series_spec, result["date"])
        result = result.merge(features, on="date", how="left")

    if experiment.use_domestic_alpha:
        ticker = str(result["ticker"].iloc[0])
        alpha_features = load_alpha_features(raw_path, ticker, result["date"])
        if alpha_features is not None:
            result = result.merge(alpha_features, on="date", how="left")

    return result


def prepare_split(
    data: pd.DataFrame,
    feature_columns: list[str],
    split_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    subset = data[data["time_split"] == split_name].copy()
    subset = subset.dropna(subset=feature_columns + [TARGET_COLUMN, RETURN_COLUMN])
    x = subset[feature_columns].astype(float)
    y = subset[TARGET_COLUMN].astype(int)
    realized_return = subset[RETURN_COLUMN].astype(float)
    return x, y, realized_return


def sample_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    total = len(y)
    class_count = len(CLASS_LABELS)
    return y.map(lambda label: total / (class_count * counts[int(label)])).to_numpy()


def build_model() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_LABELS),
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate_predictions(
    *,
    name: str,
    ticker: str,
    experiment: ExperimentSpec,
    split_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    realized_return: pd.Series,
) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    precision = precision_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    result: dict[str, object] = {
        "name": name,
        "ticker": ticker,
        "experiment": experiment.name,
        "feature_family": experiment.feature_family,
        "split": split_name,
        "rows": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=CLASS_LABELS, average="weighted", zero_division=0),
        "log_loss": log_loss(y_true, y_proba, labels=CLASS_LABELS),
    }

    try:
        result["roc_auc_ovr_weighted"] = roc_auc_score(
            y_true,
            y_proba,
            labels=CLASS_LABELS,
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        result["roc_auc_ovr_weighted"] = np.nan

    for true_label in CLASS_LABELS:
        for pred_label in CLASS_LABELS:
            result[f"cm_{CLASS_NAMES[true_label]}_{CLASS_NAMES[pred_label]}"] = int(
                matrix[true_label, pred_label]
            )
    for index, label in enumerate(CLASS_LABELS):
        result[f"precision_{CLASS_NAMES[label]}"] = precision[index]
        result[f"recall_{CLASS_NAMES[label]}"] = recall[index]

    returns = pd.DataFrame({"prediction": y_pred, RETURN_COLUMN: realized_return.to_numpy()})
    mean_returns = returns.groupby("prediction")[RETURN_COLUMN].mean()
    for label in CLASS_LABELS:
        result[f"mean_future_5d_return_pred_{CLASS_NAMES[label]}"] = mean_returns.get(label, np.nan)
    result["long_short_spread_pred_up_minus_down"] = mean_returns.get(2, np.nan) - mean_returns.get(0, np.nan)
    return result


def run_experiments(input_dir: Path = PROCESSED_CORE_DIR, results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_run_path()
    metric_rows = []
    importance_rows = []
    feature_set_rows = []

    for path in dataset_files(input_dir):
        data = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"], dtype={"ticker": "string"})
        name = path.name.removesuffix("_core_features.csv")
        ticker = str(data["ticker"].iloc[0])
        core_columns = core_feature_columns(data.columns)

        for experiment in EXPERIMENTS:
            model_frame = build_experiment_frame(data, raw_path, experiment)
            extended_columns = extended_feature_columns(model_frame.columns)
            if experiment.use_domestic_alpha and not extended_columns:
                continue
            feature_columns = core_columns + extended_columns
            x_train, y_train, _ = prepare_split(model_frame, feature_columns, "train")
            if x_train.empty:
                continue
            if set(y_train) != set(CLASS_LABELS):
                raise ValueError(f"train split lacks all target classes for {name} {experiment.name}")

            model = build_model()
            model.fit(x_train, y_train, sample_weight=sample_weights(y_train))

            feature_set_rows.append(
                {
                    "name": name,
                    "ticker": ticker,
                    "experiment": experiment.name,
                    "feature_family": experiment.feature_family,
                    "feature_count": len(feature_columns),
                    "core_feature_count": len(core_columns),
                    "extended_feature_count": len(extended_columns),
                    "features": ",".join(feature_columns),
                }
            )

            for split_name in ["validation", "test", "recent_regime"]:
                x_eval, y_eval, realized_return = prepare_split(model_frame, feature_columns, split_name)
                if x_eval.empty:
                    continue
                metric_rows.append(
                    evaluate_predictions(
                        name=name,
                        ticker=ticker,
                        experiment=experiment,
                        split_name=split_name,
                        y_true=y_eval,
                        y_pred=model.predict(x_eval),
                        y_proba=model.predict_proba(x_eval),
                        realized_return=realized_return,
                    )
                )

            for feature, importance in zip(feature_columns, model.feature_importances_):
                importance_rows.append(
                    {
                        "name": name,
                        "ticker": ticker,
                        "experiment": experiment.name,
                        "feature_family": experiment.feature_family,
                        "feature": feature,
                        "xgboost_feature_importance": float(importance),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(results_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(importance_rows).to_csv(
        results_dir / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(feature_set_rows).to_csv(results_dir / "feature_sets.csv", index=False, encoding="utf-8-sig")
    if not metrics.empty:
        summary = (
            metrics[metrics["split"].isin(["test", "recent_regime"])]
            .pivot_table(
                index=["name", "ticker", "experiment", "feature_family"],
                columns="split",
                values=["f1_macro", "balanced_accuracy", "roc_auc_ovr_weighted"],
            )
            .reset_index()
        )
        summary.columns = [
            "_".join(column).strip("_") if isinstance(column, tuple) else column
            for column in summary.columns
        ]
        summary.to_csv(results_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Proposed XGBoost extended ablation experiments.")
    parser.add_argument("--input-dir", default=str(PROCESSED_CORE_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(Path(args.input_dir), Path(args.results_dir))
