from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "technical_baseline_xgboost"

TARGET_COLUMN = "target_5d_3class"
RETURN_COLUMN = "future_5d_return"
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = {
    0: "down",
    1: "neutral",
    2: "up",
}
RANDOM_STATE = 42


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    feature_family: str
    feature_columns: list[str]


def dataset_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*_core_features.csv"))


def raw_run_path() -> Path:
    manifest_row = read_current_raw_manifest(METADATA_ROOT / "current_raw_manifest.csv")
    path = PROJECT_ROOT / str(manifest_row["raw_run_path"])
    if not path.exists():
        raise FileNotFoundError(f"raw run path does not exist: {path}")
    return path


def load_price_frame(raw_path: Path, ticker: str) -> pd.DataFrame:
    spec_by_ticker = {spec.ticker: spec for spec in TICKER_SPECS}
    spec = spec_by_ticker[ticker]
    path = raw_path / "prices" / spec.price_file
    prices = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"], dtype={"ticker": "string"})
    required = {"date", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return prices.sort_values("date").reset_index(drop=True)


def add_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    close = result["close"].astype(float)
    ma5 = close.rolling(5, min_periods=5).mean()
    ma10 = close.rolling(10, min_periods=10).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + rs))

    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    band_range = (upper - lower).replace(0, np.nan)

    result["tech_return_1d"] = close.pct_change()
    result["tech_ma5_ratio"] = close / ma5 - 1
    result["tech_ma10_ratio"] = close / ma10 - 1
    result["tech_ma20_ratio"] = close / ma20 - 1
    result["tech_rsi14"] = rsi14
    result["tech_bb_percent_b"] = (close - lower) / band_range
    result["tech_bb_width"] = band_range / ma20
    result["tech_volatility_20d"] = close.pct_change().rolling(20, min_periods=20).std()
    return result


def core_feature_columns(columns: pd.Index) -> list[str]:
    return [
        column
        for column in columns
        if "_t_minus_" in column
        and not column.endswith("_source_date")
        and not column.endswith("_known_date")
    ]


def technical_feature_columns(columns: pd.Index) -> list[str]:
    return [column for column in columns if column.startswith("tech_")]


def build_model_frame(path: Path, raw_path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"], dtype={"ticker": "string"})
    ticker = str(data["ticker"].iloc[0])
    prices = add_technical_features(load_price_frame(raw_path, ticker))
    technical = prices[["date", *technical_feature_columns(prices.columns)]]
    return data.merge(technical, on="date", how="left")


def make_experiments(data: pd.DataFrame) -> list[ExperimentSpec]:
    technical = technical_feature_columns(data.columns)
    core = core_feature_columns(data.columns)
    if not technical:
        raise ValueError("no technical feature columns found")
    if not core:
        raise ValueError("no core feature columns found")
    return [
        ExperimentSpec("technical_baseline_xgboost", "technical", technical),
        ExperimentSpec("core_only_xgboost", "core_macro", core),
    ]


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


def run_experiments(
    input_dir: Path = PROCESSED_CORE_DIR,
    results_dir: Path = RESULTS_DIR,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_run_path()
    metric_rows = []
    importance_rows = []
    feature_set_rows = []

    for path in dataset_files(input_dir):
        data = build_model_frame(path, raw_path)
        name = path.name.removesuffix("_core_features.csv")
        ticker = str(data["ticker"].iloc[0])

        for experiment in make_experiments(data):
            x_train, y_train, _ = prepare_split(data, experiment.feature_columns, "train")
            if x_train.empty:
                raise ValueError(f"empty train split for {name} {experiment.name}")
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
                    "feature_count": len(experiment.feature_columns),
                    "features": ",".join(experiment.feature_columns),
                }
            )

            for split_name in ["validation", "test", "recent_regime"]:
                x_eval, y_eval, realized_return = prepare_split(data, experiment.feature_columns, split_name)
                if x_eval.empty:
                    continue
                y_pred = model.predict(x_eval)
                y_proba = model.predict_proba(x_eval)
                metric_rows.append(
                    evaluate_predictions(
                        name=name,
                        ticker=ticker,
                        experiment=experiment,
                        split_name=split_name,
                        y_true=y_eval,
                        y_pred=y_pred,
                        y_proba=y_proba,
                        realized_return=realized_return,
                    )
                )

            for feature, importance in zip(experiment.feature_columns, model.feature_importances_):
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

    pd.DataFrame(metric_rows).to_csv(results_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(importance_rows).to_csv(
        results_dir / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(feature_set_rows).to_csv(results_dir / "feature_sets.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run technical baseline and core-only XGBoost checkpoints.")
    parser.add_argument("--input-dir", default=str(PROCESSED_CORE_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(Path(args.input_dir), Path(args.results_dir))
