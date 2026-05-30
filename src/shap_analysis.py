from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import DMatrix

from src.preprocess import PROJECT_ROOT
from src.proposed_xgboost import (
    CLASS_LABELS,
    CLASS_NAMES,
    EXPERIMENTS,
    PROCESSED_CORE_DIR,
    TARGET_COLUMN,
    build_experiment_frame,
    build_model,
    core_feature_columns,
    dataset_files,
    extended_feature_columns,
    raw_run_path,
    sample_weights,
)

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "shap_analysis"
PROPOSED_RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "proposed_xgboost"
RETURN_COLUMN = "future_5d_return"


@dataclass(frozen=True)
class SelectedExperiment:
    ticker: str
    name: str
    experiment: str
    feature_family: str
    selection_metric: float


def read_selected_experiments(metrics_path: Path) -> dict[str, SelectedExperiment]:
    metrics = pd.read_csv(metrics_path, encoding="utf-8-sig")
    test_metrics = metrics[metrics["split"] == "test"].copy()
    if test_metrics.empty:
        raise ValueError(f"no test metrics available: {metrics_path}")
    selected = {}
    for ticker, group in test_metrics.groupby("ticker"):
        best = group.sort_values(["f1_macro", "balanced_accuracy"], ascending=False).iloc[0]
        selected[ticker] = SelectedExperiment(
            ticker=str(best["ticker"]),
            name=str(best["name"]),
            experiment=str(best["experiment"]),
            feature_family=str(best["feature_family"]),
            selection_metric=float(best["f1_macro"]),
        )
    return selected


def experiment_by_name(name: str):
    for experiment in EXPERIMENTS:
        if experiment.name == name:
            return experiment
    raise ValueError(f"unknown experiment: {name}")


def split_frame(data: pd.DataFrame, feature_columns: list[str], split_name: str) -> pd.DataFrame:
    subset = data[data["time_split"] == split_name].copy()
    return subset.dropna(subset=feature_columns + [TARGET_COLUMN, RETURN_COLUMN])


def contribution_tensor(model, x: pd.DataFrame) -> np.ndarray:
    raw = model.get_booster().predict(DMatrix(x), pred_contribs=True)
    feature_with_bias_count = len(x.columns) + 1
    if raw.ndim == 3:
        return raw
    if raw.ndim == 2 and raw.shape[1] == len(CLASS_LABELS) * feature_with_bias_count:
        return raw.reshape(raw.shape[0], len(CLASS_LABELS), feature_with_bias_count)
    if raw.ndim == 2 and raw.shape[1] == feature_with_bias_count:
        return raw[:, np.newaxis, :]
    raise ValueError(f"unexpected SHAP contribution shape: {raw.shape}")


def global_importance_rows(
    *,
    name: str,
    ticker: str,
    experiment: str,
    feature_family: str,
    split_name: str,
    feature_columns: list[str],
    contributions: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for class_index, class_label in enumerate(CLASS_LABELS):
        shap_values = contributions[:, class_index, :-1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_signed = shap_values.mean(axis=0)
        for feature, abs_value, signed_value in zip(feature_columns, mean_abs, mean_signed):
            rows.append(
                {
                    "name": name,
                    "ticker": ticker,
                    "experiment": experiment,
                    "feature_family": feature_family,
                    "split": split_name,
                    "class_label": class_label,
                    "class_name": CLASS_NAMES[class_label],
                    "feature": feature,
                    "mean_abs_shap": float(abs_value),
                    "mean_shap": float(signed_value),
                }
            )
    overall = np.abs(contributions[:, :, :-1]).mean(axis=(0, 1))
    for feature, abs_value in zip(feature_columns, overall):
        rows.append(
            {
                "name": name,
                "ticker": ticker,
                "experiment": experiment,
                "feature_family": feature_family,
                "split": split_name,
                "class_label": "overall",
                "class_name": "overall",
                "feature": feature,
                "mean_abs_shap": float(abs_value),
                "mean_shap": np.nan,
            }
        )
    return rows


def local_case_rows(
    *,
    data: pd.DataFrame,
    name: str,
    ticker: str,
    experiment: str,
    feature_family: str,
    split_name: str,
    feature_columns: list[str],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    contributions: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    working = data[["date", TARGET_COLUMN, RETURN_COLUMN]].copy().reset_index(drop=True)
    working["prediction"] = predictions
    working["prediction_probability"] = probabilities.max(axis=1)

    for class_label in [0, 2]:
        candidates = working[working["prediction"] == class_label]
        if candidates.empty:
            continue
        selected_index = int(candidates.sort_values("prediction_probability", ascending=False).index[0])
        class_index = CLASS_LABELS.index(class_label)
        shap_values = contributions[selected_index, class_index, :-1]
        bias = float(contributions[selected_index, class_index, -1])
        top_indices = np.argsort(np.abs(shap_values))[::-1][:12]
        for rank, feature_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "name": name,
                    "ticker": ticker,
                    "experiment": experiment,
                    "feature_family": feature_family,
                    "split": split_name,
                    "date": working.loc[selected_index, "date"],
                    "true_label": int(working.loc[selected_index, TARGET_COLUMN]),
                    "predicted_label": int(class_label),
                    "predicted_class_name": CLASS_NAMES[class_label],
                    "prediction_probability": float(working.loc[selected_index, "prediction_probability"]),
                    "future_5d_return": float(working.loc[selected_index, RETURN_COLUMN]),
                    "base_value": bias,
                    "rank": rank,
                    "feature": feature_columns[feature_index],
                    "shap_value": float(shap_values[feature_index]),
                    "feature_value": float(data.iloc[selected_index][feature_columns[feature_index]]),
                }
            )
    return rows


def plot_global_importance(data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = data[(data["split"] == "test") & (data["class_name"] == "overall")]
    for (name, ticker, experiment), group in overall.groupby(["name", "ticker", "experiment"]):
        top = group.sort_values("mean_abs_shap", ascending=False).head(15).sort_values("mean_abs_shap")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["feature"], top["mean_abs_shap"], color="#356859")
        ax.set_title(f"{name} {experiment} global SHAP")
        ax.set_xlabel("Mean absolute SHAP contribution")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}_{ticker}_{experiment}_global_bar.png", dpi=160)
        plt.close(fig)


def plot_local_cases(data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for (name, ticker, experiment, date, predicted_class), group in data.groupby(
        ["name", "ticker", "experiment", "date", "predicted_class_name"]
    ):
        top = group.sort_values("rank", ascending=False)
        colors = np.where(top["shap_value"] >= 0, "#b23a48", "#2f6f9f")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["feature"], top["shap_value"], color=colors)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"{name} {predicted_class} prediction on {pd.to_datetime(date).date()}")
        ax.set_xlabel("SHAP contribution to predicted class logit")
        ax.set_ylabel("")
        fig.tight_layout()
        safe_date = pd.to_datetime(date).date().isoformat()
        fig.savefig(output_dir / f"{name}_{ticker}_{experiment}_{safe_date}_{predicted_class}_local.png", dpi=160)
        plt.close(fig)


def run_shap_analysis(
    input_dir: Path = PROCESSED_CORE_DIR,
    proposed_results_dir: Path = PROPOSED_RESULTS_DIR,
    results_dir: Path = RESULTS_DIR,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    selected = read_selected_experiments(proposed_results_dir / "model_metrics.csv")
    raw_path = raw_run_path()
    selected_rows = []
    global_rows = []
    local_rows = []

    for path in dataset_files(input_dir):
        core_data = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"], dtype={"ticker": "string"})
        name = path.name.removesuffix("_core_features.csv")
        ticker = str(core_data["ticker"].iloc[0])
        if ticker not in selected:
            continue

        selection = selected[ticker]
        experiment = experiment_by_name(selection.experiment)
        core_columns = core_feature_columns(core_data.columns)
        model_frame = build_experiment_frame(core_data, raw_path, experiment)
        feature_columns = core_columns + extended_feature_columns(model_frame.columns)
        train = split_frame(model_frame, feature_columns, "train")
        x_train = train[feature_columns].astype(float)
        y_train = train[TARGET_COLUMN].astype(int)

        model = build_model()
        model.fit(x_train, y_train, sample_weight=sample_weights(y_train))

        selected_rows.append(
            {
                "name": name,
                "ticker": ticker,
                "experiment": selection.experiment,
                "feature_family": selection.feature_family,
                "selected_by": "test_f1_macro",
                "selection_metric": selection.selection_metric,
                "feature_count": len(feature_columns),
            }
        )

        for split_name in ["test", "recent_regime"]:
            evaluation = split_frame(model_frame, feature_columns, split_name)
            if evaluation.empty:
                continue
            x_eval = evaluation[feature_columns].astype(float)
            predictions = model.predict(x_eval)
            probabilities = model.predict_proba(x_eval)
            contributions = contribution_tensor(model, x_eval)
            global_rows.extend(
                global_importance_rows(
                    name=name,
                    ticker=ticker,
                    experiment=selection.experiment,
                    feature_family=selection.feature_family,
                    split_name=split_name,
                    feature_columns=feature_columns,
                    contributions=contributions,
                )
            )
            if split_name == "test":
                local_rows.extend(
                    local_case_rows(
                        data=evaluation,
                        name=name,
                        ticker=ticker,
                        experiment=selection.experiment,
                        feature_family=selection.feature_family,
                        split_name=split_name,
                        feature_columns=feature_columns,
                        predictions=predictions,
                        probabilities=probabilities,
                        contributions=contributions,
                    )
                )

    selected_frame = pd.DataFrame(selected_rows)
    global_frame = pd.DataFrame(global_rows)
    local_frame = pd.DataFrame(local_rows)
    selected_frame.to_csv(results_dir / "selected_models.csv", index=False, encoding="utf-8-sig")
    global_frame.to_csv(results_dir / "global_shap_importance.csv", index=False, encoding="utf-8-sig")
    local_frame.to_csv(results_dir / "local_shap_cases.csv", index=False, encoding="utf-8-sig")
    plot_global_importance(global_frame, results_dir)
    plot_local_cases(local_frame, results_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP-style XGBoost contribution artifacts.")
    parser.add_argument("--input-dir", default=str(PROCESSED_CORE_DIR))
    parser.add_argument("--proposed-results-dir", default=str(PROPOSED_RESULTS_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_shap_analysis(
        input_dir=Path(args.input_dir),
        proposed_results_dir=Path(args.proposed_results_dir),
        results_dir=Path(args.results_dir),
    )
