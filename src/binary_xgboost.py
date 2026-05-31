from __future__ import annotations

import argparse
from itertools import product
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

from src.preprocess import PROJECT_ROOT
from src.proposed_xgboost import RETURN_COLUMN, raw_run_path
from src.technical_extended_search import (
    FEATURE_RECIPES,
    PARAM_GRID,
    PROCESSED_CORE_DIR,
    RANDOM_STATE,
    FeatureRecipe,
    ModelParams,
    build_recipe_frame,
    dataset_files,
    load_core_dataset,
    recipe_feature_columns,
)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "binary_xgboost"

SOURCE_TARGET_COLUMN = "target_5d_3class"
BINARY_TARGET_COLUMN = "target_5d_binary"
BINARY_LABELS = [0, 1]
BINARY_NAMES = {
    0: "down",
    1: "up",
}
SOURCE_TO_BINARY = {
    0: 0,
    2: 1,
}
MIN_PREDICTED_CLASS_SHARE = 0.10


def add_binary_target(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result[BINARY_TARGET_COLUMN] = result[SOURCE_TARGET_COLUMN].map(SOURCE_TO_BINARY)
    return result


def prepare_split(
    data: pd.DataFrame,
    feature_columns: list[str],
    split_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    subset = data[data["time_split"] == split_name].copy()
    subset[feature_columns] = subset[feature_columns].replace([np.inf, -np.inf], np.nan)
    subset = subset.dropna(subset=feature_columns + [BINARY_TARGET_COLUMN, RETURN_COLUMN])
    x = subset[feature_columns].astype(float)
    y = subset[BINARY_TARGET_COLUMN].astype(int)
    realized_return = subset[RETURN_COLUMN].astype(float)
    return x, y, realized_return


def sample_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    total = len(y)
    class_count = len(BINARY_LABELS)
    return y.map(lambda label: total / (class_count * counts[int(label)])).to_numpy()


def build_model(params: ModelParams) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        min_child_weight=params.min_child_weight,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        reg_lambda=params.reg_lambda,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def prediction_health(y_pred: np.ndarray) -> dict[str, object]:
    counts = pd.Series(y_pred).value_counts(normalize=True).to_dict()
    down_share = float(counts.get(0, 0.0))
    up_share = float(counts.get(1, 0.0))
    min_share = min(down_share, up_share)
    return {
        "pred_down_share": down_share,
        "pred_up_share": up_share,
        "min_predicted_class_share": min_share,
        "passed_prediction_balance": min_share >= MIN_PREDICTED_CLASS_SHARE,
    }


def evaluate_predictions(
    *,
    name: str,
    ticker: str,
    recipe: FeatureRecipe,
    params: ModelParams,
    split_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba_up: np.ndarray,
    realized_return: pd.Series,
) -> dict[str, object]:
    y_proba = np.column_stack([1 - y_proba_up, y_proba_up])
    matrix = confusion_matrix(y_true, y_pred, labels=BINARY_LABELS)
    precision = precision_score(y_true, y_pred, labels=BINARY_LABELS, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=BINARY_LABELS, average=None, zero_division=0)
    result: dict[str, object] = {
        "name": name,
        "ticker": ticker,
        "recipe": recipe.name,
        "params": params.name,
        "split": split_name,
        "rows": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=BINARY_LABELS, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=BINARY_LABELS, average="weighted", zero_division=0),
        "log_loss": log_loss(y_true, y_proba, labels=BINARY_LABELS),
        **prediction_health(y_pred),
    }
    try:
        result["roc_auc"] = roc_auc_score(y_true, y_proba_up)
    except ValueError:
        result["roc_auc"] = np.nan

    for true_label, pred_label in product(BINARY_LABELS, BINARY_LABELS):
        result[f"cm_{BINARY_NAMES[true_label]}_{BINARY_NAMES[pred_label]}"] = int(
            matrix[true_label, pred_label]
        )
    for index, label in enumerate(BINARY_LABELS):
        result[f"precision_{BINARY_NAMES[label]}"] = precision[index]
        result[f"recall_{BINARY_NAMES[label]}"] = recall[index]

    returns = pd.DataFrame({"prediction": y_pred, RETURN_COLUMN: realized_return.to_numpy()})
    mean_returns = returns.groupby("prediction")[RETURN_COLUMN].mean()
    result["mean_future_5d_return_pred_down"] = mean_returns.get(0, np.nan)
    result["mean_future_5d_return_pred_up"] = mean_returns.get(1, np.nan)
    result["long_short_spread_pred_up_minus_down"] = mean_returns.get(1, np.nan) - mean_returns.get(0, np.nan)
    return result


def selected_metric_key(row: dict[str, object]) -> tuple[bool, float, float, float]:
    return (
        bool(row["passed_prediction_balance"]),
        float(row["f1_macro"]),
        float(row["balanced_accuracy"]),
        -float(row["log_loss"]),
    )


def run_search(input_dir: Path = PROCESSED_CORE_DIR, results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_run_path()
    metric_rows = []
    selected_rows = []

    for path in dataset_files(input_dir):
        core_data = add_binary_target(load_core_dataset(path))
        name = path.name.removesuffix("_core_features.csv")
        ticker = str(core_data["ticker"].iloc[0])

        for recipe in FEATURE_RECIPES:
            frame = add_binary_target(build_recipe_frame(core_data, raw_path, recipe))
            features = recipe_feature_columns(frame, core_data, recipe)
            x_train, y_train, _ = prepare_split(frame, features, "train")
            x_validation, y_validation, validation_return = prepare_split(frame, features, "validation")
            if x_train.empty or x_validation.empty:
                continue
            if set(y_train) != set(BINARY_LABELS):
                continue

            recipe_metrics = []
            fitted_models = {}
            for params in PARAM_GRID:
                model = build_model(params)
                if params.use_sample_weight:
                    model.fit(x_train, y_train, sample_weight=sample_weights(y_train))
                else:
                    model.fit(x_train, y_train)
                fitted_models[params.name] = model
                y_proba_up = model.predict_proba(x_validation)[:, 1]
                y_pred = (y_proba_up >= 0.5).astype(int)
                validation_metrics = evaluate_predictions(
                    name=name,
                    ticker=ticker,
                    recipe=recipe,
                    params=params,
                    split_name="validation",
                    y_true=y_validation,
                    y_pred=y_pred,
                    y_proba_up=y_proba_up,
                    realized_return=validation_return,
                )
                recipe_metrics.append(validation_metrics)
                metric_rows.append(validation_metrics)

            best_metric = sorted(recipe_metrics, key=selected_metric_key, reverse=True)[0]
            best_params = next(params for params in PARAM_GRID if params.name == best_metric["params"])
            best_model = fitted_models[best_params.name]
            selected_rows.append(
                {
                    "name": name,
                    "ticker": ticker,
                    "recipe": recipe.name,
                    "selected_params": best_params.name,
                    "selected_by": "validation_f1_macro_with_prediction_balance",
                    "validation_f1_macro": best_metric["f1_macro"],
                    "validation_balanced_accuracy": best_metric["balanced_accuracy"],
                    "validation_accuracy": best_metric["accuracy"],
                    "validation_log_loss": best_metric["log_loss"],
                    "validation_passed_prediction_balance": best_metric["passed_prediction_balance"],
                    "feature_count": len(features),
                }
            )

            for split_name in ["test", "recent_regime"]:
                x_eval, y_eval, realized_return = prepare_split(frame, features, split_name)
                if x_eval.empty:
                    continue
                y_proba_up = best_model.predict_proba(x_eval)[:, 1]
                metric_rows.append(
                    evaluate_predictions(
                        name=name,
                        ticker=ticker,
                        recipe=recipe,
                        params=best_params,
                        split_name=split_name,
                        y_true=y_eval,
                        y_pred=(y_proba_up >= 0.5).astype(int),
                        y_proba_up=y_proba_up,
                        realized_return=realized_return,
                    )
                )

    metrics = pd.DataFrame(metric_rows)
    selected = pd.DataFrame(selected_rows)
    metrics.to_csv(results_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(results_dir / "best_by_recipe.csv", index=False, encoding="utf-8-sig")
    if metrics.empty or selected.empty:
        return

    selected_by_ticker = (
        selected.sort_values(
            [
                "ticker",
                "validation_passed_prediction_balance",
                "validation_f1_macro",
                "validation_balanced_accuracy",
                "validation_log_loss",
            ],
            ascending=[True, False, False, False, True],
        )
        .groupby("ticker", as_index=False)
        .head(1)
    )
    selected_by_ticker.to_csv(results_dir / "selected_by_ticker.csv", index=False, encoding="utf-8-sig")

    summary = (
        metrics[metrics["split"].isin(["test", "recent_regime"])]
        .merge(
            selected[["name", "ticker", "recipe", "selected_params"]],
            left_on=["name", "ticker", "recipe", "params"],
            right_on=["name", "ticker", "recipe", "selected_params"],
            how="inner",
        )
        .pivot_table(
            index=["name", "ticker", "recipe", "params"],
            columns="split",
            values=[
                "accuracy",
                "balanced_accuracy",
                "f1_macro",
                "roc_auc",
                "long_short_spread_pred_up_minus_down",
                "passed_prediction_balance",
            ],
        )
        .reset_index()
    )
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary.to_csv(results_dir / "search_summary.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search binary up/down XGBoost models without neutral rows.")
    parser.add_argument("--input-dir", default=str(PROCESSED_CORE_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_search(Path(args.input_dir), Path(args.results_dir))
