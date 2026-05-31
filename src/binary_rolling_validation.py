from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from src.binary_xgboost import (
    BINARY_LABELS,
    BINARY_TARGET_COLUMN,
    RESULTS_DIR as BINARY_RESULTS_DIR,
    add_binary_target,
    build_model,
    sample_weights,
)
from src.preprocess import LABEL_HORIZON_DAYS, PROJECT_ROOT
from src.proposed_xgboost import RETURN_COLUMN, raw_run_path
from src.technical_extended_search import (
    PROCESSED_CORE_DIR,
    dataset_files,
    load_core_dataset,
    recipe_feature_columns,
    build_recipe_frame,
)
from src.binary_shap_analysis import DEFAULT_RECIPE, params_by_name, read_selection, recipe_by_name

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "binary_rolling_validation"
N_SPLITS = 4
INITIAL_TRAIN_FRACTION = 0.50


def prepare_model_frame(path: Path, recipe_name: str) -> tuple[str, str, pd.DataFrame, list[str]]:
    core_data = add_binary_target(load_core_dataset(path))
    name = path.name.removesuffix("_core_features.csv")
    ticker = str(core_data["ticker"].iloc[0])
    frame = add_binary_target(build_recipe_frame(core_data, raw_run_path(), recipe_by_name(recipe_name)))
    features = recipe_feature_columns(frame, core_data, recipe_by_name(recipe_name))
    frame[features] = frame[features].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=features + [BINARY_TARGET_COLUMN, RETURN_COLUMN]).copy()
    frame = frame[frame["period_bucket"] == "model_2018_2024"].sort_values("date").reset_index(drop=True)
    return name, ticker, frame, features


def fold_boundaries(row_count: int) -> list[tuple[int, int, int]]:
    initial_train_end = int(row_count * INITIAL_TRAIN_FRACTION)
    remaining = row_count - initial_train_end
    fold_size = remaining // N_SPLITS
    boundaries = []
    for fold in range(N_SPLITS):
        train_end = initial_train_end + fold * fold_size
        eval_start = min(train_end + LABEL_HORIZON_DAYS, row_count)
        eval_end = initial_train_end + (fold + 1) * fold_size if fold < N_SPLITS - 1 else row_count
        if eval_end > eval_start:
            boundaries.append((train_end, eval_start, eval_end))
    return boundaries


def prediction_balance(y_pred: np.ndarray) -> tuple[float, float, bool]:
    counts = pd.Series(y_pred).value_counts(normalize=True).to_dict()
    down_share = float(counts.get(0, 0.0))
    up_share = float(counts.get(1, 0.0))
    return down_share, up_share, min(down_share, up_share) >= 0.10


def evaluate_fold(
    *,
    name: str,
    ticker: str,
    recipe_name: str,
    params_name: str,
    fold: int,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
) -> dict[str, object]:
    x_train = train[features].astype(float)
    y_train = train[BINARY_TARGET_COLUMN].astype(int)
    x_eval = evaluation[features].astype(float)
    y_eval = evaluation[BINARY_TARGET_COLUMN].astype(int)
    if set(y_train) != set(BINARY_LABELS) or set(y_eval) != set(BINARY_LABELS):
        raise ValueError("rolling fold lacks both binary classes")

    params = params_by_name(params_name)
    model = build_model(params)
    if params.use_sample_weight:
        model.fit(x_train, y_train, sample_weight=sample_weights(y_train))
    else:
        model.fit(x_train, y_train)

    proba_up = model.predict_proba(x_eval)[:, 1]
    pred = (proba_up >= 0.5).astype(int)
    down_share, up_share, passed_balance = prediction_balance(pred)
    returns = pd.DataFrame({"prediction": pred, RETURN_COLUMN: evaluation[RETURN_COLUMN].to_numpy()})
    mean_returns = returns.groupby("prediction")[RETURN_COLUMN].mean()
    matrix = confusion_matrix(y_eval, pred, labels=BINARY_LABELS)
    return {
        "name": name,
        "ticker": ticker,
        "recipe": recipe_name,
        "params": params_name,
        "fold": fold,
        "train_start": train["date"].min().date().isoformat(),
        "train_end": train["date"].max().date().isoformat(),
        "eval_start": evaluation["date"].min().date().isoformat(),
        "eval_end": evaluation["date"].max().date().isoformat(),
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "accuracy": accuracy_score(y_eval, pred),
        "balanced_accuracy": balanced_accuracy_score(y_eval, pred),
        "f1_macro": f1_score(y_eval, pred, labels=BINARY_LABELS, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_eval, proba_up),
        "pred_down_share": down_share,
        "pred_up_share": up_share,
        "passed_prediction_balance": passed_balance,
        "cm_down_down": int(matrix[0, 0]),
        "cm_down_up": int(matrix[0, 1]),
        "cm_up_down": int(matrix[1, 0]),
        "cm_up_up": int(matrix[1, 1]),
        "mean_future_5d_return_pred_down": mean_returns.get(0, np.nan),
        "mean_future_5d_return_pred_up": mean_returns.get(1, np.nan),
        "long_short_spread_pred_up_minus_down": mean_returns.get(1, np.nan) - mean_returns.get(0, np.nan),
    }


def run_rolling_validation(
    input_dir: Path = PROCESSED_CORE_DIR,
    binary_results_dir: Path = BINARY_RESULTS_DIR,
    results_dir: Path = RESULTS_DIR,
    recipe_name: str = DEFAULT_RECIPE,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    selected = read_selection(binary_results_dir, recipe_name)
    params_by_ticker = {str(row["ticker"]): str(row["selected_params"]) for _, row in selected.iterrows()}
    rows = []
    for path in dataset_files(input_dir):
        name, ticker, frame, features = prepare_model_frame(path, recipe_name)
        if ticker not in params_by_ticker:
            continue
        for fold, (train_end, eval_start, eval_end) in enumerate(fold_boundaries(len(frame)), start=1):
            train = frame.iloc[:train_end].copy()
            evaluation = frame.iloc[eval_start:eval_end].copy()
            if train.empty or evaluation.empty:
                continue
            if set(train[BINARY_TARGET_COLUMN].astype(int)) != set(BINARY_LABELS):
                continue
            if set(evaluation[BINARY_TARGET_COLUMN].astype(int)) != set(BINARY_LABELS):
                continue
            rows.append(
                evaluate_fold(
                    name=name,
                    ticker=ticker,
                    recipe_name=recipe_name,
                    params_name=params_by_ticker[ticker],
                    fold=fold,
                    train=train,
                    evaluation=evaluation,
                    features=features,
                )
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(results_dir / "rolling_metrics.csv", index=False, encoding="utf-8-sig")
    if metrics.empty:
        return
    summary = (
        metrics.groupby(["name", "ticker", "recipe", "params"], as_index=False)
        .agg(
            folds=("fold", "count"),
            mean_f1_macro=("f1_macro", "mean"),
            std_f1_macro=("f1_macro", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            balance_pass_rate=("passed_prediction_balance", "mean"),
            mean_long_short_spread=("long_short_spread_pred_up_minus_down", "mean"),
        )
    )
    summary.to_csv(results_dir / "rolling_summary.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanding rolling validation for binary technical-only models.")
    parser.add_argument("--input-dir", default=str(PROCESSED_CORE_DIR))
    parser.add_argument("--binary-results-dir", default=str(BINARY_RESULTS_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--recipe", default=DEFAULT_RECIPE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_rolling_validation(
        input_dir=Path(args.input_dir),
        binary_results_dir=Path(args.binary_results_dir),
        results_dir=Path(args.results_dir),
        recipe_name=args.recipe,
    )
