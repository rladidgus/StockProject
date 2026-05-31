from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import DMatrix

from src.binary_xgboost import (
    BINARY_LABELS,
    BINARY_NAMES,
    BINARY_TARGET_COLUMN,
    RESULTS_DIR as BINARY_RESULTS_DIR,
    add_binary_target,
    prepare_split,
    sample_weights,
)
from src.preprocess import PROJECT_ROOT
from src.proposed_xgboost import RETURN_COLUMN, raw_run_path
from src.technical_extended_search import (
    FEATURE_RECIPES,
    PARAM_GRID,
    PROCESSED_CORE_DIR,
    build_recipe_frame,
    dataset_files,
    load_core_dataset,
    recipe_feature_columns,
)
from src.binary_xgboost import build_model

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "binary_shap_analysis"
DEFAULT_RECIPE = "technical_only"


def recipe_by_name(name: str):
    for recipe in FEATURE_RECIPES:
        if recipe.name == name:
            return recipe
    raise ValueError(f"unknown recipe: {name}")


def params_by_name(name: str):
    for params in PARAM_GRID:
        if params.name == name:
            return params
    raise ValueError(f"unknown params: {name}")


def read_selection(binary_results_dir: Path, recipe_name: str) -> pd.DataFrame:
    best = pd.read_csv(binary_results_dir / "best_by_recipe.csv", encoding="utf-8-sig")
    selected = best[best["recipe"] == recipe_name].copy()
    if selected.empty:
        raise ValueError(f"no binary selection rows for recipe={recipe_name}")
    selected = selected.sort_values(
        ["ticker", "validation_passed_prediction_balance", "validation_f1_macro", "validation_balanced_accuracy"],
        ascending=[True, False, False, False],
    )
    return selected.groupby("ticker", as_index=False).head(1)


def clear_previous_outputs(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["*.csv", "*.png"]:
        for path in results_dir.glob(pattern):
            path.unlink()


def contribution_matrix(model, x: pd.DataFrame) -> np.ndarray:
    raw = model.get_booster().predict(DMatrix(x), pred_contribs=True)
    if raw.ndim != 2 or raw.shape[1] != len(x.columns) + 1:
        raise ValueError(f"unexpected binary SHAP contribution shape: {raw.shape}")
    return raw


def class_contributions(contributions: np.ndarray, class_label: int) -> np.ndarray:
    if class_label == 1:
        return contributions
    if class_label == 0:
        return -contributions
    raise ValueError(f"unsupported binary class label: {class_label}")


def global_importance_rows(
    *,
    name: str,
    ticker: str,
    recipe: str,
    params: str,
    split_name: str,
    feature_columns: list[str],
    contributions: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for class_label in BINARY_LABELS:
        shap_values = class_contributions(contributions, class_label)[:, :-1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_signed = shap_values.mean(axis=0)
        for feature, abs_value, signed_value in zip(feature_columns, mean_abs, mean_signed):
            rows.append(
                {
                    "name": name,
                    "ticker": ticker,
                    "recipe": recipe,
                    "params": params,
                    "split": split_name,
                    "class_label": class_label,
                    "class_name": BINARY_NAMES[class_label],
                    "feature": feature,
                    "mean_abs_shap": float(abs_value),
                    "mean_shap": float(signed_value),
                }
            )
    overall = np.abs(contributions[:, :-1]).mean(axis=0)
    for feature, abs_value in zip(feature_columns, overall):
        rows.append(
            {
                "name": name,
                "ticker": ticker,
                "recipe": recipe,
                "params": params,
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
    recipe: str,
    params: str,
    split_name: str,
    feature_columns: list[str],
    predictions: np.ndarray,
    probabilities_up: np.ndarray,
    contributions: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    selection_rows = []
    working = data[["date", BINARY_TARGET_COLUMN, RETURN_COLUMN]].copy().reset_index(drop=True)
    working["prediction"] = predictions
    working["probability_down"] = 1 - probabilities_up
    working["probability_up"] = probabilities_up

    for class_label in BINARY_LABELS:
        class_name = BINARY_NAMES[class_label]
        probability_column = f"probability_{class_name}"
        candidates = working[working["prediction"] == class_label]
        fallback_used = candidates.empty
        if fallback_used:
            candidates = working.sort_values(probability_column, ascending=False).head(1)
        selected_index = int(candidates.sort_values(probability_column, ascending=False).index[0])
        shap_values = class_contributions(contributions, class_label)[selected_index, :-1]
        bias = float(class_contributions(contributions, class_label)[selected_index, -1])
        selection_rows.append(
            {
                "name": name,
                "ticker": ticker,
                "recipe": recipe,
                "params": params,
                "split": split_name,
                "requested_class": class_name,
                "date": working.loc[selected_index, "date"],
                "true_label": int(working.loc[selected_index, BINARY_TARGET_COLUMN]),
                "predicted_label": int(working.loc[selected_index, "prediction"]),
                "requested_class_probability": float(working.loc[selected_index, probability_column]),
                "fallback_used": fallback_used,
            }
        )
        top_indices = np.argsort(np.abs(shap_values))[::-1][:12]
        for rank, feature_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "name": name,
                    "ticker": ticker,
                    "recipe": recipe,
                    "params": params,
                    "split": split_name,
                    "date": working.loc[selected_index, "date"],
                    "true_label": int(working.loc[selected_index, BINARY_TARGET_COLUMN]),
                    "predicted_label": int(working.loc[selected_index, "prediction"]),
                    "explained_class_name": class_name,
                    "prediction_probability": float(working.loc[selected_index, probability_column]),
                    "future_5d_return": float(working.loc[selected_index, RETURN_COLUMN]),
                    "base_value": bias,
                    "rank": rank,
                    "feature": feature_columns[feature_index],
                    "shap_value": float(shap_values[feature_index]),
                    "feature_value": float(data.iloc[selected_index][feature_columns[feature_index]]),
                }
            )
    return rows, selection_rows


def plot_global_importance(data: pd.DataFrame, output_dir: Path) -> None:
    overall = data[(data["split"] == "test") & (data["class_name"] == "overall")]
    for (name, ticker, recipe), group in overall.groupby(["name", "ticker", "recipe"]):
        top = group.sort_values("mean_abs_shap", ascending=False).head(15).sort_values("mean_abs_shap")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["feature"], top["mean_abs_shap"], color="#356859")
        ax.set_title(f"{name} {recipe} binary global SHAP")
        ax.set_xlabel("Mean absolute SHAP contribution")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}_{ticker}_{recipe}_binary_global_bar.png", dpi=160)
        plt.close(fig)


def plot_local_cases(data: pd.DataFrame, output_dir: Path) -> None:
    for (name, ticker, recipe, date, explained_class), group in data.groupby(
        ["name", "ticker", "recipe", "date", "explained_class_name"]
    ):
        top = group.sort_values("rank", ascending=False)
        colors = np.where(top["shap_value"] >= 0, "#b23a48", "#2f6f9f")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["feature"], top["shap_value"], color=colors)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"{name} {explained_class} binary explanation on {pd.to_datetime(date).date()}")
        ax.set_xlabel("SHAP contribution to class logit")
        ax.set_ylabel("")
        fig.tight_layout()
        safe_date = pd.to_datetime(date).date().isoformat()
        fig.savefig(output_dir / f"{name}_{ticker}_{recipe}_{safe_date}_{explained_class}_binary_local.png", dpi=160)
        plt.close(fig)


def run_binary_shap_analysis(
    input_dir: Path = PROCESSED_CORE_DIR,
    binary_results_dir: Path = BINARY_RESULTS_DIR,
    results_dir: Path = RESULTS_DIR,
    recipe_name: str = DEFAULT_RECIPE,
) -> None:
    clear_previous_outputs(results_dir)
    selected = read_selection(binary_results_dir, recipe_name)
    raw_path = raw_run_path()
    recipe = recipe_by_name(recipe_name)
    selected_by_ticker = {str(row["ticker"]): row for _, row in selected.iterrows()}

    selected_rows = []
    global_rows = []
    local_rows = []
    local_selection_rows = []

    for path in dataset_files(input_dir):
        core_data = add_binary_target(load_core_dataset(path))
        name = path.name.removesuffix("_core_features.csv")
        ticker = str(core_data["ticker"].iloc[0])
        if ticker not in selected_by_ticker:
            continue

        selection = selected_by_ticker[ticker]
        params = params_by_name(str(selection["selected_params"]))
        frame = add_binary_target(build_recipe_frame(core_data, raw_path, recipe))
        feature_columns = recipe_feature_columns(frame, core_data, recipe)
        x_train, y_train, _ = prepare_split(frame, feature_columns, "train")
        model = build_model(params)
        if params.use_sample_weight:
            model.fit(x_train, y_train, sample_weight=sample_weights(y_train))
        else:
            model.fit(x_train, y_train)

        selected_rows.append(
            {
                "name": name,
                "ticker": ticker,
                "recipe": recipe_name,
                "params": params.name,
                "selected_by": "binary_validation_f1_macro_with_prediction_balance",
                "selection_metric": float(selection["validation_f1_macro"]),
                "feature_count": len(feature_columns),
            }
        )

        for split_name in ["test", "recent_regime"]:
            x_eval, y_eval, realized_return = prepare_split(frame, feature_columns, split_name)
            if x_eval.empty:
                continue
            evaluation = frame.loc[x_eval.index].copy()
            probabilities_up = model.predict_proba(x_eval)[:, 1]
            predictions = (probabilities_up >= 0.5).astype(int)
            contributions = contribution_matrix(model, x_eval)
            global_rows.extend(
                global_importance_rows(
                    name=name,
                    ticker=ticker,
                    recipe=recipe_name,
                    params=params.name,
                    split_name=split_name,
                    feature_columns=feature_columns,
                    contributions=contributions,
                )
            )
            if split_name == "test":
                case_rows, selection_case_rows = local_case_rows(
                    data=evaluation,
                    name=name,
                    ticker=ticker,
                    recipe=recipe_name,
                    params=params.name,
                    split_name=split_name,
                    feature_columns=feature_columns,
                    predictions=predictions,
                    probabilities_up=probabilities_up,
                    contributions=contributions,
                )
                local_rows.extend(case_rows)
                local_selection_rows.extend(selection_case_rows)

    selected_frame = pd.DataFrame(selected_rows)
    global_frame = pd.DataFrame(global_rows)
    local_frame = pd.DataFrame(local_rows)
    local_selection_frame = pd.DataFrame(local_selection_rows)
    selected_frame.to_csv(results_dir / "selected_models.csv", index=False, encoding="utf-8-sig")
    global_frame.to_csv(results_dir / "global_shap_importance.csv", index=False, encoding="utf-8-sig")
    local_frame.to_csv(results_dir / "local_shap_cases.csv", index=False, encoding="utf-8-sig")
    local_selection_frame.to_csv(results_dir / "local_case_selection.csv", index=False, encoding="utf-8-sig")
    plot_global_importance(global_frame, results_dir)
    plot_local_cases(local_frame, results_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate binary technical-only XGBoost contribution artifacts.")
    parser.add_argument("--input-dir", default=str(PROCESSED_CORE_DIR))
    parser.add_argument("--binary-results-dir", default=str(BINARY_RESULTS_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--recipe", default=DEFAULT_RECIPE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_binary_shap_analysis(
        input_dir=Path(args.input_dir),
        binary_results_dir=Path(args.binary_results_dir),
        results_dir=Path(args.results_dir),
        recipe_name=args.recipe,
    )
