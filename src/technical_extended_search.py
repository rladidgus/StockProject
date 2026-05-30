from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from src.preprocess import METADATA_ROOT, PROJECT_ROOT, TICKER_SPECS, read_current_raw_manifest
from src.proposed_xgboost import (
    CLASS_LABELS,
    CLASS_NAMES,
    LAGS,
    RETURN_COLUMN,
    TARGET_COLUMN,
    SeriesSpec,
    build_experiment_frame,
    core_feature_columns,
    extended_feature_columns,
    raw_run_path,
    sample_weights,
)
from src.technical_baseline_xgboost import add_technical_features, load_price_frame, technical_feature_columns

PROCESSED_CORE_DIR = PROJECT_ROOT / "data" / "processed" / "core"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "technical_extended_search"
RANDOM_STATE = 42


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    macro_series: tuple[SeriesSpec, ...] = ()
    use_domestic_alpha: bool = False


@dataclass(frozen=True)
class FeatureRecipe:
    name: str
    groups: tuple[FeatureGroup, ...]


@dataclass(frozen=True)
class ModelParams:
    name: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    use_sample_weight: bool = True


GROUPS = {
    "rates": FeatureGroup(
        "rates",
        macro_series=(
            SeriesSpec("us_10y_treasury_fred"),
            SeriesSpec("us_2y_treasury_fred"),
            SeriesSpec("us_10y_2y_spread"),
        ),
    ),
    "dollar": FeatureGroup("dollar", macro_series=(SeriesSpec("broad_dollar_index"),)),
    "semi_etf": FeatureGroup("semi_etf", macro_series=(SeriesSpec("smh"), SeriesSpec("soxx"))),
    "monthly_industry": FeatureGroup(
        "monthly_industry",
        macro_series=(
            SeriesSpec("semiconductor_electronic_component_industrial_production", "monthly"),
            SeriesSpec("semiconductor_related_device_ppi", "monthly"),
            SeriesSpec("computers_electronic_products_new_orders", "monthly"),
        ),
    ),
    "domestic_alpha": FeatureGroup("domestic_alpha", use_domestic_alpha=True),
}


FEATURE_RECIPES = [
    FeatureRecipe("technical_only", ()),
    FeatureRecipe("technical_core", ()),
    FeatureRecipe("technical_rates", (GROUPS["rates"],)),
    FeatureRecipe("technical_dollar", (GROUPS["dollar"],)),
    FeatureRecipe("technical_semi_etf", (GROUPS["semi_etf"],)),
    FeatureRecipe("technical_monthly_industry", (GROUPS["monthly_industry"],)),
    FeatureRecipe("technical_domestic_alpha", (GROUPS["domestic_alpha"],)),
    FeatureRecipe("technical_rates_dollar", (GROUPS["rates"], GROUPS["dollar"])),
    FeatureRecipe("technical_rates_semi_etf", (GROUPS["rates"], GROUPS["semi_etf"])),
    FeatureRecipe("technical_dollar_semi_etf", (GROUPS["dollar"], GROUPS["semi_etf"])),
    FeatureRecipe("technical_alpha_semi_etf", (GROUPS["domestic_alpha"], GROUPS["semi_etf"])),
    FeatureRecipe("technical_alpha_rates", (GROUPS["domestic_alpha"], GROUPS["rates"])),
    FeatureRecipe(
        "technical_daily_extended",
        (GROUPS["rates"], GROUPS["dollar"], GROUPS["semi_etf"]),
    ),
    FeatureRecipe(
        "technical_all_guarded",
        (
            GROUPS["rates"],
            GROUPS["dollar"],
            GROUPS["semi_etf"],
            GROUPS["monthly_industry"],
            GROUPS["domestic_alpha"],
        ),
    ),
]


PARAM_GRID = [
    ModelParams("baseline_depth3", 300, 3, 0.05, 1.0, 0.85, 0.85, 1.0),
    ModelParams("shallow_regularized", 500, 2, 0.03, 3.0, 0.80, 0.80, 2.0),
    ModelParams("depth3_regularized", 500, 3, 0.03, 3.0, 0.80, 0.80, 3.0),
    ModelParams("depth4_slow", 400, 4, 0.03, 2.0, 0.80, 0.80, 2.0),
    ModelParams("depth2_fast", 250, 2, 0.08, 2.0, 0.90, 0.90, 1.5),
    ModelParams("depth3_conservative", 700, 3, 0.02, 5.0, 0.75, 0.75, 5.0),
    ModelParams("baseline_depth3_unweighted", 300, 3, 0.05, 1.0, 0.85, 0.85, 1.0, False),
    ModelParams("depth2_fast_unweighted", 250, 2, 0.08, 2.0, 0.90, 0.90, 1.5, False),
    ModelParams("depth3_regularized_unweighted", 500, 3, 0.03, 3.0, 0.80, 0.80, 3.0, False),
    ModelParams("depth4_slow_unweighted", 400, 4, 0.03, 2.0, 0.80, 0.80, 2.0, False),
]


class RecipeExperiment:
    def __init__(self, recipe: FeatureRecipe) -> None:
        self.name = recipe.name
        self.feature_family = "+".join(["technical", *[group.name for group in recipe.groups]])
        self.macro_series = tuple(series for group in recipe.groups for series in group.macro_series)
        self.use_domestic_alpha = any(group.use_domestic_alpha for group in recipe.groups)


def dataset_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*_core_features.csv"))


def load_core_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"], dtype={"ticker": "string"})


def build_technical_frame(data: pd.DataFrame) -> pd.DataFrame:
    manifest_row = read_current_raw_manifest(METADATA_ROOT / "current_raw_manifest.csv")
    raw_path = PROJECT_ROOT / str(manifest_row["raw_run_path"])
    ticker = str(data["ticker"].iloc[0])
    prices = add_technical_features(load_price_frame(raw_path, ticker))
    technical = prices[["date", *technical_feature_columns(prices.columns)]]
    return data.merge(technical, on="date", how="left")


def build_recipe_frame(data: pd.DataFrame, raw_path: Path, recipe: FeatureRecipe) -> pd.DataFrame:
    frame = build_technical_frame(data)
    if recipe.groups:
        frame = build_experiment_frame(frame, raw_path, RecipeExperiment(recipe))
    return frame


def recipe_feature_columns(frame: pd.DataFrame, core_data: pd.DataFrame, recipe: FeatureRecipe) -> list[str]:
    technical = technical_feature_columns(frame.columns)
    if recipe.name == "technical_only":
        return technical
    core = core_feature_columns(core_data.columns)
    extended = extended_feature_columns(frame.columns)
    return technical + core + extended


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


def build_model(params: ModelParams) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_LABELS),
        eval_metric="mlogloss",
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


def evaluate_predictions(
    *,
    name: str,
    ticker: str,
    recipe: FeatureRecipe,
    params: ModelParams,
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
        "recipe": recipe.name,
        "feature_family": RecipeExperiment(recipe).feature_family,
        "params": params.name,
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
    for true_label, pred_label in product(CLASS_LABELS, CLASS_LABELS):
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


def run_search(input_dir: Path = PROCESSED_CORE_DIR, results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_run_path()
    metric_rows = []
    feature_rows = []
    best_rows = []

    for path in dataset_files(input_dir):
        core_data = load_core_dataset(path)
        name = path.name.removesuffix("_core_features.csv")
        ticker = str(core_data["ticker"].iloc[0])

        for recipe in FEATURE_RECIPES:
            frame = build_recipe_frame(core_data, raw_path, recipe)
            features = recipe_feature_columns(frame, core_data, recipe)
            x_train, y_train, _ = prepare_split(frame, features, "train")
            x_validation, y_validation, validation_return = prepare_split(frame, features, "validation")
            if x_train.empty or x_validation.empty:
                continue
            if set(y_train) != set(CLASS_LABELS):
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
                y_pred = model.predict(x_validation)
                y_proba = model.predict_proba(x_validation)
                validation_metrics = evaluate_predictions(
                    name=name,
                    ticker=ticker,
                    recipe=recipe,
                    params=params,
                    split_name="validation",
                    y_true=y_validation,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    realized_return=validation_return,
                )
                recipe_metrics.append(validation_metrics)
                metric_rows.append(validation_metrics)

            best_metric = sorted(
                recipe_metrics,
                key=lambda row: (row["f1_macro"], row["balanced_accuracy"], -row["log_loss"]),
                reverse=True,
            )[0]
            best_params = next(params for params in PARAM_GRID if params.name == best_metric["params"])
            best_model = fitted_models[best_params.name]

            feature_rows.append(
                {
                    "name": name,
                    "ticker": ticker,
                    "recipe": recipe.name,
                    "feature_family": RecipeExperiment(recipe).feature_family,
                    "params": best_params.name,
                    "feature_count": len(features),
                    "technical_feature_count": len(technical_feature_columns(frame.columns)),
                    "core_feature_count": 0 if recipe.name == "technical_only" else len(core_feature_columns(core_data.columns)),
                    "extended_feature_count": 0 if recipe.name == "technical_only" else len(extended_feature_columns(frame.columns)),
                    "features": ",".join(features),
                }
            )
            best_rows.append(
                {
                    "name": name,
                    "ticker": ticker,
                    "recipe": recipe.name,
                    "feature_family": RecipeExperiment(recipe).feature_family,
                    "selected_params": best_params.name,
                    "selected_by": "validation_f1_macro",
                    "validation_f1_macro": best_metric["f1_macro"],
                    "validation_balanced_accuracy": best_metric["balanced_accuracy"],
                    "validation_accuracy": best_metric["accuracy"],
                    "validation_log_loss": best_metric["log_loss"],
                }
            )

            for split_name in ["test", "recent_regime"]:
                x_eval, y_eval, realized_return = prepare_split(frame, features, split_name)
                if x_eval.empty:
                    continue
                metric_rows.append(
                    evaluate_predictions(
                        name=name,
                        ticker=ticker,
                        recipe=recipe,
                        params=best_params,
                        split_name=split_name,
                        y_true=y_eval,
                        y_pred=best_model.predict(x_eval),
                        y_proba=best_model.predict_proba(x_eval),
                        realized_return=realized_return,
                    )
                )

    metrics = pd.DataFrame(metric_rows)
    best = pd.DataFrame(best_rows)
    features = pd.DataFrame(feature_rows)
    metrics.to_csv(results_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    best.to_csv(results_dir / "best_by_recipe.csv", index=False, encoding="utf-8-sig")
    features.to_csv(results_dir / "feature_sets.csv", index=False, encoding="utf-8-sig")

    if not metrics.empty:
        selected = (
            best.sort_values(["ticker", "validation_f1_macro", "validation_balanced_accuracy"], ascending=[True, False, False])
            .groupby("ticker", as_index=False)
            .head(1)
        )
        selected.to_csv(results_dir / "selected_by_ticker.csv", index=False, encoding="utf-8-sig")
        summary = (
            metrics[metrics["split"].isin(["test", "recent_regime"])]
            .merge(
                best[["name", "ticker", "recipe", "selected_params"]],
                left_on=["name", "ticker", "recipe", "params"],
                right_on=["name", "ticker", "recipe", "selected_params"],
                how="inner",
            )
            .pivot_table(
                index=["name", "ticker", "recipe", "feature_family", "params"],
                columns="split",
                values=["accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr_weighted"],
            )
            .reset_index()
        )
        summary.columns = [
            "_".join(column).strip("_") if isinstance(column, tuple) else column
            for column in summary.columns
        ]
        summary.to_csv(results_dir / "search_summary.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search technical + extended XGBoost feature recipes.")
    parser.add_argument("--input-dir", default=str(PROCESSED_CORE_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_search(Path(args.input_dir), Path(args.results_dir))
