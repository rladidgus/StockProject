from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_CORE_DIR = PROJECT_ROOT / "data" / "processed" / "core"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "model_comparison"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "model_comparison"

LAGS = [1, 3, 5]
TARGET_COLUMN = "target_direction_1pct"
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    scale: bool


MODEL_SPECS = [
    ModelSpec(
        "svm",
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
        True,
    ),
    ModelSpec(
        "decision_tree",
        DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=RANDOM_STATE),
        False,
    ),
    ModelSpec(
        "random_forest",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        False,
    ),
    ModelSpec(
        "ann",
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=0.001,
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True,
        ),
        True,
    ),
    ModelSpec(
        "logistic_regression",
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        True,
    ),
]


def dataset_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*_core_features.csv"))


def feature_columns_for_lag(columns: pd.Index, lag: int) -> list[str]:
    suffix = f"_t_minus_{lag}"
    return [
        column
        for column in columns
        if column.endswith(suffix)
        and not column.endswith("_source_date")
        and not column.endswith("_known_date")
    ]


def model_pipeline(spec: ModelSpec) -> object:
    estimator = clone(spec.estimator)
    if not spec.scale:
        return estimator
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def prepare_split(
    data: pd.DataFrame,
    feature_columns: list[str],
    split_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    subset = data[data["time_split"] == split_name].copy()
    subset = subset.dropna(subset=feature_columns + [TARGET_COLUMN])
    x = subset[feature_columns].astype(float)
    y = subset[TARGET_COLUMN].astype(int)
    return x, y


def evaluate_predictions(
    *,
    name: str,
    ticker: str,
    lag: int,
    model_name: str,
    split_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, object]:
    labels = [0, 1]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = matrix.ravel()
    return {
        "name": name,
        "ticker": ticker,
        "lag_days": lag,
        "model": model_name,
        "split": split_name,
        "rows": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def shap_importance(
    model: object,
    x_background: pd.DataFrame,
    x_explain: pd.DataFrame,
) -> pd.Series:
    if x_background.empty or x_explain.empty:
        return pd.Series(dtype=float)

    background = x_background.sample(min(len(x_background), 30), random_state=RANDOM_STATE)
    explain = x_explain.sample(min(len(x_explain), 40), random_state=RANDOM_STATE)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.Explainer(model.predict_proba, background)
        explanation = explainer(explain)

    values = explanation.values
    if values.ndim == 3:
        values = values[:, :, 1]
    importances = np.abs(values).mean(axis=0)
    return pd.Series(importances, index=explain.columns).sort_values(ascending=False)


def save_shap_bar(
    importances: pd.Series,
    *,
    name: str,
    lag: int,
    model_name: str,
    output_dir: Path,
) -> None:
    if importances.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    importances.sort_values().plot(kind="barh", color="#4C78A8")
    plt.title(f"{name} lag {lag} {model_name} SHAP importance")
    plt.xlabel("mean(|SHAP value|)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_lag{lag}_{model_name}_shap_importance.png", dpi=160)
    plt.close()


def run_model_comparison(
    input_dir: Path = PROCESSED_CORE_DIR,
    results_dir: Path = RESULTS_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    shap_rows = []

    for path in dataset_files(input_dir):
        data = pd.read_csv(path, encoding="utf-8-sig", dtype={"ticker": "string"})
        name = path.name.removesuffix("_core_features.csv")
        ticker = str(data["ticker"].iloc[0])

        for lag in LAGS:
            feature_columns = feature_columns_for_lag(data.columns, lag)
            if not feature_columns:
                raise ValueError(f"no lag {lag} features found in {path}")

            x_train, y_train = prepare_split(data, feature_columns, "train")
            if x_train.empty:
                raise ValueError(f"empty train split for {name} lag {lag}")

            for spec in MODEL_SPECS:
                model = model_pipeline(spec)
                model.fit(x_train, y_train)

                for split_name in ["validation", "test", "recent_regime"]:
                    x_eval, y_eval = prepare_split(data, feature_columns, split_name)
                    if x_eval.empty:
                        continue
                    y_pred = model.predict(x_eval)
                    metric_rows.append(
                        evaluate_predictions(
                            name=name,
                            ticker=ticker,
                            lag=lag,
                            model_name=spec.name,
                            split_name=split_name,
                            y_true=y_eval,
                            y_pred=y_pred,
                        )
                    )

                x_test, _ = prepare_split(data, feature_columns, "test")
                importances = shap_importance(model, x_train, x_test)
                save_shap_bar(
                    importances,
                    name=name,
                    lag=lag,
                    model_name=spec.name,
                    output_dir=figures_dir,
                )
                for feature, value in importances.items():
                    shap_rows.append(
                        {
                            "name": name,
                            "ticker": ticker,
                            "lag_days": lag,
                            "model": spec.name,
                            "split": "test",
                            "feature": feature,
                            "mean_abs_shap": value,
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    shap_summary = pd.DataFrame(shap_rows)
    metrics.to_csv(results_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    shap_summary.to_csv(results_dir / "shap_feature_importance.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline classifiers on core preprocessed datasets.")
    parser.add_argument("--input-dir", default=str(PROCESSED_CORE_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_model_comparison(
        input_dir=Path(args.input_dir),
        results_dir=Path(args.results_dir),
        figures_dir=Path(args.figures_dir),
    )
