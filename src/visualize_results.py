from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("outputs/results")
FIGURE_DIR = Path("outputs/figures/results")
TOP_N = 20

TICKER_NAMES = {
    "005930": "Samsung Electronics",
    "000660": "SK hynix",
}


def normalize_ticker(value) -> str:
    return str(value).zfill(6)


def ticker_label(ticker: str) -> str:
    ticker = normalize_ticker(ticker)
    return f"{TICKER_NAMES.get(ticker, ticker)} ({ticker})"


def setup_output_dir() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_horizontal_bar(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    title: str,
    output_path: Path,
    color_col: str | None = None,
) -> None:
    plot_df = df.copy().iloc[::-1]
    colors = "#4c78a8"
    if color_col:
        colors = plot_df[color_col].map(lambda value: "#2a9d8f" if value >= 0 else "#e76f51")

    fig_height = max(6, 0.38 * len(plot_df))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(plot_df[label_col], plot_df[value_col], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel(value_col)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def visualize_feature_target_correlation() -> list[Path]:
    path = RESULTS_DIR / "feature_target_correlation.csv"
    df = pd.read_csv(path, dtype={"ticker": str})
    df["ticker"] = df["ticker"].map(normalize_ticker)
    outputs = []

    for ticker, group in df.groupby("ticker"):
        top = group.nlargest(TOP_N, "abs_pearson_corr_train").copy()
        top["feature_label"] = top["feature"]
        output_path = FIGURE_DIR / f"feature_target_correlation_top{TOP_N}_{ticker}.png"
        save_horizontal_bar(
            top,
            value_col="pearson_corr_train",
            label_col="feature_label",
            title=f"Top {TOP_N} Feature vs Target Correlation - {ticker_label(ticker)}",
            output_path=output_path,
            color_col="pearson_corr_train",
        )
        outputs.append(output_path)

    combined = (
        df.sort_values(["ticker", "abs_pearson_corr_train"], ascending=[True, False])
        .groupby("ticker")
        .head(10)
        .copy()
    )
    combined["label"] = combined["ticker"].map(ticker_label) + " | " + combined["feature"]
    output_path = FIGURE_DIR / "feature_target_correlation_top10_all_tickers.png"
    save_horizontal_bar(
        combined.sort_values("abs_pearson_corr_train"),
        value_col="pearson_corr_train",
        label_col="label",
        title="Top Feature vs Target Correlation by Ticker",
        output_path=output_path,
        color_col="pearson_corr_train",
    )
    outputs.append(output_path)
    return outputs


def visualize_logistic_coefficients() -> list[Path]:
    path = RESULTS_DIR / "logistic_coefficients.csv"
    df = pd.read_csv(path, dtype={"ticker": str})
    df["ticker"] = df["ticker"].map(normalize_ticker)
    outputs = []

    for ticker, group in df.groupby("ticker"):
        top = group.nlargest(TOP_N, "abs_coefficient").copy()
        output_path = FIGURE_DIR / f"logistic_coefficients_top{TOP_N}_{ticker}.png"
        save_horizontal_bar(
            top,
            value_col="coefficient",
            label_col="feature",
            title=f"Top {TOP_N} Logistic Coefficients - {ticker_label(ticker)}",
            output_path=output_path,
            color_col="coefficient",
        )
        outputs.append(output_path)

    return outputs


def visualize_model_comparison() -> list[Path]:
    path = RESULTS_DIR / "model_comparison.csv"
    df = pd.read_csv(path, dtype={"ticker": str})
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["ticker_label"] = df["ticker"].map(ticker_label)
    df["model_label"] = df["model"].str.replace("xgboost_tuned_train_val_", "xgb_", regex=False)
    df["model_label"] = df["model_label"].str.replace("logistic_regression_", "logreg_", regex=False)
    outputs = []

    for metric in ["auc", "f1", "recall", "precision"]:
        pivot = df.pivot_table(index="model_label", columns="ticker_label", values=metric, aggfunc="max")
        pivot = pivot.sort_index()
        fig, ax = plt.subplots(figsize=(12, max(6, 0.45 * len(pivot))))
        pivot.plot(kind="barh", ax=ax, width=0.78)
        ax.set_title(f"Model Comparison - {metric.upper()}", fontsize=14, pad=12)
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(title="")
        fig.tight_layout()
        output_path = FIGURE_DIR / f"model_comparison_{metric}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output_path)

    return outputs


def visualize_period_feature_importance() -> list[Path]:
    outputs = []
    for path in sorted(RESULTS_DIR.glob("period_feature_importance_*.csv")):
        df = pd.read_csv(path)
        top = df.nlargest(TOP_N, "mean_abs_shap").copy()
        top["signed_importance"] = top["mean_abs_shap"]
        top.loc[top["dominant_direction"] == "negative", "signed_importance"] *= -1
        output_path = FIGURE_DIR / f"{path.stem}_top{TOP_N}.png"
        save_horizontal_bar(
            top,
            value_col="signed_importance",
            label_col="feature_name",
            title=f"Top {TOP_N} Period SHAP Importance - {path.stem}",
            output_path=output_path,
            color_col="signed_importance",
        )
        outputs.append(output_path)

    return outputs


def main() -> None:
    setup_output_dir()
    outputs = []
    outputs.extend(visualize_feature_target_correlation())
    outputs.extend(visualize_logistic_coefficients())
    outputs.extend(visualize_model_comparison())
    outputs.extend(visualize_period_feature_importance())

    print("[visualization complete]")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
