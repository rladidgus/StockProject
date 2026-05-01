import shap
import pandas as pd
import matplotlib.pyplot as plt
from src.db import get_engine

SAVE_PATH = "outputs/figures/"

CASE_STUDIES = [
    {"date": "2022-06-15", "label": "rate_hike_2022",  "desc": "2022년 금리 급등기"},
    {"date": "2024-03-08", "label": "ai_rally_2024",   "desc": "2024년 AI 반도체 랠리"},
]


def global_shap(model, X_test: pd.DataFrame) -> shap.Explanation:
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(f"{SAVE_PATH}global_bar.png", bbox_inches="tight", dpi=150)
    plt.close()

    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(f"{SAVE_PATH}global_beeswarm.png", bbox_inches="tight", dpi=150)
    plt.close()

    return shap_values


def local_shap(model, X_test: pd.DataFrame, date_index: str, label: str):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    idx = X_test.index.get_loc(date_index)

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[idx],
            feature_names=X_test.columns.tolist(),
        ),
        show=False,
    )
    plt.savefig(f"{SAVE_PATH}local_{label}.png", bbox_inches="tight", dpi=150)
    plt.close()


def sanity_check(shap_values, X_test: pd.DataFrame, feature_col: str, expected_direction: str) -> bool:
    corr = pd.Series(shap_values[:, X_test.columns.get_loc(feature_col)]).corr(X_test[feature_col])
    actual_direction = "positive" if corr > 0 else "negative"
    match = (actual_direction == expected_direction)
    icon  = "✅" if match else "⚠️ 경고"
    print(f"[{icon}] {feature_col}: 예상={expected_direction}, 실제={actual_direction}")
    return match


def save_shap_to_db(shap_values, X_test: pd.DataFrame, ticker: str):
    engine = get_engine()
    rows = [
        {
            "date": str(date),
            "ticker": ticker,
            "feature_name": feature,
            "shap_value": float(shap_values[i, j]),
            "feature_value": float(X_test.iloc[i, j]),
        }
        for i, date in enumerate(X_test.index)
        for j, feature in enumerate(X_test.columns)
    ]
    pd.DataFrame(rows).to_sql(
        "shap_results", engine,
        if_exists="append", index=False,
        method="multi", chunksize=1000,
    )
    print(f"[SHAP 저장 완료] {ticker}: {len(rows)}행")
