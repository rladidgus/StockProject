import shap
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sqlalchemy import text

try:
    from src.db import get_engine
    from src.model import (
        FEATURE_SETS,
        combine_train_validation,
        load_preprocessed_data,
        split_features_target,
        split_train_val_test,
        train_xgboost_with_params,
    )
except ModuleNotFoundError:
    from db import get_engine
    from model import (
        FEATURE_SETS,
        combine_train_validation,
        load_preprocessed_data,
        split_features_target,
        split_train_val_test,
        train_xgboost_with_params,
    )

SAVE_PATH = Path("outputs/figures/")
TICKER = "005930"
DEFAULT_PREPROCESSED_PATHS = {
    "005930": Path("data/processed/samsung_preprocessed.csv"),
    "000660": Path("data/processed/sk_hynix_preprocessed.csv"),
}
OUTPUT_LABELS = {
    "005930": "samsung",
    "000660": "sk_hynix",
}
BEST_PARAMS_BY_TICKER_FEATURE_SET = {
    ("005930", "proposed"): {
        "n_estimators": 278,
        "max_depth": 3,
        "learning_rate": 0.05082341959721458,
        "min_child_weight": 2,
        "gamma": 4.010984903770199,
        "subsample": 0.6298202574719083,
        "colsample_bytree": 0.9947547746402069,
    },
    ("000660", "proposed"): {
        "n_estimators": 263,
        "max_depth": 6,
        "learning_rate": 0.1539036762661932,
        "min_child_weight": 10,
        "gamma": 0.04613834522225535,
        "subsample": 0.8945158729359323,
        "colsample_bytree": 0.624228663593609,
    },
    ("005930", "macro_alpha"): {
        "n_estimators": 412,
        "max_depth": 5,
        "learning_rate": 0.06755699887253351,
        "min_child_weight": 2,
        "gamma": 4.98421901282484,
        "subsample": 0.768954939720415,
        "colsample_bytree": 0.9398318228555668,
    },
    ("000660", "macro_alpha"): {
        "n_estimators": 287,
        "max_depth": 6,
        "learning_rate": 0.08960785365368121,
        "min_child_weight": 6,
        "gamma": 0.7800932022121826,
        "subsample": 0.662397808134481,
        "colsample_bytree": 0.6232334448672797,
    },
}

CASE_STUDIES = [
    {"date": "2022-06-15", "label": "rate_hike_2022",  "desc": "2022년 금리 급등기"},
    {"date": "2024-03-08", "label": "ai_rally_2024",   "desc": "2024년 AI 반도체 랠리"},
]


# Test set 전체의 글로벌 SHAP bar/beeswarm 그래프를 생성한다.
def global_shap(model, X_test: pd.DataFrame, output_label: str):
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(SAVE_PATH / f"{output_label}_global_bar.png", bbox_inches="tight", dpi=150)
    plt.close()

    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(SAVE_PATH / f"{output_label}_global_beeswarm.png", bbox_inches="tight", dpi=150)
    plt.close()

    return explainer, shap_values


# 특정 날짜 한 건에 대한 로컬 SHAP waterfall 그래프를 생성한다.
def local_shap(explainer, shap_values, X_test: pd.DataFrame, date_index: str, label: str, output_label: str):
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
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
    plt.savefig(SAVE_PATH / f"{output_label}_local_{label}.png", bbox_inches="tight", dpi=150)
    plt.close()


# 특정 feature의 값과 SHAP 방향이 예상 경제 방향과 맞는지 확인한다.
def sanity_check(shap_values, X_test: pd.DataFrame, feature_col: str, expected_direction: str) -> bool:
    corr = pd.Series(shap_values[:, X_test.columns.get_loc(feature_col)]).corr(X_test[feature_col])
    actual_direction = "positive" if corr > 0 else "negative"
    match = (actual_direction == expected_direction)
    icon  = "✅" if match else "⚠️ 경고"
    print(f"[{icon}] {feature_col}: 예상={expected_direction}, 실제={actual_direction}")
    return match


# 날짜, 티커, feature set, feature별 SHAP 값을 PostgreSQL에 저장한다.
def save_shap_to_db(shap_values, X_test: pd.DataFrame, ticker: str, feature_set: str):
    engine = get_engine()
    date_index = pd.to_datetime(X_test.index).date
    rows = [
        {
            "date": str(date_index[i]),
            "ticker": ticker,
            "feature_set": feature_set,
            "feature_name": feature,
            "shap_value": float(shap_values[i, j]),
            "feature_value": float(X_test.iloc[i, j]),
        }
        for i, date in enumerate(X_test.index)
        for j, feature in enumerate(X_test.columns)
    ]
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE shap_results
                ADD COLUMN IF NOT EXISTS feature_set VARCHAR(50) NOT NULL DEFAULT 'proposed',
                ALTER COLUMN shap_value TYPE NUMERIC(20,6),
                ALTER COLUMN feature_value TYPE NUMERIC(20,6)
        """))
        conn.execute(
            text("DELETE FROM shap_results WHERE ticker = :ticker AND feature_set = :feature_set"),
            {"ticker": ticker, "feature_set": feature_set},
        )
        conn.commit()

    pd.DataFrame(rows).to_sql(
        "shap_results", engine,
        if_exists="append", index=False,
        method="multi", chunksize=1000,
    )
    print(f"[SHAP 저장 완료] {ticker} / {feature_set}: {len(rows)}행")


# SHAP 계산을 위해 Train+Validation으로 최종 XGBoost 모델을 학습하고 Test X를 준비한다.
def prepare_final_model_and_test_data(ticker: str, input_path: Path, params: dict, feature_set: str):
    data = load_preprocessed_data(input_path)
    X, y = split_features_target(data, feature_set=feature_set)
    splits = split_train_val_test(X, y, data["date"])
    X_train_val, y_train_val = combine_train_validation(splits)
    model = train_xgboost_with_params(X_train_val, y_train_val, params)

    X_test = splits["X_test"].copy()
    X_test.index = pd.to_datetime(splits["date_test"])
    return model, X_test


# 지정한 티커와 feature set에 대해 SHAP 분석 전체 과정을 실행한다.
def run_shap_analysis(ticker: str = TICKER, input_path: Path | None = None, feature_set: str = "proposed"):
    input_path = input_path or DEFAULT_PREPROCESSED_PATHS[ticker]
    output_label = f"{OUTPUT_LABELS[ticker]}_{feature_set}"
    params = BEST_PARAMS_BY_TICKER_FEATURE_SET[(ticker, feature_set)]
    model, X_test = prepare_final_model_and_test_data(ticker, input_path, params, feature_set)
    explainer, shap_values = global_shap(model, X_test, output_label)
    save_shap_to_db(shap_values, X_test, ticker, feature_set)

    for case in CASE_STUDIES:
        date = case["date"]
        if date in X_test.index.strftime("%Y-%m-%d"):
            local_shap(explainer, shap_values, X_test, date, case["label"], output_label)
        else:
            print(f"[로컬 SHAP 건너뜀] {date}: Test set에 없음")

    sanity_check(shap_values, X_test, "us_rate", expected_direction="negative")
    sanity_check(shap_values, X_test, "vix", expected_direction="negative")
    print("[SHAP 분석 완료]")


# 삼성전자 기본 SHAP 분석을 실행하는 호환용 래퍼 함수다.
def run_samsung_shap_analysis():
    run_shap_analysis(TICKER)


# 커맨드라인 실행 옵션을 파싱한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="국내 반도체 종목 SHAP 분석")
    parser.add_argument(
        "--ticker",
        default=TICKER,
        choices=sorted(DEFAULT_PREPROCESSED_PATHS),
        help="SHAP 분석할 종목 티커입니다. 삼성전자=005930, SK하이닉스=000660",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="전처리 CSV 경로입니다. 생략하면 티커별 기본 경로를 사용합니다.",
    )
    parser.add_argument(
        "--feature-set",
        default="proposed",
        choices=sorted(FEATURE_SETS),
        help="SHAP 분석할 feature set입니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_shap_analysis(args.ticker, args.input, args.feature_set)
