import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

try:
    from src.db import get_engine
except ModuleNotFoundError:
    from db import get_engine


RESULT_PATH = Path("outputs/results")
DEFAULT_FEATURE_SET = "macro_alpha"
TICKER_LABELS = {
    "005930": "samsung",
    "000660": "sk_hynix",
    "NVDA": "nvda",
}


# 지정한 날짜 구간의 feature별 평균 SHAP 영향력을 DB에서 조회한다.
def load_period_importance(
    ticker: str,
    feature_set: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    query = text("""
        SELECT
            feature_name,
            COUNT(*) AS sample_count,
            AVG(ABS(shap_value)) AS mean_abs_shap,
            AVG(shap_value) AS mean_shap,
            AVG(feature_value) AS mean_feature_value,
            SUM(CASE WHEN shap_value > 0 THEN 1 ELSE 0 END) AS positive_shap_count,
            SUM(CASE WHEN shap_value < 0 THEN 1 ELSE 0 END) AS negative_shap_count
        FROM shap_results
        WHERE ticker = :ticker
          AND feature_set = :feature_set
          AND date BETWEEN :start_date AND :end_date
        GROUP BY feature_name
        ORDER BY mean_abs_shap DESC
    """)
    params = {
        "ticker": ticker,
        "feature_set": feature_set,
        "start_date": start_date,
        "end_date": end_date,
    }
    df = pd.read_sql(query, get_engine(), params=params)
    if df.empty:
        raise ValueError(
            "해당 조건의 SHAP 결과가 없습니다. "
            f"ticker={ticker}, feature_set={feature_set}, period={start_date}~{end_date}"
        )

    df["positive_shap_ratio"] = df["positive_shap_count"] / df["sample_count"]
    df["negative_shap_ratio"] = df["negative_shap_count"] / df["sample_count"]
    df["dominant_direction"] = df["mean_shap"].apply(
        lambda value: "positive" if value > 0 else "negative" if value < 0 else "neutral"
    )
    return df


# 구간별 지표 영향력 결과를 CSV 파일로 저장한다.
def save_period_importance(
    df: pd.DataFrame,
    ticker: str,
    feature_set: str,
    start_date: str,
    end_date: str,
    output_dir: Path = RESULT_PATH,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    label = TICKER_LABELS.get(ticker, ticker.lower())
    output_path = output_dir / (
        f"period_feature_importance_{label}_{feature_set}_{start_date}_{end_date}.csv"
    )
    df.to_csv(output_path, index=False)
    return output_path


# 구간별 지표 영향력 상위 결과를 콘솔에 출력한다.
def print_period_importance(
    df: pd.DataFrame,
    ticker: str,
    feature_set: str,
    start_date: str,
    end_date: str,
    top_n: int,
) -> None:
    print("[구간별 지표 영향력 분석]")
    print(f"Ticker: {ticker}")
    print(f"Feature set: {feature_set}")
    print(f"Period: {start_date} ~ {end_date}")
    print(f"Features: {len(df)}개")
    print()

    display_columns = [
        "feature_name",
        "mean_abs_shap",
        "mean_shap",
        "dominant_direction",
        "positive_shap_ratio",
        "mean_feature_value",
    ]
    print(f"[Top {top_n}]")
    print(df.head(top_n)[display_columns].to_string(index=False))


# 구간별 영향력 조회, 저장, 출력을 한 번에 실행한다.
def run_period_interpretation(
    ticker: str,
    feature_set: str,
    start_date: str,
    end_date: str,
    top_n: int,
) -> Path:
    df = load_period_importance(ticker, feature_set, start_date, end_date)
    output_path = save_period_importance(df, ticker, feature_set, start_date, end_date)
    print_period_importance(df, ticker, feature_set, start_date, end_date, top_n)
    print()
    print(f"[CSV 저장 완료] {output_path}")
    return output_path


# 커맨드라인 실행 옵션을 파싱한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="날짜 구간별 SHAP 지표 영향력 분석")
    parser.add_argument(
        "--ticker",
        required=True,
        choices=sorted(TICKER_LABELS),
        help="분석할 종목 티커입니다. 삼성전자=005930, SK하이닉스=000660, 엔비디아=NVDA",
    )
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FEATURE_SET,
        help="조회할 SHAP feature set입니다. 기본값은 macro_alpha입니다.",
    )
    parser.add_argument("--start", required=True, help="분석 시작일입니다. 예: 2024-01-01")
    parser.add_argument("--end", required=True, help="분석 종료일입니다. 예: 2024-12-31")
    parser.add_argument("--top-n", type=int, default=10, help="출력할 상위 feature 개수입니다.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_period_interpretation(
        ticker=args.ticker,
        feature_set=args.feature_set,
        start_date=args.start,
        end_date=args.end,
        top_n=args.top_n,
    )
