import pandas as pd
import argparse
from pathlib import Path


PREPROCESSED_PATH = Path("data/processed/samsung_preprocessed.csv")
DEFAULT_PREPROCESSED_PATHS = {
    "005930": Path("data/processed/samsung_preprocessed.csv"),
    "000660": Path("data/processed/sk_hynix_preprocessed.csv"),
}
RESULT_PATH = Path("outputs/results/model_comparison.csv")
CORRELATION_PATH = Path("outputs/results/feature_target_correlation.csv")
LOGISTIC_COEFFICIENT_PATH = Path("outputs/results/logistic_coefficients.csv")
TARGET_COLUMN = "target_up"
EXCLUDE_FEATURE_COLUMNS = ["date", "target_up", "next_return_pct"]
TICKER = "005930"

BASELINE_FEATURES = ["ma5", "ma10", "ma20", "rsi", "bb_upper", "bb_lower"]
PROPOSED_FEATURES = [
    "open", "high", "low", "close", "volume", "return_pct",
    "nasdaq", "sox", "vix", "us_rate", "usd_krw", "kospi", "kospi200",
    "nasdaq_lag1", "nasdaq_lag3", "nasdaq_lag5",
    "vix_lag1", "vix_lag3", "vix_lag5",
    "us_rate_lag1", "us_rate_lag3", "us_rate_lag5",
    "usd_krw_lag1", "usd_krw_lag3", "usd_krw_lag5",
    "sox_lag1", "sox_lag3", "sox_lag5",
    "kospi_lag1", "kospi_lag3", "kospi_lag5",
    "kospi200_lag1", "kospi200_lag3", "kospi200_lag5",
    "foreign_net_buy", "institution_net_buy",
    "nasdaq_return_1d", "nasdaq_return_3d",
    "sox_return_1d", "sox_return_3d",
    "vix_change_1d", "vix_change_3d",
    "usd_krw_return_1d", "usd_krw_return_3d",
    "kospi_return_1d", "kospi_return_3d",
    "kospi200_return_1d", "kospi200_return_3d",
    "stock_vs_kospi_return_1d", "stock_vs_kospi200_return_1d",
    "foreign_net_buy_5d_sum", "foreign_net_buy_20d_sum",
    "institution_net_buy_5d_sum", "institution_net_buy_20d_sum",
    "trading_value",
    "foreign_net_buy_to_volume", "institution_net_buy_to_volume",
    "foreign_net_buy_to_trading_value", "institution_net_buy_to_trading_value",
    "foreign_net_buy_5d_to_trading_value_5d",
    "foreign_net_buy_20d_to_trading_value_20d",
    "institution_net_buy_5d_to_trading_value_5d",
    "institution_net_buy_20d_to_trading_value_20d",
]
MACRO_ALPHA_FEATURES = [
    feature
    for feature in PROPOSED_FEATURES
    if feature not in {"open", "high", "low", "close", "trading_value"}
]
PROPOSED_PLUS_FEATURES = PROPOSED_FEATURES + BASELINE_FEATURES
FEATURE_SETS = {
    "baseline": BASELINE_FEATURES,
    "macro_alpha": MACRO_ALPHA_FEATURES,
    "proposed": PROPOSED_FEATURES,
    "proposed_plus": PROPOSED_PLUS_FEATURES,
}


# 전처리 CSV를 읽고 날짜순으로 정렬한 뒤 필수 컬럼을 검증한다.
def load_preprocessed_data(path: Path = PREPROCESSED_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"전처리 파일이 없습니다: {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required_columns = {"date", "target_up", "next_return_pct"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError(f"전처리 파일이 비어 있습니다: {path}")

    return df


# 선택한 feature set 기준으로 모델 입력 X와 타겟 y를 분리한다.
def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    exclude_cols: list[str] = EXCLUDE_FEATURE_COLUMNS,
    feature_set: str = "proposed_plus",
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"타겟 컬럼이 없습니다: {target_col}")

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"지원하지 않는 feature_set입니다: {feature_set}")

    selected_columns = [col for col in FEATURE_SETS[feature_set] if col in df.columns]
    missing_columns = sorted(set(FEATURE_SETS[feature_set]) - set(selected_columns))
    if missing_columns:
        print(f"[feature 제외] {feature_set}: 데이터에 없는 컬럼 {missing_columns}")

    feature_df = df[selected_columns].drop(columns=[col for col in exclude_cols if col in selected_columns])
    feature_df = feature_df.select_dtypes(include=["number"])
    if feature_df.empty:
        raise ValueError("모델 입력 feature가 없습니다.")

    target = df[target_col].astype(int)
    return feature_df, target


# 하나의 DataFrame을 시간 순서대로 Train/Validation/Test로 나눈다.
def time_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# X, y, 날짜를 같은 기준으로 시간순 Train/Validation/Test 세트로 나눈다.
def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, pd.DataFrame | pd.Series]:
    if not len(X) == len(y) == len(dates):
        raise ValueError("X, y, dates 길이가 서로 다릅니다.")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("train_ratio와 val_ratio는 0과 1 사이여야 합니다.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio는 1보다 작아야 합니다.")

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "date_train": dates.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end],
        "y_val": y.iloc[train_end:val_end],
        "date_val": dates.iloc[train_end:val_end],
        "X_test": X.iloc[val_end:],
        "y_test": y.iloc[val_end:],
        "date_test": dates.iloc[val_end:],
    }


# 분할된 데이터의 기간, 행 수, 타겟 분포를 문자열로 요약한다.
def describe_split(name: str, dates: pd.Series, y: pd.Series) -> str:
    counts = y.value_counts().sort_index().to_dict()
    return (
        f"{name}: {len(y)}행, "
        f"{dates.min().date()} ~ {dates.max().date()}, "
        f"target={counts}"
    )


# 기본 설정의 XGBoost 분류 모델을 생성한다.
def make_default_xgb_model(scale_pos_weight: float = 1.0):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )


# 학습 데이터의 상승/하락 클래스 비율로 XGBoost 불균형 보정값을 계산한다.
def get_scale_pos_weight(y_train: pd.Series) -> float:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    if pos == 0:
        return 1.0
    return neg / pos


# 기본 파라미터 XGBoost 모델을 학습한다.
def train_default_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    scale_pos_weight = get_scale_pos_weight(y_train)
    model = make_default_xgb_model(scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    return model


# 분류 모델의 Accuracy, Precision, Recall, F1, AUC를 계산한다.
def evaluate_classifier(model, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X)
    pred = (proba >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, proba),
        "threshold": threshold,
    }


# 성능 지표 딕셔너리를 보기 좋은 문자열로 변환한다.
def format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())


# Validation F1-score가 가장 높은 분류 threshold를 찾는다.
def find_best_threshold(model, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[float, float]:
    from sklearn.metrics import f1_score

    proba = model.predict_proba(X_val)[:, 1]
    candidates = [i / 100 for i in range(10, 91)]
    scores = [(threshold, f1_score(y_val, proba >= threshold, zero_division=0)) for threshold in candidates]
    return max(scores, key=lambda item: item[1])


# Optuna가 Validation AUC를 최대화하도록 사용할 objective 함수를 만든다.
def build_objective(X_train, y_train, X_val, y_val, scale_weight):
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": scale_weight,
            "random_state": 42,
            "eval_metric": "auc",
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, val_proba)
    return objective


# Optuna로 XGBoost 하이퍼파라미터를 튜닝하고 최적 모델과 study를 반환한다.
def train_tuned_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 30):
    import optuna
    from xgboost import XGBClassifier

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_weight = neg / pos

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(build_objective(X_train, y_train, X_val, y_val, scale_weight), n_trials=n_trials)

    best_params = study.best_params | {
        "scale_pos_weight": scale_weight,
        "random_state": 42,
        "eval_metric": "auc",
    }
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, study


# 이미 정해진 파라미터로 XGBoost 모델을 학습한다.
def train_xgboost_with_params(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    from xgboost import XGBClassifier

    final_params = params | {
        "scale_pos_weight": get_scale_pos_weight(y_train),
        "random_state": 42,
        "eval_metric": "auc",
    }
    model = XGBClassifier(**final_params)
    model.fit(X_train, y_train)
    return model


# Train과 Validation 세트를 합쳐 최종 학습용 데이터로 만든다.
def combine_train_validation(splits: dict[str, pd.DataFrame | pd.Series]) -> tuple[pd.DataFrame, pd.Series]:
    X_train_val = pd.concat([splits["X_train"], splits["X_val"]], axis=0)
    y_train_val = pd.concat([splits["y_train"], splits["y_val"]], axis=0)
    return X_train_val, y_train_val


# 모델 성능 결과를 model_comparison.csv에 저장하거나 기존 행을 갱신한다.
def save_model_result(
    metrics: dict[str, float],
    model_name: str,
    ticker: str = TICKER,
    output_path: Path = RESULT_PATH,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ticker": ticker,
        "model": model_name,
        **metrics,
    }
    result_df = pd.DataFrame([row])
    if output_path.exists():
        existing = pd.read_csv(output_path, dtype={"ticker": str})
        existing["ticker"] = existing["ticker"].str.zfill(6)
        existing = existing[
            ~((existing["ticker"] == ticker) & (existing["model"] == model_name))
        ]
        result_df = pd.concat([existing, result_df], ignore_index=True)
    result_df.to_csv(output_path, index=False)
    print(f"[결과 저장 완료] {output_path}")


# Train 구간에서 feature와 target의 피어슨 상관계수를 계산해 저장한다.
def save_feature_target_correlation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_set: str,
    ticker: str = TICKER,
    output_path: Path = CORRELATION_PATH,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for feature in X_train.columns:
        corr = X_train[feature].corr(y_train)
        rows.append({
            "ticker": ticker,
            "feature_set": feature_set,
            "feature": feature,
            "pearson_corr_train": corr,
            "abs_pearson_corr_train": abs(corr) if pd.notna(corr) else pd.NA,
        })

    current_corr_df = pd.DataFrame(rows).sort_values(
        "abs_pearson_corr_train",
        ascending=False,
        na_position="last",
    )
    output_df = current_corr_df
    if output_path.exists():
        existing = pd.read_csv(output_path, dtype={"ticker": str})
        existing["ticker"] = existing["ticker"].str.zfill(6)
        existing = existing[
            ~((existing["ticker"] == ticker) & (existing["feature_set"] == feature_set))
        ]
        output_df = pd.concat([existing, current_corr_df], ignore_index=True)
    output_df.to_csv(output_path, index=False)
    print(f"[상관분석 저장 완료] {output_path}")
    return current_corr_df


# 표준화와 class_weight를 포함한 Logistic Regression 보조 검증 모델을 학습한다.
def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    return model


# Logistic Regression 계수를 feature별로 저장한다.
def save_logistic_coefficients(
    model,
    feature_names: list[str],
    feature_set: str,
    ticker: str = TICKER,
    output_path: Path = LOGISTIC_COEFFICIENT_PATH,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coefficients = model.named_steps["logistic"].coef_[0]
    current_coef_df = pd.DataFrame({
        "ticker": ticker,
        "feature_set": feature_set,
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": abs(coefficients),
    }).sort_values("abs_coefficient", ascending=False)
    output_df = current_coef_df
    if output_path.exists():
        existing = pd.read_csv(output_path, dtype={"ticker": str})
        existing["ticker"] = existing["ticker"].str.zfill(6)
        existing = existing[
            ~((existing["ticker"] == ticker) & (existing["feature_set"] == feature_set))
        ]
        output_df = pd.concat([existing, current_coef_df], ignore_index=True)
    output_df.to_csv(output_path, index=False)
    print(f"[로지스틱 계수 저장 완료] {output_path}")
    return current_coef_df


# Logistic Regression과 상관분석으로 XGBoost 결과를 보조 검증한다.
def run_statistical_validation(
    data: pd.DataFrame,
    feature_set: str = "proposed",
    ticker: str = TICKER,
) -> dict[str, float]:
    X, y = split_features_target(data, feature_set=feature_set)
    splits = split_train_val_test(X, y, data["date"])

    corr_df = save_feature_target_correlation(
        splits["X_train"],
        splits["y_train"],
        feature_set=feature_set,
        ticker=ticker,
    )

    validation_model = train_logistic_regression(splits["X_train"], splits["y_train"])
    best_threshold, best_threshold_f1 = find_best_threshold(
        validation_model,
        splits["X_val"],
        splits["y_val"],
    )
    validation_metrics = evaluate_classifier(
        validation_model,
        splits["X_val"],
        splits["y_val"],
        threshold=best_threshold,
    )

    X_train_val, y_train_val = combine_train_validation(splits)
    final_model = train_logistic_regression(X_train_val, y_train_val)
    test_metrics = evaluate_classifier(
        final_model,
        splits["X_test"],
        splits["y_test"],
        threshold=best_threshold,
    )
    save_logistic_coefficients(final_model, X.columns.tolist(), feature_set=feature_set, ticker=ticker)
    save_model_result(test_metrics, f"logistic_regression_{feature_set}", ticker=ticker)

    print(f"[{feature_set} 통계 검증 완료]")
    print(f"Logistic Validation: {format_metrics(validation_metrics)}")
    print(f"Best threshold: {best_threshold:.2f} (validation f1={best_threshold_f1:.4f})")
    print(f"Logistic Test: {format_metrics(test_metrics)}")
    print("[상관분석 상위 10개]")
    print(corr_df.head(10).to_string(index=False))
    return test_metrics


# 선택한 feature set으로 XGBoost 튜닝, 최종 재학습, Test 평가를 실행한다.
def run_tuned_pipeline(
    data: pd.DataFrame,
    feature_set: str,
    n_trials: int,
    ticker: str = TICKER,
    model_name: str | None = None,
) -> dict[str, float]:
    X, y = split_features_target(data, feature_set=feature_set)
    splits = split_train_val_test(X, y, data["date"])
    tuned_model, study = train_tuned_xgboost(
        splits["X_train"],
        splits["y_train"],
        splits["X_val"],
        splits["y_val"],
        n_trials=n_trials,
    )
    best_threshold, best_threshold_f1 = find_best_threshold(
        tuned_model,
        splits["X_val"],
        splits["y_val"],
    )
    tuned_val_metrics = evaluate_classifier(
        tuned_model,
        splits["X_val"],
        splits["y_val"],
        threshold=best_threshold,
    )
    X_train_val, y_train_val = combine_train_validation(splits)
    final_model = train_xgboost_with_params(X_train_val, y_train_val, study.best_params)
    test_metrics = evaluate_classifier(
        final_model,
        splits["X_test"],
        splits["y_test"],
        threshold=best_threshold,
    )

    print(f"[{feature_set} 튜닝 완료]")
    print(f"Feature: {X.shape[1]}개")
    print(f"Best trial AUC: {study.best_value:.4f}")
    print(f"Best threshold: {best_threshold:.2f} (validation f1={best_threshold_f1:.4f})")
    print(f"Best params: {study.best_params}")
    print(f"Validation: {format_metrics(tuned_val_metrics)}")
    print(f"Test: {format_metrics(test_metrics)}")

    save_model_result(test_metrics, model_name or f"xgboost_tuned_train_val_{feature_set}", ticker=ticker)
    return test_metrics


# 외부 호출용 XGBoost 튜닝 학습 래퍼 함수다.
def train(X_train, y_train, X_val, y_val, n_trials: int = 30):
    model, _ = train_tuned_xgboost(X_train, y_train, X_val, y_val, n_trials=n_trials)
    return model


# 커맨드라인 실행 옵션을 파싱한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="국내 반도체 종목 XGBoost 모델링")
    parser.add_argument(
        "--ticker",
        default=TICKER,
        choices=sorted(DEFAULT_PREPROCESSED_PATHS),
        help="모델링할 종목 티커입니다. 삼성전자=005930, SK하이닉스=000660",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="전처리 CSV 경로입니다. 생략하면 티커별 기본 경로를 사용합니다.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Validation AUC 기준으로 Optuna 하이퍼파라미터 튜닝을 실행합니다.",
    )
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trial 횟수")
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="proposed_plus",
        help="모델에 사용할 feature set",
    )
    parser.add_argument(
        "--compare-feature-sets",
        action="store_true",
        help="baseline/proposed/proposed_plus를 같은 방식으로 비교합니다.",
    )
    parser.add_argument(
        "--validate-statistical",
        action="store_true",
        help="Logistic Regression과 feature-target 상관분석으로 보조 검증을 실행합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input or DEFAULT_PREPROCESSED_PATHS[args.ticker]
    data = load_preprocessed_data(input_path)
    X, y = split_features_target(data, feature_set=args.feature_set)
    splits = split_train_val_test(X, y, data["date"])
    print(f"[로드 완료] {input_path}")
    print(f"Ticker: {args.ticker}")
    print(f"행/컬럼: {data.shape[0]}행, {data.shape[1]}컬럼")
    print(f"기간: {data['date'].min().date()} ~ {data['date'].max().date()}")
    print(f"Feature: {X.shape[1]}개")
    print(f"Target: {TARGET_COLUMN}")
    print("target_up 분포:")
    print(y.value_counts().sort_index())
    print("[시간순 분할]")
    print(describe_split("Train", splits["date_train"], splits["y_train"]))
    print(describe_split("Validation", splits["date_val"], splits["y_val"]))
    print(describe_split("Test", splits["date_test"], splits["y_test"]))
    model = train_default_xgboost(splits["X_train"], splits["y_train"])
    val_metrics = evaluate_classifier(model, splits["X_val"], splits["y_val"])
    print("[XGBoost 기본 모델 학습 완료]")
    print(f"Validation: {format_metrics(val_metrics)}")
    if args.tune:
        if args.compare_feature_sets:
            for feature_set in ["baseline", "proposed", "proposed_plus"]:
                run_tuned_pipeline(data, feature_set=feature_set, n_trials=args.n_trials, ticker=args.ticker)
        else:
            run_tuned_pipeline(data, feature_set=args.feature_set, n_trials=args.n_trials, ticker=args.ticker)
    if args.validate_statistical:
        run_statistical_validation(data, feature_set=args.feature_set, ticker=args.ticker)
