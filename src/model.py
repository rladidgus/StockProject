import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE


BASELINE_FEATURES = ["ma5", "ma10", "ma20", "rsi", "bb_upper", "bb_lower"]
PROPOSED_FEATURES = [
    "nasdaq", "vix", "us_rate", "usd_krw", "sox",
    "nasdaq_lag1", "vix_lag1", "us_rate_lag1", "usd_krw_lag1",
    "revenue", "eps", "foreign_net_buy",
]


def time_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def build_objective(X_train, y_train, X_val, y_val, scale_weight):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": scale_weight,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False,
        )
        return f1_score(y_val, model.predict(X_val))
    return objective


def train(X_train, y_train, X_val, y_val, n_trials: int = 100) -> XGBClassifier:
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_weight = neg / pos

    study = optuna.create_study(direction="maximize")
    study.optimize(build_objective(X_train, y_train, X_val, y_val, scale_weight), n_trials=n_trials)

    best_params = study.best_params | {"scale_pos_weight": scale_weight, "random_state": 42}
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
    return model
