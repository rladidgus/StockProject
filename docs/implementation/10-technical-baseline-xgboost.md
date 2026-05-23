# 기술적 baseline + core-only XGBoost

## 목적

기획안의 Baseline vs Proposed 구조 중 첫 checkpoint를 만드는 단계다.
이번 구현에서는 두 실험을 비교한다.

1. 기술적 baseline XGBoost
2. core macro-only XGBoost

둘 다 기존 전처리 dataset의 `target_5d_3class`와 embargo split을 그대로 사용한다.

## Label과 split

공식 target은 5거래일 미래수익률 3-class다.

```text
future_5d_return = close[t+5] / close[t] - 1
0 = down
1 = neutral
2 = up
```

예측 시점은 `t` 장마감 이후다. 따라서 기술적 지표는 `t` 장마감까지의 가격으로
계산할 수 있다. `close[t+5]`가 없는 row는 학습과 평가에서 제외한다.

train/validation/test는 전처리 단계에서 만든 시간 순서 split과 5거래일 embargo를 그대로 쓴다.

## Feature set

### 기술적 baseline

raw 가격에서 다음 feature를 계산한다.

- `tech_return_1d`
- `tech_ma5_ratio`
- `tech_ma10_ratio`
- `tech_ma20_ratio`
- `tech_rsi14`
- `tech_bb_percent_b`
- `tech_bb_width`
- `tech_volatility_20d`

MA와 Bollinger Band는 가격 수준 자체보다 현재 종가가 이동평균과 밴드 안에서 어디에 있는지를
나타내는 비율 feature로 사용한다.

### core macro-only

전처리 dataset의 core macro lag feature만 사용한다.

- NASDAQ
- SOX
- VIX
- FEDFUNDS
- USD/KRW
- lag: `t-1`, `t-3`, `t-5`

`*_source_date`, `*_known_date`, label audit column은 모델 feature에서 제외한다.

## 모델

두 실험 모두 `XGBClassifier`를 사용한다.

```text
objective = multi:softprob
num_class = 3
eval_metric = mlogloss
```

3-class 문제이므로 binary 전용 `scale_pos_weight` 대신 train class 빈도 기반 sample weight를 사용한다.

## 산출물

```text
outputs/results/technical_baseline_xgboost/model_metrics.csv
outputs/results/technical_baseline_xgboost/feature_importance.csv
outputs/results/technical_baseline_xgboost/feature_sets.csv
```

`model_metrics.csv`에는 accuracy, balanced accuracy, macro F1, weighted F1,
class별 precision/recall, multiclass log loss, weighted one-vs-rest ROC-AUC,
3-class confusion matrix, 예측 class별 평균 향후 5거래일 수익률,
up/down long-short spread를 저장한다.

`feature_importance.csv`는 XGBoost 내장 feature importance를 저장한다.
이 값은 checkpoint용이며, 최종 요인 해석은 SHAP 단계에서 별도로 수행한다.

## 실행

```bash
python3 -m src.technical_baseline_xgboost
```

## 1차 결과 요약

test split 기준 macro F1은 다음과 같다.

| 종목 | technical baseline | core-only |
|---|---:|---:|
| NVIDIA | 0.2563 | 0.1557 |
| 삼성전자 | 0.4111 | 0.2165 |
| SK하이닉스 | 0.3572 | 0.1520 |

삼성전자와 SK하이닉스에서는 기술적 baseline이 core-only보다 명확히 높다.
NVIDIA는 두 실험 모두 약하며, core-only는 일부 class로 예측이 쏠리는 경향이 있다.

이 결과는 core macro 피처만으로는 5거래일 3-class 방향 예측을 안정적으로 설명하기 어렵다는
checkpoint로 해석한다. 다음 단계에서는 extended feature를 무작정 모두 넣기보다,
금리/달러/반도체 ETF/업황 후보군을 나누어 ablation으로 검증한다.
