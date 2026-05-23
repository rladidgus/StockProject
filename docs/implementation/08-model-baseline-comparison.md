# 모델 baseline 비교 실험

## 목적

전처리된 core dataset을 사용해 lag 기준별 분류 모델 결과를 비교한다.

## 입력

```text
data/processed/core/*_core_features.csv
```

공식 target은 `target_5d_3class`다.

```text
0 = down
1 = neutral
2 = up
```

예측일 `t` 장마감 이후 기준으로 `close[t+5] / close[t] - 1`을 계산하고,
종목별 train split의 30%/70% 분위수 threshold로 하락/중립/상승을 나눈다.
target이 비어 있는 row는 supervised 학습과 평가에서 제외한다.

## 모델

| 모델 | 스케일링 |
|---|---|
| SVM | `StandardScaler` 적용 |
| Decision Tree | 적용하지 않음 |
| Random Forest | 적용하지 않음 |
| ANN | `StandardScaler` 적용 |
| Logistic Regression | `StandardScaler` 적용 |

Tree 계열은 스케일에 민감하지 않아 원값을 사용하고,
거리/선형/신경망 계열은 스케일 차이를 줄이기 위해 표준화한다.

## 실험 축

- 종목: 삼성전자, SK하이닉스, NVIDIA
- lag 기준: `t-1`, `t-3`, `t-5`
- 평가 기간:
  - validation
  - test
  - recent regime

## 산출물

```text
outputs/results/model_comparison/model_metrics.csv
outputs/results/model_comparison/shap_feature_importance.csv
outputs/figures/model_comparison/*_shap_importance.png
```

`model_metrics.csv`에는 accuracy, balanced accuracy, macro F1, weighted F1,
class별 precision/recall, 3-class confusion matrix, multiclass log loss,
예측 class별 평균 향후 5거래일 수익률, up/down long-short spread를 저장한다.

SHAP은 test split 기준 class별 mean absolute SHAP value를 저장하고 bar plot으로 내보낸다.

## 실행

```bash
python3 -m src.model_compare
```
