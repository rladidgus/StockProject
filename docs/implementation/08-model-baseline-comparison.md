# 모델 baseline 비교 실험

## 목적

전처리된 core dataset을 사용해 lag 기준별 분류 모델 결과를 비교한다.

## 입력

```text
data/processed/core/*_core_features.csv
```

중립 구간은 아직 최종 정책이 확정되지 않았으므로,
이번 baseline 비교에서는 `target_direction_1pct`가 비어 있는 row를 제외하고
상승(1)과 하락(0)만 이진분류한다.

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

`model_metrics.csv`에는 accuracy와 confusion matrix 구성값(`tn/fp/fn/tp`)을 저장한다.
SHAP은 test split 기준 mean absolute SHAP value를 저장하고 bar plot으로 내보낸다.

## 실행

```bash
python3 -m src.model_compare
```
