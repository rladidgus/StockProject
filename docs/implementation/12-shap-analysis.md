# XGBoost SHAP 분석

## 목적

Proposed XGBoost ablation 결과에서 종목별 validation macro F1이 가장 높은 모델을 선택하고,
XGBoost tree contribution 값을 사용해 test/recent regime global SHAP과 test local SHAP 산출물을 만든다.

이 산출물은 현실 인과를 증명하는 자료가 아니라,
모델이 각 class 확률을 예측할 때 어떤 feature가 더 크게 기여했는지 설명하는 자료다.

## 실행

```bash
python3 -m src.shap_analysis
```

사전에 아래 산출물이 있어야 한다.

```text
outputs/results/proposed_xgboost/model_metrics.csv
```

## 모델 선택 기준

종목별로 `validation` split의 `f1_macro`가 가장 높은 ablation 실험을 선택한다.
현재 선택 결과는 다음 파일에 저장된다.

```text
outputs/results/shap_analysis/selected_models.csv
```

## 산출물

```text
outputs/results/shap_analysis/selected_models.csv
outputs/results/shap_analysis/global_shap_importance.csv
outputs/results/shap_analysis/local_shap_cases.csv
outputs/results/shap_analysis/local_case_selection.csv
outputs/results/shap_analysis/*_global_bar.png
outputs/results/shap_analysis/*_local.png
```

### Global

`global_shap_importance.csv`는 split, class, feature별 평균 절대 contribution과
평균 signed contribution을 저장한다.

PNG global bar는 test split의 overall mean absolute contribution 상위 feature를 보여준다.

### Local

`local_shap_cases.csv`는 test split에서 down/up class 설명 사례를 골라
상위 contribution feature를 저장한다. 해당 class로 예측된 row가 없으면
그 class 확률이 가장 높은 row를 fallback으로 선택한다.

`local_case_selection.csv`는 각 local case가 실제 predicted class에서 나온 것인지,
fallback으로 선택된 것인지 기록한다.

PNG local plot은 해당 날짜의 explained class logit에 대한 feature별 contribution을 보여준다.

## 해석 주의

- 양수 contribution은 해당 class 쪽 모델 출력을 높였다는 뜻이다.
- 음수 contribution은 해당 class 쪽 모델 출력을 낮췄다는 뜻이다.
- feature 중요도가 높다고 현실 원인이라고 말하면 안 된다.
- 중복 피처군에서는 중요도가 여러 feature로 분산될 수 있다.
- 성능이 낮은 모델의 SHAP 결과는 발표에서 보조 해석 자료로만 사용한다.

## 도메인 안전장치

- split 방식: Proposed ablation과 동일한 시간 순서 split을 사용한다.
- look-ahead 방지: SHAP 대상 모델은 lag/as-of 처리된 feature만 사용한다.
- 모델 선택: validation macro F1 기준으로 선택하고, test/recent regime은 선택 이후 평가와 해석에만 사용한다.
- 모델 투입 제외 피처: 뉴스 감성, HTML archive, 공시일 정렬이 불완전한 재무제표는 제외한다.
- 발표 시 해석 주의: SHAP 값은 인과 효과가 아니라 XGBoost 예측 기여도다.
