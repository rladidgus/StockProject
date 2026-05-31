# Model review and binary pivot

## 목적

현재 3-class 5거래일 방향 예측 모델을 코드 리뷰 관점에서 점검하고,
발표용으로 방어 가능한 모델링 방향을 정리한다. 핵심 결론은 다음과 같다.

- `proposed_daily_extended`는 최종 성능 모델 후보로 두기 어렵다.
- `technical_only` 계열이 전반적으로 가장 안정적이다.
- 3-class는 high-noise baseline/ablation 파트로 두고, 최종 성능 주장은 binary up/down 실험으로 옮기는 편이 안전하다.
- binary에서도 F1만 보면 안 되고, prediction balance와 long-short spread를 같이 봐야 한다.

## 코드 리뷰 결과

### 1. 누수 방지 구조

Core macro feature는 `src/preprocess.py`에서 `t-1`, `t-3`, `t-5` lag와 `known_date` 기준 as-of join으로 생성된다.
월간 지표는 월말 이후 5영업일 뒤부터 사용하도록 처리되어 있어 보수적이다.

검증 스크립트 `src/validate_preprocess.py`는 다음을 확인한다.

- core model feature에 same-day price column이 들어가지 않음
- 각 lag feature의 `known_date`가 lag cutoff보다 늦지 않음
- train/validation/test 사이 5거래일 horizon embargo가 유지됨
- 라벨 threshold boundary가 일관됨

현재 `python3 -m src.validate_preprocess`는 통과한다.

주의할 점은 technical feature다. `src/technical_baseline_xgboost.py`와
`src/technical_extended_search.py`의 technical feature는 당일 `close/high/low/volume`까지 반영한다.
따라서 발표에서는 예측 시점을 "장 마감 후 다음 5거래일 방향 예측"으로 정의해야 한다.
"당일 장중 예측"으로 말하면 technical feature는 누수가 된다.

### 2. 라벨 정책

현재 3-class label은 train split의 30/70 quantile로 `down/neutral/up`을 만든다.
이 방식은 누수 측면에서는 안전하지만, `neutral` 구간이 노이즈를 많이 포함한다.
실제 결과에서도 core/proposed 모델은 특정 class로 쏠리며 macro-F1이 낮게 나온다.

따라서 `neutral`을 제거한 binary up/down 실험을 추가했다.

추가 파일:

```text
src/binary_xgboost.py
src/binary_shap_analysis.py
src/binary_rolling_validation.py
outputs/results/binary_xgboost/model_metrics.csv
outputs/results/binary_xgboost/best_by_recipe.csv
outputs/results/binary_xgboost/selected_by_ticker.csv
outputs/results/binary_xgboost/search_summary.csv
outputs/results/binary_shap_analysis/*
outputs/results/binary_rolling_validation/*
```

Binary 실험은 기존 3-class target에서 `neutral` 행을 제거하고 `down=0`, `up=1`만 학습한다.
기존 `technical_extended_search`의 feature recipe와 XGBoost parameter grid를 재사용한다.

### 3. 모델 선택 로직

`src/technical_extended_search.py`는 validation 기준으로 recipe별 best parameter를 고르고,
test/recent는 선택 이후 평가로만 사용한다. 이 원칙은 유지해야 한다.

다만 validation F1만으로 고르면 한쪽 class 예측에 쏠린 모델이 선택될 수 있다.
새 binary 실험은 다음 품질 플래그를 추가했다.

- `pred_down_share`
- `pred_up_share`
- `min_predicted_class_share`
- `passed_prediction_balance`

Validation 선택 시 `passed_prediction_balance=True`인 모델을 우선한다.
기본 기준은 각 예측 class가 최소 10% 이상 나오는 것이다.

### 4. 현재 성능 비교

3-class technical extended search의 test macro-F1 상위 후보:

| ticker | recipe | params | test macro-F1 | recent macro-F1 |
|---|---|---|---:|---:|
| `000660` | `technical_only` | `depth4_slow_unweighted` | 0.4132 | 0.2960 |
| `005930` | `technical_only` | `depth2_fast_unweighted` | 0.3381 | 0.3434 |
| `NVDA` | `technical_only` | `depth3_conservative` | 0.3154 | 0.3360 |

기존 `technical_baseline_xgboost` 기준으로는 삼성전자 test macro-F1이 0.4111까지 나온다.
따라서 삼성전자는 extended search의 selected model보다 기존 technical baseline도 같이 보고해야 한다.

Binary search의 test macro-F1 상위 안정 후보:

| ticker | recipe | params | test macro-F1 | recent macro-F1 | test prediction balance |
|---|---|---|---:|---:|---|
| `000660` | `technical_only` | `shallow_regularized` | 0.4873 | 0.4758 | pass |
| `005930` | `technical_only` | `depth3_regularized` | 0.4687 | 0.4958 | pass |
| `NVDA` | `technical_only` | `baseline_depth3` | 0.4676 | 0.5223 | pass |

Binary는 3-class보다 F1이 보기 좋지만, long-short spread는 약하다.
특히 일부 모델은 F1이 올라도 `pred_up`과 `pred_down`의 평균 미래수익률 차이가 음수다.
따라서 "수익 전략 성능"이 아니라 "방향 분류 신호"로 제한해서 말해야 한다.

### 5. Binary SHAP

`src/binary_shap_analysis.py`는 binary `technical_only` 최종 후보를 다시 학습하고,
XGBoost `pred_contribs` 기반 contribution 산출물을 만든다.

산출물:

```text
outputs/results/binary_shap_analysis/selected_models.csv
outputs/results/binary_shap_analysis/global_shap_importance.csv
outputs/results/binary_shap_analysis/local_shap_cases.csv
outputs/results/binary_shap_analysis/local_case_selection.csv
outputs/results/binary_shap_analysis/*_binary_global_bar.png
outputs/results/binary_shap_analysis/*_binary_local.png
```

Binary XGBoost contribution은 기본적으로 `up` logit 기준이다.
문서화된 down 설명은 같은 contribution의 부호를 반전해 down class logit 기여도로 해석한다.

현재 binary SHAP 대상 모델:

| ticker | recipe | params |
|---|---|---|
| `000660` | `technical_only` | `shallow_regularized` |
| `005930` | `technical_only` | `depth3_regularized` |
| `NVDA` | `technical_only` | `baseline_depth3` |

### 6. Rolling validation

`src/binary_rolling_validation.py`는 binary `technical_only` 후보를 대상으로
2018~2024 model window 안에서 expanding train과 forward evaluation을 수행한다.
각 fold 사이에는 5-row embargo를 둔다.

Rolling summary:

| ticker | mean macro-F1 | std macro-F1 | mean balanced acc. | mean ROC-AUC | balance pass rate | mean long-short spread |
|---|---:|---:|---:|---:|---:|---:|
| `000660` | 0.5414 | 0.0397 | 0.5442 | 0.5693 | 1.00 | 0.0091 |
| `005930` | 0.4842 | 0.0425 | 0.5078 | 0.5354 | 1.00 | 0.0039 |
| `NVDA` | 0.4580 | 0.0549 | 0.4658 | 0.4629 | 1.00 | -0.0109 |

해석:

- SK하이닉스는 binary 전환 효과가 가장 방어 가능하다.
- 삼성전자는 성능이 중간 수준이지만 prediction balance와 rolling 안정성은 괜찮다.
- NVDA는 test/recent F1은 보기 좋지만 rolling ROC-AUC와 long-short spread가 약해, 최종 주장에서는 보수적으로 다뤄야 한다.

## 발표용 해석

권장 발표 구조:

1. 3-class 방향 예측을 먼저 시도했다.
2. 금융 시계열 5거래일 방향 예측은 노이즈와 regime shift가 커서 3-class macro-F1이 제한됐다.
3. Extended feature를 단순히 많이 넣는 방식은 성능을 일관되게 높이지 않았다.
4. Technical feature 중심 모델이 가장 안정적이었다.
5. Neutral 구간을 제거한 binary up/down 문제로 재정의하자 test macro-F1이 약 0.47~0.49 수준까지 개선됐다.
6. Rolling validation에서는 SK하이닉스와 삼성전자가 상대적으로 안정적이고, NVDA는 약한 후보로 남았다.
7. 다만 long-short spread가 강하지 않아, 투자 수익률 모델이 아니라 방향성 분류 실험으로 해석한다.

## 다음 작업 순서

1. 발표 최종 성능표는 binary `technical_only`와 rolling summary를 중심으로 구성한다.
2. SHAP 이미지는 `outputs/results/binary_shap_analysis/*_binary_global_bar.png`를 사용한다.
3. Local SHAP은 맞춘 사례와 틀린 사례를 섞어 보여주되, 인과가 아니라 예측 기여도라고 명시한다.
4. 3-class는 `technical_only`와 `proposed_daily_extended`의 대비를 ablation/한계 분석으로 제시한다.
5. Long-short spread가 음수인 NVDA는 최종 성능 주장보다 한계 사례로 다룬다.
