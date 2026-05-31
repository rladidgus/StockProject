# Proposal completion audit

## 기준 문서

기준 문서:

```text
/Users/gumasaje/d/Projects/머신러닝 제안서.pdf
```

제안서의 핵심 목표는 다음으로 정리된다.

1. 삼성전자, SK하이닉스, 엔비디아의 주가 방향을 분류한다.
2. 최근 5~7년의 주가, 거시, 미시, 수급 데이터를 사용한다.
3. `t-1`, `t-3`, `t-5` lag feature와 look-ahead bias 방지를 적용한다.
4. XGBoost와 SHAP을 사용한다.
5. 단순 예측이 아니라 금리, 환율, 변동성, 수급 등 어떤 요인이 하락 또는 상승 판단에 기여했는지 설명한다.
6. Baseline과 proposed 모델을 비교한다.

## 현재 달성 상태

| 제안서 요구 | 구현/산출물 | 판정 |
|---|---|---|
| 3개 종목 방향 분류 | `src/binary_xgboost.py`, `outputs/results/binary_xgboost/*` | 달성 |
| 5~7년 학습 데이터 | 2018~2024 model window, 2025 recent regime | 달성 |
| 시간 기반 train/validation/test | `src/preprocess.py`, `src/validate_preprocess.py` | 달성 |
| look-ahead 방지 | lag/as-of join, embargo, `leakage_checks.csv` 통과 | 달성 |
| XGBoost | baseline/proposed/binary XGBoost 실험 | 달성 |
| SHAP/XAI | `src/binary_shap_analysis.py`, `outputs/results/binary_shap_analysis/*` | 달성 |
| 거시/미시/수급 XAI | `outputs/results/proposal_xai_risk_analysis/*` | 달성 |
| Baseline vs proposed 비교 | technical-only, proposed, all-guarded 비교 산출 | 달성 |
| 뉴스 감성 | 수집/정제/모델 투입 없음 | 미달성 |
| 분기 실적 모델 투입 | 원천 일부 수집, 최종 모델 투입 제외 | 부분 달성 |

## 모델을 둘로 나눈 이유

제안서 원안은 거시/미시/수급 지표로 원인을 설명하는 모델을 목표로 했다.
하지만 실제 검증에서는 extended feature를 많이 넣은 모델이 가장 높은 예측 성능을 내지 않았다.

따라서 최종 구조를 다음처럼 나눈다.

| 용도 | 모델/산출물 | 사용 이유 |
|---|---|---|
| 최종 예측 성능 | binary `technical_only` | rolling 성능과 prediction balance가 가장 안정적 |
| 제안서 정합성/XAI 리스크 분석 | binary `technical_all_guarded` | 거시, 수급, 업황 feature가 포함되어 원인 설명 가능 |

이 구조는 제안서 목표를 더 정직하게 만족한다.
성능이 좋은 모델과 원인 설명 범위가 넓은 모델을 억지로 하나로 합치지 않고,
각 모델의 역할과 한계를 분리한다.

## 최종 예측 모델 성능

Binary `technical_only` rolling validation:

| ticker | params | mean macro-F1 | mean ROC-AUC | balance pass rate | mean long-short spread |
|---|---|---:|---:|---:|---:|
| `000660` | `shallow_regularized` | 0.5414 | 0.5693 | 1.00 | 0.0091 |
| `005930` | `depth3_regularized` | 0.4842 | 0.5354 | 1.00 | 0.0039 |
| `NVDA` | `baseline_depth3` | 0.4580 | 0.4629 | 1.00 | -0.0109 |

해석:

- SK하이닉스는 최종 발표 주력 후보로 가장 방어 가능하다.
- 삼성전자는 중간 수준의 방향성 신호가 있다.
- NVDA는 binary F1은 나오지만 ROC-AUC와 long-short spread가 약해 한계 사례로 다룬다.

## 제안서 정합성 모델

Binary `technical_all_guarded`는 다음 계열을 포함한다.

- technical feature
- core macro: NASDAQ, SOX, VIX, 기준금리, USD/KRW
- rates: 미국 10년물, 2년물, 장단기 금리차
- dollar: broad dollar index
- semi ETF: SMH, SOXX
- monthly industry: 반도체 산업생산, PPI, 전자제품 신규주문
- domestic alpha: 국내 외국인/기관 순매수, 국내 종목 한정

Rolling validation:

| ticker | params | mean macro-F1 | mean ROC-AUC | balance pass rate | mean long-short spread |
|---|---|---:|---:|---:|---:|
| `000660` | `depth4_slow` | 0.4010 | 0.5594 | 0.25 | 0.0064 |
| `005930` | `depth3_conservative` | 0.4345 | 0.5089 | 0.75 | 0.0106 |
| `NVDA` | `depth2_fast` | 0.3996 | 0.4552 | 1.00 | -0.0354 |

이 모델은 최종 성능 모델로 주장하지 않는다.
대신 제안서의 "금리, 환율, 변동성, 수급 요인이 예측에 얼마나 기여했는가"를 설명하는 XAI 리스크 분석 레이어로 사용한다.

## Proposal-aligned SHAP 결과

산출물:

```text
outputs/results/proposal_xai_risk_analysis/selected_models.csv
outputs/results/proposal_xai_risk_analysis/global_shap_importance.csv
outputs/results/proposal_xai_risk_analysis/local_shap_cases.csv
outputs/results/proposal_xai_risk_analysis/local_case_selection.csv
outputs/results/proposal_xai_risk_analysis/*_binary_global_bar.png
outputs/results/proposal_xai_risk_analysis/*_binary_local.png
```

Test split global SHAP 상위 feature 예시:

| ticker | 주요 feature |
|---|---|
| `000660` | `tech_macd_signal_ratio`, `us_rate_t_minus_5`, `us_2y_treasury_fred_t_minus_1`, `us_10y_treasury_fred_t_minus_5`, `broad_dollar_index_t_minus_1` |
| `005930` | `smh_t_minus_1`, `tech_macd_signal_ratio`, `us_2y_treasury_fred_t_minus_1`, `us_10y_treasury_fred_t_minus_1`, `soxx_t_minus_1` |
| `NVDA` | `us_10y_treasury_fred_t_minus_1`, `sox_t_minus_5`, `broad_dollar_index_t_minus_3`, `us_10y_treasury_fred_t_minus_3`, `computers_electronic_products_new_orders_t_minus_5` |

이 결과를 통해 제안서의 XAI 설명 목표는 달성 가능하다.
단, SHAP은 인과가 아니라 모델 예측 기여도라고 명시해야 한다.

## 남은 미달 항목

### 뉴스 감성

제안서에는 DeepSearch API 기반 뉴스 감성 지수가 포함되어 있다.
현재 프로젝트에는 뉴스 감성 수집, 정제, 시점 정렬, 라벨링이 없다.
이 항목을 무리하게 추가하면 재현성과 look-ahead 검증이 흔들릴 가능성이 크다.

발표에서는 다음처럼 처리한다.

> 뉴스 감성은 제안서 후보였으나, 과거 시점 재현성과 시점 정렬 검증이 부족해 최종 모델에서는 제외했다.

### 분기 실적

국내 분기 실적과 NVIDIA SEC 원천은 일부 수집되어 있다.
다만 공시일 기준 `effective_from` 정렬이 완전히 검증된 모델 feature로 승격되지는 않았다.
따라서 최종 모델에는 넣지 않고, 수집 완료/후속 과제로 설명한다.

## 최종 판정

제안서 원안의 모든 후보 feature를 모델에 넣은 것은 아니다.
하지만 제안서의 핵심인 다음 항목은 달성했다.

- 세 종목 방향성 분류
- XGBoost 기반 예측 모델
- 시간 기반 검증과 look-ahead 방지
- baseline/proposed 비교
- SHAP 기반 예측 기여도 설명
- 거시/수급/업황 feature가 포함된 proposal-aligned XAI 리스크 분석

따라서 발표 기준 판정은 다음과 같다.

> 원안 후보 중 뉴스 감성과 공시 기반 실적 feature는 제외했지만, 핵심 목표인 반도체 핵심주 방향성 예측과 XAI 기반 영향 요인 분석은 달성했다.

100%라고 말하려면 "제안서의 모든 후보 feature를 넣었다"가 아니라,
"검증 가능한 범위에서 제안서의 핵심 기능을 구현했고, 검증 불가능한 후보는 제외 사유를 명시했다"는 의미로 정리해야 한다.
