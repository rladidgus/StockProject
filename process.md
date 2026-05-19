# 반도체 핵심주 모델링 진행 순서

## 현재 완료 상태
- PostgreSQL DB와 기본 테이블 생성 완료
- 삼성전자(`005930`) 주가 데이터 수집 완료
- SK하이닉스(`000660`) 주가 데이터 수집 완료
- 거시 지표 수집 완료
- zip 데이터 수집 패키지의 확장 시장/거시/FRED 수집 항목을 `market_features` 테이블과 `data/raw/fdr/` 백업 구조로 접목 완료
- 삼성전자와 SK하이닉스 외국인/기관 수급 데이터 수집 완료
- 전처리 산출물 생성 완료: `data/processed/samsung_preprocessed.csv`
- 전처리 산출물 생성 완료: `data/processed/sk_hynix_preprocessed.csv`
- 삼성전자와 SK하이닉스 모두 `macro_alpha` feature set 기준 모델링 완료
- 삼성전자와 SK하이닉스 모두 `macro_alpha` SHAP 분석 완료
- 날짜 구간별 지표 영향력 조회 스크립트 추가 완료: `src/interpret_period.py`

## 현재 기준 결론

- 예측 모델 성능은 아직 강하지 않다.
- 삼성전자 `macro_alpha` XGBoost Test AUC는 `0.4749`다.
- SK하이닉스 `macro_alpha` XGBoost Test AUC는 `0.5417`다.
- 따라서 현재 결과는 “실전 예측 모델”보다 “거시/수급 지표 영향력 탐색 모델”로 해석하는 것이 적절하다.
- 영향력 비교는 가격 레벨 피처를 제외한 `macro_alpha` 기준을 중심으로 진행한다.
- 원하는 날짜 구간을 지정해 그 기간의 상위 영향 지표를 조회할 수 있다.

## 1. 전처리 데이터 불러오기
- 입력 파일: `data/processed/samsung_preprocessed.csv`
- 이 파일에는 주가, 거시 지표, lag feature, 거시 변화율, 외국인/기관 수급, rolling 수급 feature, 모델 타겟이 포함되어 있다.
- 국내 시장 피처로 `kospi`, `kospi200`, KOSPI/KOSPI200 변화율, 삼성전자 대비 상대수익률을 추가했다.
- 수급 강도 피처로 외국인/기관 순매수를 거래량과 추정 거래대금 대비 비율로 변환한 feature를 추가했다.
- 예측 대상 컬럼은 `target_up`이다.
- `target_up`의 의미는 5거래일 후 삼성전자 종가가 오늘 종가보다 상승했는지 여부다.

## 2. Feature와 Target 분리
- Target: `target_up`
- Feature에서 제외할 컬럼:
  - `date`
  - `target_up`
  - `next_return_pct`
- 오늘 시점에 알 수 있는 `close`, `volume`, `return_pct`, 거시 lag feature, 거시 변화율, 수급 feature, rolling 수급 feature는 모델 입력으로 사용할 수 있다.
- KOSPI/KOSPI200은 국내 시장 피처이므로 미국 지표와 달리 전체 컬럼 `shift(1)` 대상에서 제외한다.

## 3. 시간 순서 기반 데이터 분할
- 주가 데이터는 시계열이므로 랜덤 셔플을 사용하지 않는다.
- 분할 기준:
  - Train: 앞 70%
  - Validation: 다음 15%
  - Test: 마지막 15%
- Test set은 최종 평가에만 사용한다.

## 4. XGBoost 모델 학습
- 1차 모델은 Optuna 튜닝 없이 기본 XGBoost로 먼저 학습한다.
- 현재 삼성전자 모델은 지표 영향력 분석에 맞춰 5거래일 후 상승 타겟과 Proposed feature 63개로 학습했다.
- 기본 모델 예시:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)
```

- 이후 Validation ROC-AUC를 기준으로 하이퍼파라미터 튜닝을 추가한다.
- 1차 Validation 결과:
  - Accuracy: `0.5137`
  - Precision: `0.8000`
  - Recall: `0.0315`
  - F1-score: `0.0606`
  - ROC-AUC: `0.5979`
- Optuna 20 trials AUC 튜닝 및 threshold 조정 후 Proposed Validation 결과:
  - Accuracy: `0.5827`
  - Precision: `0.5505`
  - Recall: `0.8651`
  - F1-score: `0.6728`
  - ROC-AUC: `0.6296`
- 튜닝 모델은 Validation ROC-AUC를 기준으로 선택하고, threshold는 Validation F1-score를 기준으로 후처리한다.
- 고도화 단계에서 `ma5`, `ma10`, `ma20`, `rsi`, `bb_upper`, `bb_lower` 기술적 지표를 추가했다.
- Baseline, Proposed, Proposed+Technical feature set을 비교하고 Validation ROC-AUC 기준으로 하이퍼파라미터를 튜닝했다.
- XGBoost + SHAP 해석을 보조 검증하기 위해 Logistic Regression과 feature-target 상관분석을 추가했다.

## 5. 성능 평가
- Validation과 Test에서 아래 지표를 확인한다.
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion Matrix
- 이 프로젝트에서는 단순 Accuracy보다 F1-score, Recall, ROC-AUC를 더 중요하게 본다.
- 최종 모델은 튜닝된 best parameter를 사용해 Train + Validation 구간으로 재학습한 뒤 Test set에서 1회 평가한다.
- 삼성전자 5거래일 후 상승 타겟 기준 Test 결과 중 지표 영향력 분석용 최종 후보:
  - Model: `xgboost_tuned_train_val_proposed`
  - Accuracy: `0.4510`
  - Precision: `0.4397`
  - Recall: `0.9107`
  - F1-score: `0.5930`
  - ROC-AUC: `0.4949`
  - Threshold: `0.18`
- 주의: 수급 강도 피처 추가 후 Proposed Test AUC는 `0.4745`에서 `0.4949`로 소폭 회복됐지만 여전히 0.5 근처다.
- 참고: Test AUC만 보면 Baseline 모델이 `0.5600`으로 가장 높지만, 기술적 지표 중심이라 이 프로젝트의 거시 지표 영향력 분석 목적에는 덜 적합하다.
- 참고: Proposed+Technical 모델은 Test AUC `0.5365`로 개선됐지만, 기술적 지표가 섞이므로 순수 거시/수급 영향력 해석용 모델과는 구분해서 본다.

## 6. 보조 검증
- 목적은 XGBoost SHAP에서 중요하게 나온 지표가 단순 선형 모델이나 상관분석에서도 어느 정도 확인되는지 보는 것이다.
- Logistic Regression은 `proposed` feature set으로 학습했다.
- Feature-target 상관분석은 Train 구간에서만 계산해 미래 Test 데이터가 해석에 섞이지 않도록 했다.
- Logistic Regression Test 결과:
  - Accuracy: `0.4549`
  - Precision: `0.4426`
  - Recall: `0.9286`
  - F1-score: `0.5994`
  - ROC-AUC: `0.4724`
  - Threshold: `0.22`
- 상관분석 상위 피처:
  - `institution_net_buy_20d_sum`
  - `kospi`
  - `kospi200`
  - `kospi_lag5`
  - `kospi_lag1`
- Logistic Regression 계수 절댓값 상위 피처:
  - `usd_krw_lag5`
  - `nasdaq_lag1`
  - `sox_lag5`
  - `usd_krw_lag1`
  - `foreign_net_buy_to_trading_value`
- 보조 검증 결과도 Test AUC가 낮기 때문에, 현재 삼성전자 모델은 예측 모델보다는 지표 영향력 탐색 모델에 가깝다.

## 7. 결과 저장
- 모델 성능 결과 저장 위치: `outputs/results/model_comparison.csv`
- 상관분석 결과 저장 위치: `outputs/results/feature_target_correlation.csv`
- Logistic Regression 계수 저장 위치: `outputs/results/logistic_coefficients.csv`
- 삼성전자 1차 결과는 아래 형태로 저장한다.

```csv
ticker,model,accuracy,precision,recall,f1,auc,threshold
005930,xgboost_tuned_train_val_proposed,0.4470588235294118,0.4312796208530806,0.8125,0.5634674922600619,0.4744630369630369,0.11
005930,xgboost_tuned_train_val_proposed,0.4509803921568627,0.4396551724137931,0.9107142857142856,0.5930232558139535,0.4948801198801199,0.18
005930,logistic_regression_proposed,0.4549019607843137,0.4425531914893617,0.9285714285714286,0.5994236311239193,0.4724025974025974,0.22
```

## 전체 흐름 요약
```text
CSV 로드
→ X/y 분리
→ 시간순 Train/Validation/Test 분할
→ XGBoost 학습
→ 성능 평가
→ 결과 저장
```

## 확장 데이터 수집

zip 수집 패키지의 넓은 수집 항목은 기존 `macro_features`를 깨지 않도록 별도 long-form 테이블인 `market_features`에 저장한다.

```bash
python src/db.py
python src/collect.py --extended-market --skip-core
```

- DB 저장 위치: `market_features(date, feature_name, feature_group, source, value)`
- 원본 CSV 백업: `data/raw/fdr/prices`, `semi_indices`, `global_indices`, `fx_rates`, `rates`, `futures`, `fred`
- 수집 메타데이터: `data/metadata/extended_collection_summary.csv`
- 포함 항목: 반도체 ETF/지수, 글로벌 지수, 환율, 미국 금리/달러, 원자재 선물, FRED 거시·반도체 지표

## 다음 작업
- 삼성전자 SHAP 산출물 생성 완료:
  - `outputs/figures/samsung_global_bar.png`
  - `outputs/figures/samsung_global_beeswarm.png`
  - `outputs/figures/samsung_local_ai_rally_2024.png`
- 삼성전자 SHAP DB 저장 완료:
  - `shap_results`: 16,065행
  - 기간: `2023-12-06 ~ 2024-12-20`
  - Feature 수: 63개
- Test set 기준 평균 절대 SHAP 상위 피처:
  - `nasdaq_lag5`
  - `nasdaq_lag3`
  - `us_rate_lag5`
  - `foreign_net_buy_5d_sum`
  - `institution_net_buy_20d_to_trading_value_20d`
- 삼성전자 단일 파이프라인을 기준으로 SK하이닉스와 NVDA로 확장한다.

## SK하이닉스 확장 결과

- 티커: `000660`
- 전처리 산출물: `data/processed/sk_hynix_preprocessed.csv`
- 전처리 데이터:
  - 기간: `2018-01-29 ~ 2024-12-20`
  - 행 수: 1,697행
  - 컬럼 수: 72컬럼
  - Proposed feature 수: 63개
- 수집 완료 데이터:
  - 주가 데이터: `stock_prices`
  - 거시 지표: `macro_features`
  - 외국인/기관 수급: `alpha_features`

### SK하이닉스 모델 성능

- Logistic Regression proposed Test:
  - Accuracy: `0.5804`
  - Precision: `0.6104`
  - Recall: `0.6667`
  - F1-score: `0.6373`
  - ROC-AUC: `0.5842`
  - Threshold: `0.19`
- XGBoost proposed Test:
  - Accuracy: `0.4902`
  - Precision: `0.6000`
  - Recall: `0.2340`
  - F1-score: `0.3367`
  - ROC-AUC: `0.5181`
  - Threshold: `0.26`
- XGBoost proposed_plus Test:
  - Accuracy: `0.5529`
  - Precision: `0.5730`
  - Recall: `0.7518`
  - F1-score: `0.6503`
  - ROC-AUC: `0.5617`
  - Threshold: `0.10`

### SK하이닉스 보조 검증

- 상관분석 상위 피처:
  - `institution_net_buy_20d_to_trading_value_20d`
  - `close`
  - `low`
  - `high`
  - `open`
- Logistic Regression 계수 절댓값 상위 피처:
  - `close`
  - `institution_net_buy_20d_to_trading_value_20d`
  - `open`
  - `us_rate_lag5`
  - `institution_net_buy_20d_sum`

### SK하이닉스 SHAP 결과

- SHAP DB 저장:
  - `shap_results`: 16,065행
  - 기간: `2023-12-06 ~ 2024-12-20`
  - Feature 수: 63개
- SHAP 산출물:
  - `outputs/figures/sk_hynix_global_bar.png`
  - `outputs/figures/sk_hynix_global_beeswarm.png`
  - `outputs/figures/sk_hynix_local_ai_rally_2024.png`
- Test set 기준 평균 절대 SHAP 상위 피처:
  - `close`
  - `low`
  - `high`
  - `vix_lag5`
  - `nasdaq_return_3d`
  - `foreign_net_buy_20d_to_trading_value_20d`
  - `institution_net_buy_20d_sum`

### 현재 해석

- SK하이닉스는 삼성전자보다 Logistic Regression 보조 검증 성능이 좋다.
- 수급 강도 피처인 `institution_net_buy_20d_to_trading_value_20d`가 상관분석과 Logistic 계수에서 모두 상위권에 있다.
- SHAP에서는 가격 레벨 피처(`close`, `low`, `high`)가 크게 잡혀 있어, 다음 단계에서는 가격 레벨 의존도를 줄이고 거시/수급 중심 feature set을 따로 비교할 필요가 있다.
- 다음 확장 대상은 NVDA다.

## 거시/수급 중심 Feature Set 추가 결과

### 추가 목적

- 기존 `proposed` 모델은 `open`, `high`, `low`, `close` 같은 주가 레벨 피처의 영향이 크게 나타났다.
- 프로젝트의 핵심 질문은 “거시경제 지표와 수급 지표가 종목별로 어떤 영향을 주는가”이므로, 가격 레벨 의존도를 줄인 별도 feature set이 필요하다.
- 이를 위해 `macro_alpha` feature set을 추가했다.

### `macro_alpha` 기준

- 포함:
  - 거시 지표: `nasdaq`, `sox`, `vix`, `us_rate`, `usd_krw`
  - 국내 시장 지표: `kospi`, `kospi200`
  - 거시/시장 lag 및 변화율 피처
  - 수급 피처: 외국인/기관 순매수, rolling sum/mean
  - 수급 강도 피처: 거래대금 대비 외국인/기관 순매수 비율
  - `volume`, `return_pct`
- 제외:
  - `open`
  - `high`
  - `low`
  - `close`
  - `trading_value`
- Feature 수:
  - 58개

### 삼성전자 `macro_alpha` 결과

- Logistic Regression Test:
  - Accuracy: `0.4706`
  - Precision: `0.4519`
  - Recall: `0.9643`
  - F1-score: `0.6154`
  - ROC-AUC: `0.4750`
  - Threshold: `0.22`
- XGBoost Tuned Test:
  - Accuracy: `0.4392`
  - Precision: `0.4351`
  - Recall: `0.9286`
  - F1-score: `0.5926`
  - ROC-AUC: `0.4749`
  - Threshold: `0.18`

### SK하이닉스 `macro_alpha` 결과

- Logistic Regression Test:
  - Accuracy: `0.5647`
  - Precision: `0.5750`
  - Recall: `0.8156`
  - F1-score: `0.6745`
  - ROC-AUC: `0.5286`
  - Threshold: `0.31`
- XGBoost Tuned Test:
  - Accuracy: `0.5451`
  - Precision: `0.5532`
  - Recall: `0.9220`
  - F1-score: `0.6915`
  - ROC-AUC: `0.5417`
  - Threshold: `0.13`

### 현재 해석

- 삼성전자는 `macro_alpha` 기준 ROC-AUC가 0.5보다 낮아 예측력은 아직 약하다.
- SK하이닉스는 `macro_alpha` 기준 XGBoost ROC-AUC가 `0.5417`로 삼성전자보다 높다.
- 두 종목 모두 Recall은 높지만 Precision이 낮아, 상승 신호를 넓게 잡는 대신 오탐이 많은 구조다.
- 영향력 분석 목적에서는 `proposed`와 `macro_alpha`를 함께 비교하는 것이 좋다.

## `macro_alpha` 기준 SHAP 분석 결과

### 저장 구조 변경

- `shap_results` 테이블에 `feature_set` 컬럼을 추가했다.
- 이제 `proposed`와 `macro_alpha` SHAP 결과를 같은 테이블에 함께 저장할 수 있다.
- 저장 기준:
  - `ticker`
  - `feature_set`
  - `date`
  - `feature_name`

### 삼성전자 `macro_alpha` SHAP

- DB 저장:
  - `shap_results`: 14,790행
  - Ticker: `005930`
  - Feature set: `macro_alpha`
  - Feature 수: 58개
- 산출물:
  - `outputs/figures/samsung_macro_alpha_global_bar.png`
  - `outputs/figures/samsung_macro_alpha_global_beeswarm.png`
  - `outputs/figures/samsung_macro_alpha_local_ai_rally_2024.png`
- 평균 절대 SHAP 상위 피처:
  - `nasdaq_lag5`
  - `nasdaq_lag3`
  - `foreign_net_buy_5d_sum`
  - `us_rate_lag5`
  - `institution_net_buy_20d_to_trading_value_20d`
  - `sox_lag5`
  - `vix_lag5`
  - `us_rate`
  - `kospi200_return_3d`
  - `sox_lag1`
- Sanity check:
  - `us_rate`: 예상 방향 `negative`, 실제 방향 `negative`
  - `vix`: 예상 방향 `negative`, 실제 방향 `negative`

### SK하이닉스 `macro_alpha` SHAP

- DB 저장:
  - `shap_results`: 14,790행
  - Ticker: `000660`
  - Feature set: `macro_alpha`
  - Feature 수: 58개
- 산출물:
  - `outputs/figures/sk_hynix_macro_alpha_global_bar.png`
  - `outputs/figures/sk_hynix_macro_alpha_global_beeswarm.png`
  - `outputs/figures/sk_hynix_macro_alpha_local_ai_rally_2024.png`
- 평균 절대 SHAP 상위 피처:
  - `usd_krw_lag5`
  - `nasdaq_lag1`
  - `kospi200_return_3d`
  - `sox`
  - `foreign_net_buy_5d_sum`
  - `nasdaq_lag3`
  - `institution_net_buy_20d_sum`
  - `foreign_net_buy`
  - `nasdaq_return_3d`
  - `vix_lag1`
- Sanity check:
  - `us_rate`: 예상 방향 `negative`, 실제 방향 `negative`
  - `vix`: 예상 방향 `negative`, 실제 방향 `negative`

### 종목별 비교 해석

- 삼성전자는 나스닥 지연 피처(`nasdaq_lag5`, `nasdaq_lag3`)와 금리/변동성 지표가 상위권에 있다.
- SK하이닉스는 환율(`usd_krw_lag5`), 나스닥 단기 지연 피처(`nasdaq_lag1`), KOSPI200 3일 수익률, SOX가 상위권에 있다.
- 두 종목 모두 `foreign_net_buy_5d_sum`이 상위권에 있어 외국인 단기 수급이 공통적으로 중요한 피처로 잡힌다.
- 삼성전자는 글로벌 지수의 지연 효과와 금리 민감도가 상대적으로 두드러지고, SK하이닉스는 환율·반도체 지수·국내 시장 흐름에 더 민감하게 나타난다.
- 다음 단계는 NVDA를 같은 방식으로 수집, 전처리, 모델링, SHAP 분석까지 확장하는 것이다.

## 날짜 구간별 지표 영향력 조회

### 목적

- 전체 Test 기간 평균 SHAP만 보면 특정 시기별 시장 환경 차이를 보기 어렵다.
- 그래서 사용자가 원하는 날짜 구간을 입력하면, 해당 기간에 한정해 평균 절대 SHAP 기준 상위 지표를 조회하는 스크립트를 추가했다.
- 예를 들어 2024년 AI 반도체 랠리 구간, 2022년 금리 인상 구간처럼 특정 기간만 잘라서 비교할 수 있다.

### 실행 파일

- `src/interpret_period.py`

### 사용 예시

```bash
python src/interpret_period.py \
  --ticker 005930 \
  --feature-set macro_alpha \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --top-n 10
```

```bash
python src/interpret_period.py \
  --ticker 000660 \
  --feature-set macro_alpha \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --top-n 10
```

### 출력 컬럼

- `feature_name`: 지표 이름
- `mean_abs_shap`: 해당 구간에서의 평균 절대 SHAP 값
- `mean_shap`: SHAP 평균값
- `dominant_direction`: 평균적으로 상승 예측에 양의 기여인지 음의 기여인지
- `positive_shap_ratio`: 해당 지표가 양의 SHAP을 가진 비율
- `mean_feature_value`: 해당 구간의 평균 지표값

### 2024년 구간 예시 결과

- 삼성전자(`005930`) 상위 지표:
  - `nasdaq_lag5`
  - `nasdaq_lag3`
  - `foreign_net_buy_5d_sum`
  - `us_rate_lag5`
  - `institution_net_buy_20d_to_trading_value_20d`
  - `sox_lag5`
  - `vix_lag5`
  - `us_rate`
  - `kospi200_return_3d`
  - `sox_lag1`
- SK하이닉스(`000660`) 상위 지표:
  - `usd_krw_lag5`
  - `nasdaq_lag1`
  - `sox`
  - `kospi200_return_3d`
  - `foreign_net_buy_5d_sum`
  - `nasdaq_lag3`
  - `foreign_net_buy`
  - `institution_net_buy_20d_sum`
  - `nasdaq_return_3d`
  - `vix_lag1`

### 저장 파일

- `outputs/results/period_feature_importance_samsung_macro_alpha_2024-01-01_2024-12-31.csv`
- `outputs/results/period_feature_importance_sk_hynix_macro_alpha_2024-01-01_2024-12-31.csv`

### 해석 포인트

- 삼성전자는 2024년 구간에서 나스닥 지연 피처와 외국인 5일 누적 수급이 크게 나타났다.
- SK하이닉스는 2024년 구간에서 환율, 나스닥 단기 지연 피처, SOX, KOSPI200 3일 수익률이 크게 나타났다.
- 이 방식으로 원하는 구간을 바꿔가며 종목별 민감 지표 변화를 비교할 수 있다.
