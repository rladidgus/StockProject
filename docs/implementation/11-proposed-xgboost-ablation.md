# Proposed XGBoost extended ablation

## 목적

기술적 baseline과 core-only XGBoost 이후, 수집된 extended 피처가 실제로
추가 설명력을 주는지 피처군별로 비교한다. Extended를 한 번에 모두 넣지 않고
금리, 달러, 반도체 ETF, 월간 업황, 국내 수급 단위로 ablation한다.

## 실행

```bash
python3 -m src.proposed_xgboost
```

## 입력

- 전처리 dataset: `data/processed/core/*_core_features.csv`
- 공식 raw run: `data/metadata/current_raw_manifest.csv`가 가리키는 run
- target/split/label 정책은 기존 core 전처리 산출물을 그대로 사용한다.

## 실험군

| experiment | 추가 피처군 |
|---|---|
| `core_plus_rates` | `DGS10`, `DGS2`, `T10Y2Y` |
| `core_plus_broad_dollar` | `DTWEXBGS` |
| `core_plus_semi_etf` | `SMH`, `SOXX` |
| `core_plus_monthly_industry` | `IPG3344S`, `PCU334413334413P`, `A34SNO` |
| `core_plus_domestic_alpha` | 국내 외국인/기관 순매수, 국내 종목만 |
| `proposed_daily_extended` | 금리 + 달러 + 반도체 ETF |

모든 추가 피처는 `t-1`, `t-3`, `t-5` lag로만 투입한다.
월간 업황 지표는 월말 이후 5영업일 뒤부터 알려진 값으로 보수 처리한다.
국내 수급은 당일 값 직접 투입을 피하고 lag feature로만 사용한다.

## 산출물

```text
outputs/results/proposed_xgboost/model_metrics.csv
outputs/results/proposed_xgboost/feature_importance.csv
outputs/results/proposed_xgboost/feature_sets.csv
outputs/results/proposed_xgboost/ablation_summary.csv
```

## 1차 결과 해석

Extended를 추가해도 모든 종목에서 일관된 성능 개선이 나오지는 않는다.
특히 core-only에서 이미 일부 class 쏠림이 있었던 구간은 extended를 넣어도
macro F1이 낮게 유지되는 경우가 있다.

발표에서는 이 결과를 다음처럼 해석하는 것이 안전하다.

- Extended 피처는 수집 가치가 있지만 무조건 성능을 올리지는 않는다.
- 금리/달러/반도체 ETF처럼 중복 가능성이 큰 피처군은 조합별 검증이 필요하다.
- SHAP 해석은 최종 선택 모델에서 모델 예측 기여도로만 설명한다.

## 도메인 안전장치

- split 방식: 기존 전처리의 시간 순서 split과 5거래일 embargo를 그대로 사용한다.
- look-ahead 방지: 추가 피처는 prediction date 기준 lag/as-of join만 사용한다.
- lag 처리: extended 피처도 `t-1`, `t-3`, `t-5`로 통일한다.
- 모델 투입 제외 피처: HTML archive, 공시일 정렬이 불완전한 재무제표 원천, 뉴스 감성은 제외한다.
- 발표 시 해석 주의: SHAP 또는 feature importance는 현실 인과가 아니라 모델 예측 기여도다.
