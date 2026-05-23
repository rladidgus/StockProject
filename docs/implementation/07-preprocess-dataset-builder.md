# 전처리 dataset builder 구현

## 목적

`data/metadata/current_raw_manifest.csv`가 가리키는 공식 raw run을 입력으로 사용해
종목별 core dataset을 생성한다. 이 dataset은 예측일 장마감 이후 시점에
알려진 정보만 feature로 사용해 향후 5거래일 수익률 방향을 예측하는 것을
기본 문제로 둔다.

공식 입력 run:

```text
20260520T155315691434Z-82034936
```

## 구현 원칙

- 팀 기획안의 Baseline vs Proposed 구조를 따른다.
- 첫 전처리 artifact는 core macro만 사용한다.
- 출력 dataset에는 당일 `open/high/low/close/volume/return_pct`를 포함하지 않는다.
- 모든 core macro feature는 prediction date 기준 as-of lag join으로 생성한다.
- core macro lag feature는 `t-1`, `t-3`, `t-5`를 생성한다.
- `us_rate`는 월간 FEDFUNDS 관측일을 그대로 known date로 쓰지 않고, 관측월 말 이후 5영업일 뒤부터 사용 가능하다고 보수 처리한다.
- 각 macro feature에는 `*_source_date`, `*_known_date` audit 컬럼을 함께 남긴다.
- 재생성 전 기존 core dataset/validation 산출물을 삭제해 stale 파일이 검증에 섞이지 않게 한다.
- label은 `target_5d_3class` 하나를 공식 target으로 사용한다.
- label 기준은 `future_5d_return = close[t+5] / close[t] - 1`이다.
- `prediction_date = t`는 장마감 이후 시점으로 정의한다.
- 종목별 train 구간에서만 30%/70% 분위수 threshold를 계산해 하락/중립/상승을 나눈다.
- 2018~2024 기본 모델 구간과 2025 이후 recent regime 구간을 `period_bucket`으로 분리한다.
- 2018~2024 기본 모델 구간 안에서 시간 순서로 split을 만들고, split 사이에는 5거래일 embargo를 둔다.

## 생성 파일

```text
data/processed/core/samsung_electronics_core_features.csv
data/processed/core/sk_hynix_core_features.csv
data/processed/core/nvidia_core_features.csv
data/processed/core/core_dataset_summary.csv
data/processed/core/feature_coverage.csv
data/processed/core/label_distribution_by_split.csv
data/processed/core/leakage_checks.csv
data/processed/core/label_policy_checks.csv
```

## 실행

```bash
python3 -m src.preprocess
python3 -m src.validate_preprocess
```

## Label 정책

공식 target은 3-class 분류다.

```text
prediction_date = t
future_5d_return = close[t+5] / close[t] - 1

target_5d_3class:
  0 = down     if future_5d_return <= train_30pct_threshold
  1 = neutral  if train_30pct_threshold < future_5d_return < train_70pct_threshold
  2 = up       if future_5d_return >= train_70pct_threshold
```

threshold는 종목별 train split의 `future_5d_return`만 사용해 계산한다.
고정 `±2%` 기준은 설명하기 쉬운 baseline으로는 가능하지만, 종목별 변동성이
다르기 때문에 공식 target에는 쓰지 않는다.

마지막 5개 거래일은 `close[t+5]`가 없으므로 target을 비워 둔다.
이 row는 supervised 학습과 지표 계산에는 사용하지 않고, 필요하면 live-style
예측 후보로만 다룬다.

## 도메인 안전장치

- split 방식: 2018~2024 구간 안에서 시간 순서 split을 생성하고, 5거래일 horizon의 label window가 다음 split과 겹치지 않게 embargo를 둔다.
- look-ahead 방지: core macro 전체를 `t-1`, `t-3`, `t-5` as-of lag로만 제공하고, 당일 가격/수익률 컬럼은 출력하지 않는다.
- lag 처리: PDF 기준 1/3/5일 lag feature를 생성한다.
- 검증 방식: 모든 model feature의 `source_date`/`known_date` audit 컬럼 존재와 known date cutoff를 검사한다.
- label 검증: 마지막 5거래일 pending target, supervised split target 존재, threshold 경계, split embargo 조건을 검사한다.
- 모델 투입 제외 피처: 당일 OHLCV/return, extended, financials, alpha, archive는 1차 dataset에 넣지 않는다.
- 발표 시 해석 주의: SHAP은 현실 인과가 아니라 모델 예측 기여도로 설명한다.

## 검증 결과

당일 OHLCV/return 제거, `t-1/t-3/t-5` as-of join, FEDFUNDS 보수 known date,
audit column, 5거래일 3-class label 정책, split embargo, regenerated artifact 정합성을 확인한다.
