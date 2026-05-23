# 전처리 dataset builder 1차 구현

## 목적

`data/metadata/current_raw_manifest.csv`가 가리키는 공식 raw run을 입력으로 사용해
종목별 core dataset을 생성한다.

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
- `±1%` label 컬럼은 생성하되, 중립 구간 처리 방식은 다음 단계에서 확정한다.
- 2018~2024 기본 모델 구간과 2025 이후 recent regime 구간을 `period_bucket`으로 분리한다.
- 2018~2024 기본 모델 구간 안에서 시간 순서로 `train` 70%, `validation` 15%, `test` 15%를 부여한다.

## 생성 파일

```text
data/processed/core/samsung_electronics_core_features.csv
data/processed/core/sk_hynix_core_features.csv
data/processed/core/nvidia_core_features.csv
data/processed/core/core_dataset_summary.csv
data/processed/core/feature_coverage.csv
data/processed/core/label_distribution_by_split.csv
data/processed/core/leakage_checks.csv
```

## 실행

```bash
python3 -m src.preprocess
python3 -m src.validate_preprocess
```

## 도메인 안전장치

- split 방식: 2018~2024 구간 안에서 시간 순서 70/15/15 split을 생성한다.
- look-ahead 방지: core macro 전체를 `t-1`, `t-3`, `t-5` as-of lag로만 제공하고, 당일 가격/수익률 컬럼은 출력하지 않는다.
- lag 처리: PDF 기준 1/3/5일 lag feature를 생성한다.
- 검증 방식: 모든 model feature의 `source_date`/`known_date` audit 컬럼 존재와 known date cutoff를 검사한다.
- 모델 투입 제외 피처: 당일 OHLCV/return, extended, financials, alpha, archive는 1차 dataset에 넣지 않는다.
- 발표 시 해석 주의: SHAP은 현실 인과가 아니라 모델 예측 기여도로 설명한다.

## 검증 결과

당일 OHLCV/return 제거, `t-1/t-3/t-5` as-of join, FEDFUNDS 보수 known date,
audit column, regenerated artifact 정합성을 확인했다.
