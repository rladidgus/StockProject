# 구현 내용

## 이번 구현에서 바꾼 점

### 1. 수집 목표를 핵심 피처 중심으로 축소

기존 `data_collection`처럼 확장 지표를 대량 수집하는 대신,
제안서에 직접 연결되는 핵심 피처만 1차 수집 대상으로 삼았다.

이 선택은 "나머지 피처가 불필요하다"는 뜻이 아니다.
이번 구현에서는 **공식 기준선 모델에 먼저 투입할 core 피처를 고정**하는 데 목적이 있다.

### 2. `src/collect.py`를 수집 단계 전용 엔트리포인트로 재구성

새 `src/collect.py`는 다음만 담당한다.

- 종목 주가 수집
- 핵심 거시 피처 수집
- 국내 수급 데이터 수집
- raw CSV 저장
- 메타데이터 기록

수익률(`return_pct`) 같은 파생 컬럼은 raw 수집 단계에서 만들지 않고,
후속 전처리 단계로 넘긴다.

### 3. 수집 결과 메타데이터를 별도 기록

`data/metadata/collection_summary.csv`

- 무엇을 수집했는지
- 어느 소스에서 가져왔는지
- 몇 행이 저장되었는지
- 파일 경로가 어디인지

를 남긴다.

`data/metadata/source_catalog.csv`

- 각 피처가 어떤 현실 세계 의미를 가지는지
- 어떤 소스/심볼을 쓰는지

를 남긴다.

`data/metadata/current_raw_manifest.csv`

- 전처리 브랜치가 소비해야 할 최신 adopted raw run의 `run_id`
- 해당 run의 raw 경로
- 해당 run의 metadata 경로

를 안정적인 포인터로 남긴다.

여기에는 개별 `macro_series`뿐 아니라,
전처리 브랜치가 실제로 공식 입력으로 소비해야 하는
aggregate artifact `core_macro_features`도 함께 명시한다.

### 4. 운영 리스크를 줄이기 위한 보정 반영

- 수집 중 일부 소스가 실패해도 `collection_summary.csv`가 남도록 `finally` 경로에서 메타데이터를 항상 기록
- NVDA는 `yfinance`의 종료일 배타 특성을 반영해 종료일을 하루 보정
- NVDA는 국내 종목과 raw 의미를 맞추기 위해 `auto_adjust=False`로 비조정 OHLC를 고정
- `us_rate`는 FRED CSV를 직접 조회해 로컬 SSL/FDR urllib 문제로 기준금리 수집이 막히지 않게 한다
- 원시 가격과 분할 이벤트를 섞지 않기 위해 `return_pct`는 collect-only 단계에서 생성하지 않음
- PyKRX 수급 데이터는 금액 단위가 드러나게 컬럼 계약을 명시
- 국내 수급(alpha)은 수집 시도는 유지하되, KRX credential 의존성 때문에 official baseline 성공 필수 조건에서는 제외
- 0행 결과는 `success`가 아니라 `empty` 상태로 남겨 운영 신호를 분리
- core macro 묶음은 `NASDAQ`, `SOX`, `VIX`, `us_rate`, `USD/KRW` 필수 컬럼이 하나라도 빠지면 `success`로 기록하지 않음
- core macro 필수 컬럼은 존재만이 아니라 실제 값이 하나 이상 있어야 성공으로 본다
- core macro bundle은 교집합으로 축소하지 않고 daily raw grid를 그대로 보존한다
- core macro bundle의 date grid는 extended macro가 아니라 core series 집합만 기준으로 만든다
- core macro bundle의 explicit daily grid는 `nasdaq`, `sox`, `vix`, `usd_krw`의 날짜 집합으로 정의하고, `us_rate`는 이 grid에 얹는 저빈도 시리즈로 취급한다
- `us_rate` 원자료 관측일은 daily grid 밖에 있더라도 core bundle에 보존해 전처리 단계에서 forward-fill할 수 있게 한다
- 월별/저빈도 시리즈의 forward-fill과 최종 날짜 정렬 책임은 전처리 단계에서 수행한다
- `KOSPI`, `KOSPI200`는 수집하되 extended macro로 취급
- 각 실행의 `run_id`, `start/end/skip flags`를 메타데이터에 함께 남겨 raw 파일 provenance를 보존
- `collection_summary.csv`는 append 방식으로 실행 이력을 누적
- raw CSV는 고정 파일명을 덮어쓰지 않고 `data/raw/runs/<run_id>/...` 아래에 실행별로 보존
- core macro가 실패하더라도 개별 수집에 성공한 macro 시리즈 파일은 실행별 raw 아티팩트로 남김
- `core_macro_features.csv`에는 core 컬럼만 저장하고 extended macro는 섞지 않음
- macro 메타데이터는 aggregate 결과만이 아니라 개별 series 단위로도 남긴다
- macro 전체가 empty여도 개별 `macro_series` 상태를 남긴다
- macro 개별 파일 저장 실패도 나머지 series/aggregate 메타데이터를 남긴 채 failed 상태로 기록한다
- 수집 엔트리포인트에서 불필요한 `dotenv` 의존성을 제거
- `pykrx` import는 alpha 수집 함수 내부로 이동해, `--skip-alpha` 또는 alpha 미사용 환경에서도 core stock/macro 수집이 가능하도록 분리
- `requirements.txt`는 collect-only 실행 의존성만 남기고 모델링/SHAP/DB 적재 잔여 의존성은 제거
- `source_catalog.csv`는 최신본을 유지하되, run별 복사본도 `data/metadata/runs/<run_id>/source_catalog.csv`로 보존
- canonical `source_catalog.csv`는 run 성공 여부와 무관하게 항상 갱신해, 전처리 브랜치가 정적 계약을 고정 경로에서 읽을 수 있게 유지
- canonical `source_catalog.csv`는 정적 계약 파일이므로 `run_id`를 포함하지 않고, run별 provenance는 `data/metadata/runs/<run_id>/source_catalog.csv`에서만 관리
- 채택 가능한 non-debug 성공 run에서만 `current_raw_manifest.csv`를 갱신해, 전처리 브랜치가 공식 raw bundle을 결정적으로 선택할 수 있게 한다
- `current_raw_manifest.csv` 갱신 전 core 주가와 core macro series의 요청 기간 커버리지를 검증한다
- `--skip-raw`는 로컬 점검용 옵션일 뿐, 공식 raw 기준 run으로는 성공 처리하지 않는다
- `--skip-raw` 실행에서는 개별 artifact도 `success`가 아니라 `skipped`로 기록해 raw 파일 부재를 artifact 수준에서 드러낸다
- `--skip-raw` 실행의 `core_artifact_validation`은 실제 source outage와 분리해 해석한다. debug run에서는 artifact `skipped`를 누락으로 간주하지 않고, 별도의 `raw_output_contract` 실패가 baseline 부적합성을 나타낸다
- `core_macro_features`의 provenance symbol은 `IXIC,^SOX,VIX,FRED:FEDFUNDS,USD/KRW`로 상태와 무관하게 일관되게 기록
- `core_macro_features`의 provenance source는 run summary와 source catalog 모두에서 `internal_bundle`로 통일

## 왜 이렇게 구현했는가

- 지금 프로젝트 병목은 "수집 기준 부재"였기 때문이다.
- 따라서 분석 단계보다 먼저 수집 단계의 공식 입력 구조를 정하는 것이 우선이었다.
- 이번 구현은 이후 전처리/적재/EDA 단계가 신뢰할 수 있는 raw 입력을 만드는 데 초점을 맞췄다.

## 이후 단계에서의 사용 방식

이 구현이 의미하는 파이프라인은 아래와 같다.

1. core 피처 수집
2. 전처리
3. 적재
4. 기준선 모델 평가
5. extended 피처 추가 여부 판단

즉 "처음부터 모든 피처를 다 공식 모델에 넣는다"가 아니라,
**핵심 피처로 기준선을 만든 뒤 확장 피처를 비교 실험으로 붙이는 구조**를 전제로 한다.
