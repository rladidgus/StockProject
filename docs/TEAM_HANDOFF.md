# 팀 공유용 읽는 순서

## 결론

이 브랜치는 `data_collection`을 통째로 병합한 결과물이 아니라,
기존 브랜치들을 참고해서 새로 정리한 **collect-only raw baseline**이다.

현재 단계의 목표는 전처리, 적재, 모델링이 아니라
후속 단계가 공통으로 사용할 수 있는 raw CSV와 metadata 계약을 확정하는 것이다.

## 먼저 읽을 문서

1. `docs/planning/00-read-this-first.md`
   - 프로젝트를 core / extended / archive 피처로 나눠 이해하는 기준 문서다.

2. `docs/planning/01-data-collection-feature-criteria.md`
   - 어떤 피처를 왜 수집하는지 설명하는 문서다.

3. `docs/implementation/01-scope.md`
   - 이번 브랜치가 어디까지 책임지는지 정리한 문서다.

4. `docs/implementation/02-implementation.md`
   - 실제 구현 규칙과 metadata 계약을 설명한다.

5. `docs/implementation/03-validation.md`
   - 리뷰에서 지적된 문제와 수정 근거를 기록한 문서다.

## 실제 raw 데이터는 어디에 있는가

전처리 단계는 직접 run 폴더명을 고르지 않는다.
아래 manifest 파일을 먼저 읽는다.

```text
data/metadata/current_raw_manifest.csv
```

현재 adopted raw run은 아래다.

```text
20260519T142302583283Z-de7987ab
```

실제 raw CSV 위치는 아래다.

```text
data/raw/runs/20260519T142302583283Z-de7987ab/
```

핵심 파일은 다음과 같다.

```text
prices/samsung_electronics.csv
prices/sk_hynix.csv
prices/nvidia.csv
macro/nasdaq.csv
macro/sox.csv
macro/vix.csv
macro/us_rate.csv
macro/usd_krw.csv
macro/core_macro_features.csv
```

## `data_collection`

`data_collection`은 더 많은 자료와 수집 시도를 담고 있어서 참고본으로 가치가 있다.
하지만 이번 단계 목표를 **공식 raw 수집 기준선 확정**으로 보면,
현재 브랜치가 더 적합하다.

이유는 다음과 같다.

- 현재 브랜치는 수집 단계만 담당한다.
- 전처리, 적재, 모델링, SHAP 코드를 섞지 않는다.
- core 피처와 extended 피처를 구분한다.
- raw CSV를 run 단위로 보존한다.
- `current_raw_manifest.csv`로 공식 adopted run을 명시한다.
- `source_catalog.csv`와 `collection_summary.csv`로 출처, 상태, 경로를 추적한다.

따라서 비교 결론은 아래처럼 표현하는 것이 가장 정확하다.

```text
data_collection은 넓은 수집 참고본으로 가치가 있고,
현재 브랜치는 후속 전처리 단계에 넘길 공식 raw baseline으로 더 적합하다.
```

## 현재 브랜치에서 의도적으로 제외한 것

- 전처리 로직
- DB 적재 로직
- 모델링
- SHAP 분석
- 수익률 파생 컬럼 생성
- 국내 수급 alpha의 official baseline 필수화

국내 수급 alpha는 중요할 수 있지만 KRX credential 의존성이 있으므로,
현재 official baseline에서는 extended로 둔다.

## 다음 단계

1. 현재 브랜치를 커밋한다.
2. 전처리 브랜치를 새로 만든다.
3. 전처리 코드는 `data/metadata/current_raw_manifest.csv`가 가리키는 raw run을 입력으로 삼는다.
4. 수익률, lag, 시차 정렬, forward-fill은 전처리 브랜치에서 처리한다.