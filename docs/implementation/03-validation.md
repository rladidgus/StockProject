# 검증 기록

## 검증 목적

새 수집 파이프라인이 최소한 아래 조건을 만족하는지 확인한다.

- Python 문법상 문제 없음
- 실행 엔트리포인트가 명확함
- 메타데이터 파일 생성 경로가 정리돼 있음
- 수집 실패/empty 상태가 메타데이터에 남을 수 있음

## 예정 검증 항목

1. `python3 -m compileall src`
2. `python3 src/collect.py --help`

## 주의

실제 데이터 수집 실행은 외부 네트워크와 API 환경에 영향을 받는다.
따라서 이번 검증은 우선 **구조와 실행 가능성** 위주로 본다.

## 실제 검증 결과

### 1. `python3 -m compileall src`

- 결과: 성공
- 의미: 현재 `src/` 수집 코드가 문법 수준에서는 컴파일 가능함

### 2. `python3 src/collect.py --help`

- 결과: 성공
- 현재 상태: argparse 도움말 출력 가능
- 비고: 현재 환경에서는 PyKRX/KRX 로그인 관련 경고가 도움말 전에 출력될 수 있음

## 현재 검증 판단

- 코드 문법은 통과했다.
- 실행 엔트리포인트는 존재하지만, 실제 실행 검증을 위해서는 `requirements.txt` 의존성 설치가 먼저 필요하다.
- 따라서 다음 검증 단계는:
  1. 의존성 설치
  2. `src/collect.py --help` 재실행
  3. 실제 수집 dry-run 또는 부분 실행

## 잔여 리스크

- 외부 데이터 소스 응답 변화
- PyKRX / yfinance / FDR 버전 차이
- 빈 결과가 실제 실패인지 휴장/범위 문제인지 추가 판별 규칙 필요

## 리뷰 반영 사항

1. 실패한 수집 단계도 `collection_summary.csv`에 남도록 `finally` 경로 보장
2. NVDA 종료일 배타 문제 보정
3. 수급 컬럼명을 거래대금 단위가 드러나는 `*_net_buy_value`로 수정
4. `collect-only` 리팩터링 잔재인 `skip_db` 인자 불일치 제거
5. core macro 필수 컬럼 누락 시 success 금지
6. 실행 파라미터(`start/end/skip flags`) 메타데이터 기록
7. 불필요한 `dotenv` 의존성 제거
8. NVDA `auto_adjust=False` 고정
9. `KOSPI`, `KOSPI200`를 extended macro로 분리
10. `--skip-alpha` 시 종목별 skip 메타데이터 기록
11. `collection_summary.csv` append로 실행 이력 보존
12. raw CSV를 `run_id`별 경로에 보존
13. core macro 실패 시에도 개별 성공 macro raw는 남김
14. `core_macro_features.csv`를 core 컬럼만 포함하도록 고정
15. core macro 필수 컬럼은 값 존재까지 검사
16. `run_id`를 마이크로초+랜덤 suffix로 생성
17. macro 메타데이터를 개별 series 단위까지 기록
18. `source_catalog.csv`를 run별 메타데이터 경로에도 보존
19. `return_pct`를 raw 수집 단계에서 제거
20. macro 전체 empty여도 개별 `macro_series` 상태를 남김
21. `--skip-raw` 실행은 공식 raw 기준 run 실패로 처리
22. macro 개별 파일 저장 실패도 메타데이터에 남기도록 보강
23. 국내 수급(alpha)은 KRX credential 의존성 때문에 official baseline 필수 조건에서 제외
24. canonical `source_catalog.csv`에 aggregate artifact `core_macro_features`를 명시
25. core macro bundle은 교집합으로 축소하지 않고 daily raw grid를 그대로 보존
26. `pykrx` import를 alpha 수집 시점으로 늦춰 optional dependency로 분리
27. `pykrx`가 아예 설치되지 않은 환경에서도 baseline run 자체는 유지하고, alpha는 failed metadata로 흡수
28. core macro 누락 컬럼 검사는 dataframe slicing 전에 수행해 graceful failure metadata를 보장
29. canonical `collection_summary.csv`는 status row가 하나도 없는 early-failure run에서 빈 파일로 오염되지 않도록 보호
30. core macro bundle의 date grid는 extended macro availability에 영향을 받지 않도록 core series만으로 구성
31. core macro bundle 재구성 시 각 core series는 `dropna()` 후 concat해 extended가 만든 all-NaN 날짜를 제거
32. canonical `source_catalog.csv`는 run 성공 여부와 분리된 정적 계약으로 항상 갱신
33. `core_macro_features` provenance symbol은 `IXIC,^SOX,VIX,FRED:FEDFUNDS,USD/KRW`로 통일
34. canonical `source_catalog.csv`는 정적 계약 파일로 유지하기 위해 `run_id`를 포함하지 않음
35. `--skip-raw` 실행에서는 artifact 상태도 `skipped`로 기록해 raw 파일 미생성을 artifact 수준에서 명시
36. `core_macro_features` provenance source는 metadata 전반에서 `internal_bundle`로 통일
37. core macro bundle의 explicit daily grid는 `nasdaq`, `sox`, `vix`, `usd_krw` 날짜 집합으로 제한
38. `--skip-raw` debug run에서는 `skipped` artifact를 core validation 누락으로 취급하지 않고, 별도 `raw_output_contract` 실패로 baseline 부적합성을 표현
39. `raw_output_contract` row를 core validation 전에 기록해 debug run과 source outage를 구분
40. 채택 가능한 성공 run만 `data/metadata/current_raw_manifest.csv`를 갱신해 전처리 브랜치의 stable raw selector를 제공
41. `us_rate` 원자료 관측일은 daily grid 밖이어도 core macro bundle에 보존
42. adopted manifest 갱신 전 core stock과 core macro series의 수집 기간 커버리지를 검증
43. `us_rate`는 FRED CSV 직접 조회 경로로 수집해 FDR/urllib SSL 실패를 회피
44. `requirements.txt`를 collect-only 실행 의존성으로 축소

## 수정 후 추가 확인

- `python3 -m compileall src` 재실행 결과 통과
- 수급 컬럼명이 거래대금 단위를 드러내도록 반영됨
- core macro는 누락 컬럼이 있으면 실패 상태로 기록되도록 보강
- `python3 src/collect.py --help`가 import 단계 실패 없이 실행되는 것 확인

## 후속 주의사항

- 이후 전처리 단계에서 기존 `foreign_net_buy`, `institution_net_buy` 참조가 있다면
  새 raw 컬럼명(`foreign_net_buy_value`, `institution_net_buy_value`)에 맞춰 정렬이 필요하다.
- `return_pct`는 이제 raw 입력에 없으므로 전처리 단계에서 직접 계산해야 한다.
