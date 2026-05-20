# Extended 피처 후보 검토

## 문서 목적

이 문서는 제안서 기준 피처 분류와 `data_collection` 브랜치의 수집 후보를 종합해,
향후 `extended`로 수집하거나 모델 실험에 사용할 후보를 정리한다.

핵심은 "수집할 수 있는가"와 "모델에 바로 넣어도 되는가"를 분리하는 것이다.
수집 단계에서는 후보를 넓게 확보할 수 있지만, baseline 모델에는 core 피처만 사용한다.

---

## 1. 최종 의사결정 요약

- Baseline 성공 조건은 `core` 피처만으로 판단한다.
- `extended` 피처는 수집되어 있어도 baseline 성공/실패 판정에 사용하지 않는다.
- 1차 모델은 `core only`로 만들고, `extended`는 이후 ablation 실험에서만 추가한다.
- 월간·분기·공시 기반 지표는 발표일/as-of 기준 정렬이 구현되기 전까지 모델 투입 금지다.
- `archive/reference`는 현재 모델 입력에서 제외하고, 구조화 시계열·라이선스·과거 재현성이 모두 확보될 때만 재검토한다.

즉, extended 수집 실패는 core baseline 실험을 막지 않는다.
Extended 피처의 모델 투입 승격 기준은 이후 모델링/평가 단계에서 별도 문서로 정의한다.

---

## 2. 피처 분류와 실행 등급

### 2.1. 피처 분류

피처 분류는 "이 데이터가 프로젝트에서 어떤 상태인가"를 나타낸다.
실행 등급은 다음 절에서 "지금 무엇을 할 것인가"를 나타낸다.

| 구분 | 의미 | 대응 실행 등급 | baseline 성공 조건 포함 | 1차 모델 투입 |
|---|---|---|---:|---:|
| `core` | 공식 baseline에 필요한 핵심 피처 | 별도 core 수집/검증 | 포함 | 포함 |
| `extended-collect` | 나중에 쓸 수 있으므로 우선 수집 대상으로 관리하는 확장 피처 | `collect-now`, `collect-later` | 제외 | 제외 |
| `extended-model-candidate` | 수집된 extended 중 core-only baseline 이후 모델 실험 대상으로 승격된 피처 | `model-ablation-only`, `model-after-lag-rule`, `target-specific-only` | 제외 | 검증 후 가능 |
| `archive/reference` | 참고용으로 보관하지만 현재 모델 피처가 아닌 자료 | `archive-only` | 제외 | 제외 |

즉 `extended-collect`와 `extended-model-candidate`는 같은 축의 병렬 등급이 아니라 순차 관계다.
먼저 수집 가능한 후보를 `extended-collect`로 확보한다.
그중 look-ahead 검증, 중복성 검토, target별 적합성 확인을 통과한 피처만
모델링 단계에서 `extended-model-candidate`로 승격한다.
`exclude`는 피처 분류가 아니라 이번 범위에서 수집 또는 모델 투입을 하지 않겠다는 실행 판단이다.

### 2.2. 실행 등급

| 등급 | 의미 |
|---|---|
| `collect-now` | 지금 수집한다. 단 baseline 성공 조건과 1차 모델에서는 제외한다. |
| `collect-later` | 수집 후순위다. 소스, 라이선스, 재현성, 구현 비용을 더 확인한다. |
| `archive-only` | 참고용으로만 보관하고 현재 모델 입력에서 제외한다. |
| `model-ablation-only` | baseline 제외, core-only 이후 ablation 실험에서만 투입한다. |
| `model-after-lag-rule` | 발표일/as-of 규칙 구현 후에만 모델 후보가 된다. |
| `target-specific-only` | 특정 종목군에만 사용한다. |
| `exclude` | 현재 범위에서는 제외한다. |

---

## 3. 실행 판단 요약

| 피처 | 수집 실행 | 모델 실행 | 비고 |
|---|---|---|---|
| `DGS10`, `DGS2`, `T10Y2Y` | `collect-now` | `model-ablation-only` | 금리 계열 동시 기본 투입 금지 |
| `DTWEXBGS` | `collect-now` | `model-ablation-only` | Broad Dollar Index 우선, `^NYICDX`는 대체 후보 |
| `S&P500` | `collect-now` | `model-ablation-only` | NASDAQ 대비 추가 설명력 검증 |
| `SMH`, `SOXX` | `collect-now` | `model-ablation-only` | SOX 우선, ETF는 대체/괴리 실험용 |
| `IPG3344S`, `PCU334413334413P`, `A34SNO` | `collect-now` | `model-after-lag-rule` | 월간 FRED 업황 지표, 발표일/as-of 처리 필요 |
| NVIDIA SEC Company Facts | `collect-now` | `model-after-lag-rule`, `target-specific-only` | NVDA 전용, filing date 기준 |
| 국내 기업 분기 실적 | `collect-now` | `model-after-lag-rule`, `target-specific-only` | 삼성전자·SK하이닉스 전용, DART 공시일 기준 권장 |
| 국내 외국인/기관 순매수 | `collect-now` | `model-after-lag-rule`, `target-specific-only` | 국내 종목 전용, 예측 타깃에 맞춰 lag 처리 |
| 한국 반도체 수출액/증감률 | `collect-later` | `model-after-lag-rule`, `target-specific-only` | 한국 종목 전용, 자동 수집 경로와 발표일 규칙 확정 필요 |
| `KOSDAQ` | `collect-later` | `model-ablation-only` | 국내 성장주 심리 후순위 실험 후보 |
| 구리 가격 | `collect-later` | `model-ablation-only` | 경기/AI 인프라 프록시이나 직접성 약함 |
| DRAM/NAND/HBM 가격 | `archive-only` | `exclude` | 라이선스·구조화 시계열·과거 재현성 확보 전 모델 제외 |
| SEMI equipment billings | `archive-only` | `exclude` | 구조화 시계열 확보 전 모델 제외 |
| SIA/WSTS/TrendForce HTML | `archive-only` | `exclude` | HTML archive는 모델 피처가 아님 |
| TSMC monthly revenue page | `archive-only` | `exclude` | `data_collection`에서 403 실패, 대체 소스 필요 |
| Dow Jones, Russell 2000, Nikkei225 | `exclude` | `exclude` | 현재 core/extended 후보 대비 직접성 약함 |
| WTI, Brent, 천연가스, 금, 은 | `exclude` | `exclude` | 원자재 프록시로는 가능하나 현재 범위에서는 후순위 |
| CNY/KRW | `exclude` | `exclude` | 수집 row가 1개 수준이라 피처로 부적합 |
| 뉴스 감성 | `collect-later` | `exclude` | 별도 정제·라벨링·시점 정렬 설계 전까지 모델 제외 |

---

## 4. 수집 판단 기준

| 판단 | 기준 |
|---|---|
| `collect-now` | API/공개 소스가 있고, 과거 시계열 재현 가능하며, 라이선스 리스크가 낮다. |
| `collect-later` | 경제적 논리는 있으나 수집 안정성, 소스 접근성, 과거 재현성, 구현 비용 중 하나가 불확실하다. |
| `archive-only` | 구조화 수치로 자동화하기 어렵거나 라이선스·재현성 문제가 커서 현재 모델 입력으로 쓸 수 없다. |
| `exclude` | 현재 제안서 목표와 직접 연결성이 약하거나 core/extended 후보 대비 추가 설명력이 낮다. |

---

## 5. Look-Ahead 방지 규칙

월간·분기·공시 기반 피처는 다음 메타데이터를 가져야 한다.

- `period_start`
- `period_end`
- `release_date` 또는 `filing_date`
- `effective_from`: 해당 값이 모델 feature로 사용 가능한 첫 거래일
- `source_snapshot_date`: 데이터 수집 또는 스냅샷 기준일

모델 feature join은 `period_end`가 아니라
`effective_from <= prediction_date` 기준으로 수행한다.

발표일 정보가 불확실한 월간 지표는 최소 1개월 lag,
분기 지표는 최소 1분기 lag를 적용한다.
실제 발표일 캘린더가 확보되면 발표일 다음 거래일부터 사용 가능하도록
`effective_from`을 계산한다.

국내 외국인·기관 순매수도 당일 종가 방향 예측에 당일 값을 쓰면 look-ahead가 될 수 있다.
예측 대상이 다음 거래일이면 `t`일 수급으로 `t+1` 방향을 예측하는 식으로
타깃 시점을 명확히 해야 한다.

---

## 6. 중복 피처군 관리 원칙

아래 피처군은 경제적으로 의미가 있지만 동시에 넣으면 중복 정보가 커질 수 있다.

| 피처군 | 중복 위험 | 관리 방식 |
|---|---|---|
| `DGS10`, `DGS2`, `T10Y2Y` | 금리 레벨과 수익률곡선 정보가 겹침 | 모두 수집하되 모델에는 조합별 ablation |
| `SOX`, `SMH`, `SOXX` | 반도체 업종 베타가 겹침 | SOX 우선, ETF는 대체 소스 또는 별도 실험 |
| `NASDAQ`, `S&P500` | 미국 주식시장 베타가 겹침 | S&P500은 전체 위험자산 베타 통제 후보로만 사용 |
| `USD/KRW`, 달러 인덱스 | 달러 강세와 원화 약세 정보가 일부 겹침 | 한국 환율 효과와 글로벌 달러 효과를 분리해 해석 |

기본 모델 또는 단일 ablation 실험에서 아래 조합은 동시에 투입하지 않는다.

- `DGS10 + DGS2 + T10Y2Y`
- `SOX + SMH + SOXX`
- `NASDAQ + S&P500`: 추가 설명력 검증 없는 기본 동시 투입 금지
- `USD/KRW + Dollar Index`: 한국 환율 효과와 글로벌 달러 효과를 분리할 목적이 있을 때만 동시 투입

---

## 7. 상세 피처 검토

이 표는 3장의 실행 판단 요약에 나온 항목을 같은 기준으로 다시 풀어 쓴다.

| 피처 | 후보 소스/심볼 | 수집 실행 | 모델 실행 | 논리 타당성 | 실행 가능성 | core 중복 위험 | look-ahead 위험 | 논거 |
|---|---|---|---|---|---|---|---|---|
| 미국 10년물 금리 | `FRED:DGS10`, `US10YT` | `collect-now` | `model-ablation-only` | 강함 | 높음 | 중간 | 중간 | 장기 할인율 프록시로 성장주·반도체주 밸류에이션 부담을 설명한다. |
| 미국 2년물 금리 | `FRED:DGS2` | `collect-now` | `model-ablation-only` | 강함 | 높음 | 높음 | 중간 | 통화정책 기대 프록시다. `DGS10`과 함께 쓰면 금리 레벨 요인이 중복된다. |
| 장단기 금리차 | `FRED:T10Y2Y` | `collect-now` | `model-ablation-only` | 중간~강함 | 높음 | 높음 | 중간 | 경기 국면과 수익률곡선 프록시다. `DGS10`, `DGS2`에서 파생된 성격이 있다. |
| Broad Dollar Index | `FRED:DTWEXBGS` | `collect-now` | `model-ablation-only` | 강함 | 높음 | 중간 | 중간 | 글로벌 달러 유동성·위험회피 프록시다. `^NYICDX`보다 FRED 기반 broad index를 우선한다. |
| S&P500 | `S&P500` | `collect-now` | `model-ablation-only` | 중간 | 높음 | 높음 | 낮음~중간 | 미국 전체 위험자산 베타 통제 후보다. NASDAQ 대비 추가 설명력을 검증한다. |
| SMH / SOXX ETF | `SMH`, `SOXX` | `collect-now` | `model-ablation-only` | 중간~강함 | 높음 | 높음 | 낮음~중간 | SOX 대체 소스, 거래 가능한 ETF 수급 프록시, SOX 대비 ETF 괴리 실험에만 사용한다. |
| 반도체·전자부품 산업생산 | `FRED:IPG3344S` | `collect-now` | `model-after-lag-rule` | 중간~강함 | 높음 | 낮음 | 높음 | 실제 산업 활동에 가까운 업황 프록시다. 발표일 이후부터만 사용할 수 있다. |
| 반도체 관련 PPI | `FRED:PCU334413334413P` | `collect-now` | `model-after-lag-rule` | 중간 | 높음 | 낮음 | 높음 | 가격·마진 환경 프록시다. DRAM/NAND 가격이나 메모리 ASP와 동일하게 해석하지 않는다. |
| 컴퓨터·전자제품 신규주문 | `FRED:A34SNO` | `collect-now` | `model-after-lag-rule` | 중간 | 높음 | 낮음 | 높음 | 반도체 신규주문이 아니라 전방 IT 수요 프록시다. |
| NVIDIA SEC Company Facts | SEC Company Facts | `collect-now` | `model-after-lag-rule`, `target-specific-only` | 강함 | 중간~높음 | 낮음 | 높음 | NVDA 전용 미시 피처다. period end date가 아니라 filing date 기준으로 붙인다. |
| 국내 기업 분기 실적 | FDR/NAVER, DART | `collect-now` | `model-after-lag-rule`, `target-specific-only` | 강함 | 중간 | 낮음 | 높음 | 삼성전자·SK하이닉스 전용 펀더멘털 피처다. DART 공시일 기준 as-of 처리가 필요하다. |
| 국내 외국인/기관 순매수 | PyKRX/KRX | `collect-now` | `model-after-lag-rule`, `target-specific-only` | 강함 | 중간 | 낮음~중간 | 중간 | 국내 반도체주 단기 수급 설명에 유용하다. 타깃 시점에 맞춰 lag 처리한다. |
| 한국 반도체 수출액/증감률 | 관세청, 무역협회, 수동 CSV | `collect-later` | `model-after-lag-rule`, `target-specific-only` | 강함 | 중간 | 낮음 | 높음 | 삼성전자·SK하이닉스에는 타당하지만 자동 수집 경로와 발표일 규칙 확정이 필요하다. |
| KOSDAQ | `KQ11` | `collect-later` | `model-ablation-only` | 약함~중간 | 높음 | 중간 | 낮음 | 국내 성장주 심리 프록시다. 삼성전자·SK하이닉스 직접 지수는 아니므로 후순위다. |
| 구리 가격 | `HG=F` | `collect-later` | `model-ablation-only` | 약함~중간 | 높음 | 중간 | 낮음~중간 | 제조업·AI 인프라·경기 프록시이나 반도체 직접 원가로 주장하면 과장이다. |
| DRAM/NAND/HBM 가격 | TrendForce, DRAMeXchange, 수동 CSV | `archive-only` | `exclude` | 강함 | 낮음 | 낮음 | 중간~높음 | 논리는 강하지만 구조화 데이터, 라이선스, 과거 시계열 재현성이 확보되기 전까지 모델 제외다. |
| SEMI equipment billings | SEMI Billings | `archive-only` | `exclude` | 중간 | 낮음~중간 | 낮음 | 높음 | 설비투자 사이클 자료지만 구조화 시계열 확보 전까지 모델 제외다. |
| SIA/WSTS/TrendForce HTML | 외부 공개 페이지 | `archive-only` | `exclude` | 중간~강함 | 낮음 | 낮음 | 높음 | HTML 원문은 참고 자료일 뿐 구조화 수치 피처가 아니다. 수치 시계열이 확보되기 전까지 모델 제외다. |
| TSMC monthly revenue page | TSMC IR page | `archive-only` | `exclude` | 중간 | 낮음 | 낮음 | 높음 | 파운드리·AI 수요 프록시 가능성은 있으나 `data_collection`에서 403으로 실패했다. 대체 소스 확보 전까지 모델 제외다. |
| Dow Jones / Russell 2000 / Nikkei225 | `DJI`, `RUT`, `N225` | `exclude` | `exclude` | 약함~중간 | 높음 | 중간~높음 | 낮음~중간 | 시장·위험선호 프록시로는 가능하지만 현재 core/extended 후보 대비 반도체 직접성이 약하다. |
| WTI / Brent / 천연가스 / 금 / 은 | `CL=F`, `BZ=F`, `NG=F`, `GC=F`, `SI=F` | `exclude` | `exclude` | 약함~중간 | 높음 | 중간 | 낮음~중간 | 원자재는 반도체 직접 원가보다 경기, 인플레이션, 위험회피 프록시에 가깝다. 현재 범위에서는 제외한다. |
| CNY/KRW | `CNY/KRW` | `exclude` | `exclude` | 약함 | 낮음 | 중간 | 중간 | `data_collection` 결과에서 row가 1개 수준이라 재현 가능한 피처로 보기 어렵다. |
| 뉴스 감성 | 뉴스 원문, 감성 점수 | `collect-later` | `exclude` | 중간 | 낮음 | 낮음 | 높음 | 제안서 취지에는 맞지만 수집, 정제, 라벨링, 시점 정렬 난도가 높다. 별도 설계 전까지 모델 제외다. |

---

## 8. 대안 세트별 모델 실험 설계

### 8.1. 금리 계열

`DGS10`, `DGS2`, `T10Y2Y`는 모두 금리 계열이며 서로 높은 상관 또는 파생 관계를 가진다.
따라서 세 변수를 동시에 기본 모델에 투입하지 않고 다음 조합을 비교한다.

1. `DGS10` 단독: 장기 할인율 효과
2. `DGS2` 단독: 통화정책 기대 효과
3. `T10Y2Y` 단독: 경기 국면/수익률곡선 효과
4. `DGS10 + T10Y2Y`: 금리 레벨과 경기 스프레드 분리

최종 모델 투입 여부는 validation 성능과 SHAP 안정성을 기준으로 결정한다.

### 8.2. 시장 베타 계열

`S&P500`은 NASDAQ과 중복될 수 있으므로 단순 시장지수 추가가 아니라
미국 전체 위험자산 베타를 통제하기 위한 후보로 둔다.

NASDAQ이 기술주·성장주 베타를 반영한다면,
S&P500은 보다 넓은 시장 위험선호를 나타낸다.
두 지수를 동시에 사용할 경우 추가 설명력이 있는지 반드시 비교한다.

### 8.3. 반도체 ETF 계열

`SMH`, `SOXX`는 SOX와 경제적 의미가 크게 겹친다.
따라서 SOX 수집이 안정적이면 기본적으로 SOX를 우선 사용하고,
SMH/SOXX는 다음 목적일 때만 사용한다.

1. SOX 데이터의 대체 소스
2. 거래 가능한 반도체 ETF의 수급 또는 섹터 자금 흐름 프록시
3. SOX 대비 ETF 성과 괴리를 활용한 별도 실험

SOX, SMH, SOXX를 모두 동시에 투입하는 것은 기본 설정에서 제외한다.

---

## 9. 최종 정리

Extended 피처는 수집 가능하면 함께 수집할 수 있다.
다만 수집 여부와 모델 투입 여부는 분리한다.

- 수집 단계: core와 `collect-now` extended를 함께 수집하되, metadata에서 priority를 분리한다.
- baseline 판정: core 피처 기준으로 성공 여부를 판단한다.
- 모델링 단계: core only 모델을 먼저 만들고, extended는 ablation 실험으로 추가한다.
- 해석 단계: 중복 피처군은 SHAP 중요도 분산 여부를 확인한 뒤 최종 사용 여부를 정한다.

이 구조가 가장 안전한 이유는,
데이터 확보를 미루지 않으면서도 "많이 모았으니 다 넣는다"는 위험을 피할 수 있기 때문이다.

---

## 10. 수집 대상 최종 목록

수집 단계에서의 결론은 단순하다.
`exclude`를 제외하고, 자동 수집 가능성과 구현 비용에 따라 아래 순서로 진행한다.

### 10.1. 지금 수집할 피처

아래 피처는 `collect-now`다.
core 수집과 함께 진행하되, metadata에서는 `extended`로 분리하고
baseline 성공 조건에는 포함하지 않는다.

- `DGS10`
- `DGS2`
- `T10Y2Y`
- `DTWEXBGS`
- `S&P500`
- `SMH`
- `SOXX`
- `IPG3344S`
- `PCU334413334413P`
- `A34SNO`
- NVIDIA SEC Company Facts
- 국내 기업 분기 실적
- 국내 외국인/기관 순매수

### 10.2. 여유가 있으면 수집할 피처

아래 피처는 `collect-later`다.
논리는 있지만 수집 경로, 발표일 정렬, 구현 비용을 더 확인해야 하므로
지금 단계의 필수 수집 대상은 아니다.

- 한국 반도체 수출액/증감률
- `KOSDAQ`
- 구리 가격
- 뉴스 감성 원천 자료

### 10.3. 지금은 수집보다 보관 또는 제외할 자료

아래 항목은 현재 모델 피처로 쓰지 않는다.
구조화 시계열, 라이선스, 과거 재현성이 확보되기 전까지는
archive/reference 또는 제외 대상으로 둔다.

- DRAM/NAND/HBM 가격
- SEMI equipment billings
- SIA/WSTS/TrendForce HTML
- TSMC monthly revenue page
- Dow Jones, Russell 2000, Nikkei225
- WTI, Brent, 천연가스, 금, 은
- CNY/KRW

따라서 "일단 다 수집"의 실제 의미는
`core + collect-now`를 우선 수집하고,
시간이 남으면 `collect-later`를 추가 수집한다는 뜻이다.
`archive-only`와 `exclude`는 지금 모델 피처 수집 대상으로 보지 않는다.
