# 반도체 핵심주 분석 진행 계획

## 목표

- 전체 분석 대상 중 삼성전자(`005930`)를 첫 번째 타겟으로 삼아 파이프라인을 먼저 완성한다.
- 삼성전자 기준으로 데이터 수집, 전처리, 모델링, SHAP 분석까지 한 번 끝까지 실행한 뒤 SK하이닉스와 NVDA로 확장한다.
- 현재는 삼성전자와 SK하이닉스를 같은 `macro_alpha` feature set으로 비교하고, 날짜 구간별 지표 영향력 조회까지 진행한다.

## 분석 기준

- 1차 타겟 종목: 삼성전자
- 2차 확장 종목: SK하이닉스
- 남은 확장 종목: NVDA
- 티커: `005930`, `000660`, `NVDA`
- 기간: `2018-01-01 ~ 2024-12-31`
- 1차 목표: 거시경제 지표와 수급 지표가 5거래일 후 주가 방향성에 미치는 영향 확인
- 최종 산출물: 모델 성능 결과, SHAP 중요도 그래프, 날짜별 SHAP 결과, 구간별 지표 영향력 CSV

## 먼저 진행할 순서

### 1. 환경 설정 확인

- [x] `.env` 파일 생성 및 DB 접속 정보 입력
- [x] `DART_API_KEY` 입력 여부 확인
- [x] PostgreSQL 실행 확인
- [x] `semiconductor` DB 생성
- [x] `python src/db.py` 실행으로 테이블 생성 확인

### 2. 삼성전자 데이터 수집부터 진행

- [x] 삼성전자 주가 데이터 수집: `005930`
- [x] 거시 지표 수집: NASDAQ, SOX, VIX, 미국 기준금리, USD/KRW, KOSPI, KOSPI200
- [x] 국내 수급 데이터 수집: 외국인 순매수, 기관 순매수
- [x] 수집 데이터의 날짜 범위가 `2018-01-01 ~ 2024-12-31`인지 확인
- [x] DB 저장 시 중복 저장이 발생하지 않는지 확인
- [x] 수급 데이터는 `KRX_ID`, `KRX_PW` 설정 후 수집 완료

### 3. 전처리

- [x] 삼성전자는 국내 종목이므로 미국 거시 지표를 `shift(1)` 적용
- [x] 모든 모델 입력 피처에 look-ahead bias가 없는지 확인
- [x] `nasdaq`, `vix`, `us_rate`, `usd_krw`, `sox`의 lag feature 생성
- [x] 결측치 처리 방식 결정 및 적용
- [x] 모델 학습용 타겟 변수 생성: 5거래일 후 상승 여부
- [x] 거시 지표 변화율 feature 생성
- [x] 외국인/기관 수급 rolling feature 생성
- [x] 거래대금 대비 외국인/기관 수급 강도 feature 생성

### 4. 1차 모델링

- [x] 삼성전자 데이터만 사용해 Train/Validation/Test를 시간 순서로 분할
- [x] Baseline feature 모델 학습
- [x] Proposed feature 모델 학습
- [x] Validation 기준 F1, Recall, AUC 확인
- [x] Validation AUC 기준 XGBoost 하이퍼파라미터 튜닝
- [x] 기술적 지표 추가 후 Baseline/Proposed/Proposed+Technical 비교
- [x] Validation 기준 classification threshold 튜닝
- [x] 지표 영향력 분석용으로 5거래일 후 상승 타겟 모델 재학습
- [x] 가격 레벨 피처를 제외한 `macro_alpha` feature set 추가
- [x] 삼성전자와 SK하이닉스를 `macro_alpha` 기준으로 재테스트
- [x] `outputs/results/model_comparison.csv` 저장

### 5. SHAP 분석

- [x] 삼성전자 테스트셋 기준 글로벌 SHAP bar plot 생성
- [x] 글로벌 SHAP beeswarm plot 생성
- [x] 2024년 AI 랠리 날짜 로컬 SHAP 분석
- [x] `us_rate`, `vix` 방향성 sanity check 수행
- [x] SHAP 결과를 `shap_results` 테이블에 저장
- [x] 5거래일 후 상승 타겟 기준 SHAP 결과 재생성
- [x] `shap_results`에 `feature_set` 컬럼 추가
- [x] 삼성전자 `macro_alpha` SHAP 분석 완료
- [x] SK하이닉스 `macro_alpha` SHAP 분석 완료
- [x] `proposed`와 `macro_alpha` SHAP 결과를 함께 저장할 수 있게 분리

### 6. 날짜 구간별 지표 영향력 조회

- [x] `src/interpret_period.py` 추가
- [x] 사용자가 지정한 날짜 구간의 평균 절대 SHAP 기준 상위 지표 조회
- [x] 구간별 `mean_abs_shap`, `mean_shap`, `dominant_direction`, `positive_shap_ratio` 계산
- [x] 삼성전자 2024년 구간 예시 CSV 저장
- [x] SK하이닉스 2024년 구간 예시 CSV 저장

## SK하이닉스 확장 상태

- [x] SK하이닉스 주가 데이터 수집: `000660`
- [x] SK하이닉스 외국인/기관 수급 데이터 수집
- [x] SK하이닉스 전처리 CSV 생성: `data/processed/sk_hynix_preprocessed.csv`
- [x] SK하이닉스 XGBoost/Logistic Regression 모델링
- [x] SK하이닉스 `macro_alpha` 모델링
- [x] SK하이닉스 `macro_alpha` SHAP 분석
- [x] 삼성전자와 같은 기준으로 구간별 지표 영향력 조회 가능

## 후순위 작업

- [x] KRX 계정 설정 후 국내 종목 외국인/기관 수급 데이터 수집
- [ ] OpenDartReader 기반 삼성전자 분기 실적 수집 구현
- [ ] `revenue`, `operating_profit`, `eps`를 일별 forward-fill로 확장
- [x] 삼성전자 파이프라인 안정화 후 SK하이닉스 적용
- [ ] 국내 종목 2개 완료 후 NVDA용 별도 동기화 방식 적용
- [ ] NVDA 주가 및 미국 알파 피처 수집 방식 확정
- [ ] NVDA 전처리, 모델링, SHAP 분석 확장

## 오늘의 완료 기준

- [x] DB 테이블 초기화 완료
- [x] 삼성전자 주가, 거시 지표, 수급 데이터 수집 방향 확정
- [x] 삼성전자 우선 파이프라인의 실행 순서가 문서화됨
- [x] 삼성전자와 SK하이닉스 `macro_alpha` 비교 기준 확정
- [x] 원하는 날짜 구간별 지표 영향력 조회 기능 추가

## 현재 다음 작업

- [ ] NVDA 데이터 수집 구조 설계
- [ ] NVDA용 알파 피처 후보 선정: Put/Call Ratio, Short Interest 등
- [ ] 국내 종목과 미국 종목의 시계열 동기화 방식 차이 정리
- [ ] 세 종목 공통 비교표 작성
