# 반도체 주가 영향 요인 분석 데이터 수집

## 목표

삼성전자, SK하이닉스, NVIDIA 주가의 상승/하락에 영향을 주는 요인을 통계 및 머신러닝 기법으로 분석하기 위한 데이터를 수집한다.

## 수집 범위

기본 수집 기간은 `2011-01-01 ~ 현재`다. 최소 10년보다 길게 잡는 이유는 메모리 슈퍼사이클, 2018년 다운사이클, 코로나 공급망 충격, 2022년 금리 상승, 2023년 이후 AI/HBM 사이클을 함께 비교하기 위해서다.

## 실행 방법

```bash
pip install -r requirements.txt
python3 main/collect_data.py
```

## 데이터 저장 구조

```text
data/
  raw/
    fdr/
      prices/
      financials/
      semi_indices/
      global_indices/
      fx_rates/
      rates/
      futures/
      fred/
    external/
      sec/
      public_pages/
  processed/
    daily_panel.csv
    daily_returns.csv
    monthly_panel.csv
  metadata/
    source_catalog.csv
    symbol_map.csv
    data_dictionary.csv
    collection_summary.csv
    failed_collections.csv
```

## 수집 우선순위

- 1차: FinanceDataReader로 주가, 국내 재무제표, 반도체 ETF/지수, 글로벌 지수, 환율, 금리, 원자재 선물, FRED 데이터를 수집한다.
- 2차: 외부 공개 소스로 NVIDIA SEC 재무 데이터, SIA/WSTS, SEMI, DRAMeXchange/TrendForce, 한국 반도체 수출, TSMC 월간 매출 등 반도체 업황 데이터를 시도한다.
- 3차: 이벤트 데이터 후보로 실적 발표일, 컨퍼런스콜, 수출 규제, CHIPS Act, GPU 출시, HBM 공급 계약, 주요 고객사 CapEx 가이던스 소스를 시도하고 접근 가능 여부를 기록한다.

## 산출물
