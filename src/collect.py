from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import certifi
import FinanceDataReader as fdr
import pandas as pd
import requests
import yfinance as yf

START = "2011-01-01"
END = datetime.now().date().isoformat()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
METADATA_ROOT = DATA_ROOT / "metadata"


@dataclass(frozen=True)
class StockSpec:
    ticker: str
    name: str
    source: str
    provider_symbol: str


@dataclass(frozen=True)
class MacroSpec:
    feature_name: str
    source: str
    provider_symbol: str
    provider_column: str
    file_name: str
    rationale: str


@dataclass(frozen=True)
class FinancialSpec:
    feature_name: str
    source: str
    provider_symbol: str
    frequency: str
    file_name: str
    rationale: str


@dataclass(frozen=True)
class ExternalSourceSpec:
    name: str
    source: str
    url: str
    file_name: str
    priority: str
    rationale: str


KR_STOCKS = [
    StockSpec("005930", "samsung_electronics", "FDR", "005930"),
    StockSpec("000660", "sk_hynix", "FDR", "000660"),
]
US_STOCKS = [
    StockSpec("NVDA", "nvidia", "yfinance", "NVDA"),
]

CORE_MACRO_SPECS = [
    MacroSpec("nasdaq", "FDR", "IXIC", "Close", "nasdaq.csv", "미국 기술주 전반 위험선호"),
    MacroSpec("sox", "FDR", "^SOX", "Close", "sox.csv", "반도체 업종 전반 강도"),
    MacroSpec("vix", "FDR", "VIX", "Close", "vix.csv", "시장 변동성 및 위험회피 심리"),
    MacroSpec("us_rate", "FRED", "FRED:FEDFUNDS", "FEDFUNDS", "us_rate.csv", "미국 기준금리 환경"),
    MacroSpec("usd_krw", "FDR", "USD/KRW", "Close", "usd_krw.csv", "원달러 환율 환경"),
]
EXTENDED_MACRO_SPECS = [
    MacroSpec("kospi", "yfinance", "^KS11", "Close", "kospi.csv", "국내 시장 공통 흐름"),
    MacroSpec("kospi200", "yfinance", "^KS200", "Close", "kospi200.csv", "국내 대형주 공통 흐름"),
    MacroSpec("kosdaq", "yfinance", "^KQ11", "Close", "kosdaq.csv", "국내 성장주/기술주 심리"),
    MacroSpec("sp500", "FDR", "S&P500", "Close", "sp500.csv", "미국 전체 위험자산 베타"),
    MacroSpec("smh", "FDR", "SMH", "Close", "smh.csv", "거래 가능한 반도체 ETF 프록시"),
    MacroSpec("soxx", "FDR", "SOXX", "Close", "soxx.csv", "거래 가능한 반도체 ETF 프록시"),
    MacroSpec("dow_jones", "FDR", "DJI", "Close", "dow_jones.csv", "미국 대형 경기주 심리"),
    MacroSpec("russell2000", "FDR", "RUT", "Close", "russell2000.csv", "미국 소형주 위험선호"),
    MacroSpec("shanghai_composite", "FDR", "SSEC", "Close", "shanghai_composite.csv", "중국 수요와 아시아 제조업 심리"),
    MacroSpec("hang_seng", "FDR", "HSI", "Close", "hang_seng.csv", "중국/홍콩 위험자산 심리"),
    MacroSpec("nikkei225", "FDR", "N225", "Close", "nikkei225.csv", "일본 장비/소재 밸류체인 프록시"),
    MacroSpec("usd_cny", "FDR", "USD/CNY", "Close", "usd_cny.csv", "중국 환율 및 수요 스트레스"),
    MacroSpec("usd_jpy", "FDR", "USD/JPY", "Close", "usd_jpy.csv", "일본 환율 및 캐리/위험선호"),
    MacroSpec("cny_krw", "FDR", "CNY/KRW", "Close", "cny_krw.csv", "원위안 환율 관계"),
    MacroSpec("us_5y_treasury", "FDR", "US5YT", "Close", "us_5y_treasury.csv", "미국 중기 금리"),
    MacroSpec("us_10y_treasury", "FDR", "US10YT", "Close", "us_10y_treasury.csv", "미국 장기 금리"),
    MacroSpec("us_30y_treasury", "FDR", "US30YT", "Close", "us_30y_treasury.csv", "미국 초장기 금리"),
    MacroSpec("dollar_index", "FDR", "^NYICDX", "Close", "dollar_index.csv", "달러 인덱스"),
    MacroSpec("wti_crude_oil", "FDR", "CL=F", "Close", "wti_crude_oil.csv", "유가/인플레이션 프록시"),
    MacroSpec("brent_crude_oil", "FDR", "BZ=F", "Close", "brent_crude_oil.csv", "유가/인플레이션 프록시"),
    MacroSpec("natural_gas", "FDR", "NG=F", "Close", "natural_gas.csv", "전력비/인플레이션 프록시"),
    MacroSpec("gold", "FDR", "GC=F", "Close", "gold.csv", "위험회피와 실질금리 프록시"),
    MacroSpec("silver", "FDR", "SI=F", "Close", "silver.csv", "산업재와 위험회피 프록시"),
    MacroSpec("copper", "FDR", "HG=F", "Close", "copper.csv", "제조업과 AI 인프라 투자 심리"),
    MacroSpec("us_10y_treasury_fred", "FRED", "FRED:DGS10", "DGS10", "us_10y_treasury_fred.csv", "미국 장기 할인율"),
    MacroSpec("us_2y_treasury_fred", "FRED", "FRED:DGS2", "DGS2", "us_2y_treasury_fred.csv", "통화정책 기대"),
    MacroSpec("us_10y_2y_spread", "FRED", "FRED:T10Y2Y", "T10Y2Y", "us_10y_2y_spread.csv", "경기 국면과 수익률곡선"),
    MacroSpec("broad_dollar_index", "FRED", "FRED:DTWEXBGS", "DTWEXBGS", "broad_dollar_index.csv", "글로벌 달러 유동성"),
    MacroSpec("industrial_production", "FRED", "FRED:INDPRO", "INDPRO", "industrial_production.csv", "미국 산업생산 경기 사이클"),
    MacroSpec(
        "semiconductor_electronic_component_industrial_production",
        "FRED",
        "FRED:IPG3344S",
        "IPG3344S",
        "semiconductor_electronic_component_industrial_production.csv",
        "반도체·전자부품 생산 사이클",
    ),
    MacroSpec(
        "semiconductor_related_device_ppi",
        "FRED",
        "FRED:PCU334413334413P",
        "PCU334413334413P",
        "semiconductor_related_device_ppi.csv",
        "반도체 관련 생산자 가격",
    ),
    MacroSpec(
        "computers_electronic_products_new_orders",
        "FRED",
        "FRED:A34SNO",
        "A34SNO",
        "computers_electronic_products_new_orders.csv",
        "전방 IT 수요 프록시",
    ),
    MacroSpec("vix_fred", "FRED", "FRED:VIXCLS", "VIXCLS", "vix_fred.csv", "FRED 기준 VIX 보조 소스"),
    MacroSpec("nasdaq_composite_fred", "FRED", "FRED:NASDAQCOM", "NASDAQCOM", "nasdaq_composite_fred.csv", "FRED 기준 NASDAQ 보조 소스"),
]
FINANCIAL_SPECS = [
    FinancialSpec(
        "samsung_electronics_finstate_y",
        "FDR/NAVER",
        "NAVER/FINSTATE-Y/005930",
        "annual",
        "samsung_electronics_finstate_y.csv",
        "삼성전자 연간 재무제표 스냅샷",
    ),
    FinancialSpec(
        "samsung_electronics_finstate_q",
        "FDR/NAVER",
        "NAVER/FINSTATE-Q/005930",
        "quarterly",
        "samsung_electronics_finstate_q.csv",
        "삼성전자 분기 재무제표 스냅샷",
    ),
    FinancialSpec(
        "sk_hynix_finstate_y",
        "FDR/NAVER",
        "NAVER/FINSTATE-Y/000660",
        "annual",
        "sk_hynix_finstate_y.csv",
        "SK하이닉스 연간 재무제표 스냅샷",
    ),
    FinancialSpec(
        "sk_hynix_finstate_q",
        "FDR/NAVER",
        "NAVER/FINSTATE-Q/000660",
        "quarterly",
        "sk_hynix_finstate_q.csv",
        "SK하이닉스 분기 재무제표 스냅샷",
    ),
]
NVIDIA_SEC_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json"
EXTERNAL_ARCHIVE_SPECS = [
    ExternalSourceSpec(
        "sia_market_data_page",
        "external_public",
        "https://www.semiconductors.org/data-resources/market-data/",
        "sia_market_data_page.html",
        "archive",
        "SIA/WSTS 반도체 시장 데이터 공개 페이지",
    ),
    ExternalSourceSpec(
        "semi_billings_page",
        "external_public",
        "https://www.semi.org/en/products-services/market-data/equipment/billings-report",
        "semi_billings_page.html",
        "archive",
        "SEMI 장비 billings 공개 페이지",
    ),
    ExternalSourceSpec(
        "trendforce_dram_prices_page",
        "external_public",
        "https://www.trendforce.com/price",
        "trendforce_dram_prices_page.html",
        "archive",
        "TrendForce/DRAMeXchange 가격 페이지",
    ),
    ExternalSourceSpec(
        "tsmc_monthly_revenue_page",
        "external_public",
        "https://investor.tsmc.com/english/monthly-revenue",
        "tsmc_monthly_revenue_page.html",
        "archive",
        "TSMC 월간 매출 공개 페이지",
    ),
    ExternalSourceSpec(
        "nvidia_investor_events_page",
        "external_public",
        "https://investor.nvidia.com/events-and-presentations/events-and-presentations/default.aspx",
        "nvidia_investor_events_page.html",
        "archive",
        "NVIDIA 이벤트/프레젠테이션 페이지",
    ),
    ExternalSourceSpec(
        "chips_act_nist_page",
        "external_public",
        "https://www.nist.gov/chips",
        "chips_act_nist_page.html",
        "archive",
        "CHIPS Act 정책 이벤트 페이지",
    ),
]
CORE_MACRO_FEATURE_NAMES = [spec.feature_name for spec in CORE_MACRO_SPECS]
ALL_MACRO_SPECS = CORE_MACRO_SPECS + EXTENDED_MACRO_SPECS
CORE_MACRO_BUNDLE_SYMBOL = ",".join(spec.provider_symbol for spec in CORE_MACRO_SPECS)
CORE_MACRO_DAILY_GRID_FEATURE_NAMES = ["nasdaq", "sox", "vix", "usd_krw"]
CORE_MACRO_LOW_FREQUENCY_FEATURE_NAMES = [
    feature_name
    for feature_name in CORE_MACRO_FEATURE_NAMES
    if feature_name not in CORE_MACRO_DAILY_GRID_FEATURE_NAMES
]
DAILY_COVERAGE_TOLERANCE_DAYS = 7
LOW_FREQUENCY_COVERAGE_TOLERANCE_DAYS = 45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="제안서 기준 1차 데이터 수집 파이프라인",
    )
    parser.add_argument("--start", default=START, help="수집 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=END, help="수집 종료일 (YYYY-MM-DD)")
    parser.add_argument("--skip-raw", action="store_true", help="raw CSV 저장 생략")
    parser.add_argument(
        "--skip-alpha",
        action="store_true",
        help="국내 수급 데이터 수집 생략",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        RAW_ROOT / "prices",
        RAW_ROOT / "macro",
        RAW_ROOT / "alpha",
        RAW_ROOT / "financials",
        RAW_ROOT / "external" / "sec",
        RAW_ROOT / "external" / "public_pages",
        RAW_ROOT / "runs",
        METADATA_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _with_date_index(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.index = pd.to_datetime(data.index)
    data.index.name = "date"
    return data.sort_index()


def _records_from_indexed_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    records_df = df.reset_index()
    if "date" not in records_df.columns:
        records_df = records_df.rename(columns={records_df.columns[0]: "date"})
    records_df["date"] = pd.to_datetime(records_df["date"]).dt.date
    records_df = records_df.where(pd.notnull(records_df), None)
    return records_df.to_dict(orient="records")


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True, encoding="utf-8-sig")


def _run_raw_dir(run_id: str) -> Path:
    return RAW_ROOT / "runs" / run_id


def _date_bounds(df: pd.DataFrame) -> tuple[str, str]:
    if df.empty:
        return "", ""
    index = pd.to_datetime(df.index).dropna()
    if index.empty:
        return "", ""
    return index.min().date().isoformat(), index.max().date().isoformat()


def _append_status(
        rows: list[dict[str, Any]],
        *,
        run_context: dict[str, Any],
        category: str,
        name: str,
        source: str,
        symbol: str,
        status: str,
        rows_count: int = 0,
        file_path: str = "",
        data_start: str = "",
        data_end: str = "",
        note: str = "",
) -> None:
    rows.append(
        {
            **run_context,
            "category": category,
            "name": name,
            "source": source,
            "symbol": symbol,
            "status": status,
            "rows": rows_count,
            "file_path": file_path,
            "data_start": data_start,
            "data_end": data_end,
            "note": note,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def _append_failure(
        rows: list[dict[str, Any]],
        *,
        run_context: dict[str, Any],
        category: str,
        name: str,
        source: str,
        symbol: str,
        error: Exception,
        note: str = "",
) -> None:
    _append_status(
        rows,
        run_context=run_context,
        category=category,
        name=name,
        source=source,
        symbol=symbol,
        status="failed",
        rows_count=0,
        file_path="",
        note=f"{note} | {type(error).__name__}: {error}" if note else f"{type(error).__name__}: {error}",
    )


def _exclusive_end_for_yfinance(end: str) -> str:
    return (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")


def _read_fred_series(spec: MacroSpec, start: str, end: str) -> pd.Series:
    series_id = spec.provider_symbol.removeprefix("FRED:")
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}&coed={end}"
    )
    response = requests.get(url, timeout=20, verify=certifi.where())
    response.raise_for_status()
    data = pd.read_csv(StringIO(response.text))
    series = data.set_index("observation_date")[spec.provider_column].rename(spec.feature_name)
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def collect_stock_prices(
        start: str,
        end: str,
        status_rows: list[dict[str, Any]],
        run_context: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}

    for spec in KR_STOCKS:
        try:
            df = fdr.DataReader(spec.provider_symbol, start, end)[["Open", "High", "Low", "Close", "Volume"]]
            df.columns = df.columns.str.lower()
            df["ticker"] = spec.ticker
            frames[spec.name] = _with_date_index(df)
        except Exception as error:
            _append_failure(
                status_rows,
                run_context=run_context,
                category="stock_price",
                name=spec.name,
                source=spec.source,
                symbol=spec.provider_symbol,
                error=error,
                note="collect-stage",
            )

    for spec in US_STOCKS:
        try:
            df = yf.download(
                spec.provider_symbol,
                start=start,
                end=_exclusive_end_for_yfinance(end),
                auto_adjust=False,
                multi_level_index=False,
                progress=False,
            )[["Open", "High", "Low", "Close", "Volume"]]
            df.columns = df.columns.str.lower()
            df["ticker"] = spec.ticker
            frames[spec.name] = _with_date_index(df)
        except Exception as error:
            _append_failure(
                status_rows,
                run_context=run_context,
                category="stock_price",
                name=spec.name,
                source=spec.source,
                symbol=spec.provider_symbol,
                error=error,
                note="collect-stage",
            )

    return frames


def collect_macro_features(
        start: str,
        end: str,
        status_rows: list[dict[str, Any]],
        run_context: dict[str, Any],
) -> pd.DataFrame:
    series_list: list[pd.Series] = []

    for spec in ALL_MACRO_SPECS:
        try:
            if spec.source == "FRED":
                series = _read_fred_series(spec, start, end)
            elif spec.source == "yfinance":
                data = yf.download(
                    spec.provider_symbol,
                    start=start,
                    end=_exclusive_end_for_yfinance(end),
                    auto_adjust=False,
                    multi_level_index=False,
                    progress=False,
                )
                series = data[spec.provider_column].rename(spec.feature_name)
                series.index = pd.to_datetime(series.index)
            else:
                data = fdr.DataReader(spec.provider_symbol, start, end)
                series = data[spec.provider_column].rename(spec.feature_name)
                series.index = pd.to_datetime(series.index)
            series_list.append(series.sort_index())
        except Exception as error:
            _append_failure(
                status_rows,
                run_context=run_context,
                category="macro_series",
                name=spec.feature_name,
                source=spec.source,
                symbol=spec.provider_symbol,
                error=error,
                note="collect-stage",
            )

    macro_df = pd.concat(series_list, axis=1) if series_list else pd.DataFrame()
    macro_df.index.name = "date"
    return macro_df


def collect_alpha_features(
        start: str,
        end: str,
        status_rows: list[dict[str, Any]],
        run_context: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    try:
        from pykrx import stock
    except Exception as error:
        for spec in KR_STOCKS:
            _append_failure(
                status_rows,
                run_context=run_context,
                category="alpha",
                name=f"{spec.name}_alpha",
                source="PyKRX",
                symbol=spec.ticker,
                error=error,
                note="collect-stage | optional-dependency-missing",
            )
        return frames

    for spec in KR_STOCKS:
        try:
            data = stock.get_market_trading_value_by_date(
                start.replace("-", ""),
                end.replace("-", ""),
                spec.ticker,
            )
            required_columns = ["외국인합계", "기관합계"]
            if data is None or data.empty:
                raise ValueError("empty dataframe")
            missing_columns = [column for column in required_columns if column not in data.columns]
            if missing_columns:
                raise ValueError(f"missing columns: {','.join(missing_columns)}")
            df = data[["외국인합계", "기관합계"]].rename(
                columns={
                    "외국인합계": "foreign_net_buy_value",
                    "기관합계": "institution_net_buy_value",
                }
            )
            df["ticker"] = spec.ticker
            frames[spec.name] = _with_date_index(df)
        except Exception as error:
            _append_failure(
                status_rows,
                run_context=run_context,
                category="alpha",
                name=f"{spec.name}_alpha",
                source="PyKRX",
                symbol=spec.ticker,
                error=error,
                note="collect-stage",
            )

    return frames


def collect_financial_features(
        status_rows: list[dict[str, Any]],
        run_context: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for spec in FINANCIAL_SPECS:
        try:
            df = fdr.SnapDataReader(spec.provider_symbol)
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            frames[spec.feature_name] = df.copy()
        except Exception as error:
            _append_failure(
                status_rows,
                run_context=run_context,
                category="financials",
                name=spec.feature_name,
                source=spec.source,
                symbol=spec.provider_symbol,
                error=error,
                note=f"collect-stage | optional-extended | frequency={spec.frequency}",
            )
    return frames


def _extract_nvidia_companyfacts(payload: dict[str, Any]) -> pd.DataFrame:
    facts = payload.get("facts", {}).get("us-gaap", {})
    tags = {
        "Revenues": "revenue",
        "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue_contract",
        "GrossProfit": "gross_profit",
        "OperatingIncomeLoss": "operating_income",
        "NetIncomeLoss": "net_income",
        "InventoryNet": "inventory",
        "ResearchAndDevelopmentExpense": "r_and_d",
        "Assets": "assets",
        "Liabilities": "liabilities",
        "StockholdersEquity": "equity",
        "EarningsPerShareDiluted": "eps_diluted",
        "PaymentsToAcquirePropertyPlantAndEquipment": "capex_purchase_ppe",
    }
    rows: list[dict[str, Any]] = []
    for tag, metric in tags.items():
        units = facts.get(tag, {}).get("units", {})
        for unit, values in units.items():
            for value in values:
                rows.append(
                    {
                        "metric": metric,
                        "tag": tag,
                        "unit": unit,
                        "fy": value.get("fy"),
                        "fp": value.get("fp"),
                        "form": value.get("form"),
                        "filed": value.get("filed"),
                        "start": value.get("start"),
                        "end": value.get("end"),
                        "value": value.get("val"),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no selected NVIDIA SEC facts found")
    return df.sort_values(["metric", "end", "filed"]).reset_index(drop=True)


def collect_nvidia_sec_companyfacts(
        status_rows: list[dict[str, Any]],
        run_context: dict[str, Any],
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    try:
        response = requests.get(
            NVIDIA_SEC_URL,
            headers={"User-Agent": "semiconductor-stock-factor-project contact@example.com"},
            timeout=30,
            verify=certifi.where(),
        )
        response.raise_for_status()
        payload = response.json()
        return _extract_nvidia_companyfacts(payload), payload
    except Exception as error:
        _append_failure(
            status_rows,
            run_context=run_context,
            category="financials",
            name="nvidia_sec_companyfacts",
            source="SEC",
            symbol=NVIDIA_SEC_URL,
            error=error,
            note="collect-stage | optional-extended | source_dependent",
        )
        return None, None


def collect_external_archive_pages(
        status_rows: list[dict[str, Any]],
        run_context: dict[str, Any],
) -> dict[str, str]:
    pages: dict[str, str] = {}
    session = requests.Session()
    for spec in EXTERNAL_ARCHIVE_SPECS:
        try:
            response = session.get(
                spec.url,
                headers={"User-Agent": "semiconductor-stock-factor-project contact@example.com"},
                timeout=30,
                verify=certifi.where(),
            )
            response.raise_for_status()
            pages[spec.name] = response.text
        except Exception as error:
            _append_failure(
                status_rows,
                run_context=run_context,
                category="external_archive",
                name=spec.name,
                source=spec.source,
                symbol=spec.url,
                error=error,
                note=f"collect-stage | optional-{spec.priority}",
            )
    return pages


def persist_stock_frames(
        frames: dict[str, pd.DataFrame],
        status_rows: list[dict[str, Any]],
        *,
        run_context: dict[str, Any],
        skip_raw: bool,
) -> None:
    for spec in KR_STOCKS + US_STOCKS:
        if spec.name not in frames:
            continue
        df = frames[spec.name]
        if df.empty:
            _append_status(
                status_rows,
                run_context=run_context,
                category="stock_price",
                name=spec.name,
                source=spec.source,
                symbol=spec.provider_symbol,
                status="empty",
                rows_count=0,
                note="persist-stage | empty-frame",
            )
            continue
        file_path = _run_raw_dir(run_context["run_id"]) / "prices" / f"{spec.name}.csv"
        try:
            if not skip_raw:
                _save_csv(df, file_path)
            data_start, data_end = _date_bounds(df)
            _append_status(
                status_rows,
                run_context=run_context,
                category="stock_price",
                name=spec.name,
                source=spec.source,
                symbol=spec.provider_symbol,
                status="skipped" if skip_raw else "success",
                rows_count=len(df),
                file_path="" if skip_raw else str(file_path.relative_to(PROJECT_ROOT)),
                data_start=data_start,
                data_end=data_end,
                note="proposal-core | raw-write-skipped" if skip_raw else "proposal-core",
            )
        except Exception as error:
            _append_failure(
                status_rows,
                run_context=run_context,
                category="stock_price",
                name=spec.name,
                source=spec.source,
                symbol=spec.provider_symbol,
                error=error,
                note="persist-stage",
            )


def persist_macro_frame(
        macro_df: pd.DataFrame,
        status_rows: list[dict[str, Any]],
        *,
        run_context: dict[str, Any],
        skip_raw: bool,
) -> None:
    run_macro_dir = _run_raw_dir(run_context["run_id"]) / "macro"
    series_statuses: list[dict[str, Any]] = []
    for spec in ALL_MACRO_SPECS:
        tier = "core" if spec in CORE_MACRO_SPECS else "extended"
        existing_failed_series = any(
            row.get("category") == "macro_series"
            and row.get("name") == spec.feature_name
            and row.get("status") == "failed"
            for row in status_rows
        )
        if spec.feature_name not in macro_df.columns:
            if existing_failed_series:
                continue
            series_statuses.append(
                {
                    "name": spec.feature_name,
                    "source": spec.source,
                    "symbol": spec.provider_symbol,
                    "rows_count": 0,
                    "file_path": "",
                    "data_start": "",
                    "data_end": "",
                    "status": "failed",
                    "note": f"persist-stage | macro-series | missing-column | tier={tier}",
                }
            )
            continue

        series_df = macro_df[[spec.feature_name]].dropna()
        file_path = run_macro_dir / spec.file_name
        status = "success" if not series_df.empty else "empty"
        saved_file_path = ""
        note = f"persist-stage | macro-series | tier={tier}"

        try:
            if not skip_raw and not series_df.empty:
                _save_csv(series_df, file_path)
                saved_file_path = str(file_path.relative_to(PROJECT_ROOT))
            elif skip_raw and not series_df.empty:
                status = "skipped"
                note = f"{note} | raw-write-skipped"
        except Exception as error:
            status = "failed"
            note = f"{note} | file-write-error | {type(error).__name__}: {error}"

        series_statuses.append(
            {
                "name": spec.feature_name,
                "source": spec.source,
                "symbol": spec.provider_symbol,
                "rows_count": len(series_df),
                "file_path": saved_file_path,
                "data_start": _date_bounds(series_df)[0],
                "data_end": _date_bounds(series_df)[1],
                "status": status,
                "note": note,
            }
        )

    for item in series_statuses:
        _append_status(
            status_rows,
            run_context=run_context,
            category="macro_series",
            name=item["name"],
            source=item["source"],
            symbol=item["symbol"],
            status=item["status"],
            rows_count=item["rows_count"],
            file_path=item["file_path"],
            data_start=item["data_start"],
            data_end=item["data_end"],
            note=item["note"],
        )

    if macro_df.empty:
        _append_status(
            status_rows,
            run_context=run_context,
            category="macro",
            name="core_macro_features",
            source="internal_bundle",
            symbol=CORE_MACRO_BUNDLE_SYMBOL,
            status="empty",
            rows_count=0,
            note="persist-stage | empty-frame",
        )
        return
    missing_columns = [
        feature_name
        for feature_name in CORE_MACRO_FEATURE_NAMES
        if feature_name not in macro_df.columns
    ]
    if missing_columns:
        _append_status(
            status_rows,
            run_context=run_context,
            category="macro",
            name="core_macro_features",
            source="internal_bundle",
            symbol=CORE_MACRO_BUNDLE_SYMBOL,
            status="failed",
            rows_count=len(macro_df),
            note=f"persist-stage | missing-core-columns={','.join(missing_columns)}",
        )
        return

    empty_core_columns = [
        feature_name
        for feature_name in CORE_MACRO_FEATURE_NAMES
        if feature_name in macro_df.columns and not macro_df[feature_name].notna().any()
    ]
    daily_grid_frames = [
        macro_df[[feature_name]].dropna().copy()
        for feature_name in CORE_MACRO_DAILY_GRID_FEATURE_NAMES
        if feature_name in macro_df.columns
    ]
    if daily_grid_frames:
        core_daily_grid = pd.concat(daily_grid_frames, axis=1).index.unique().sort_values()
    else:
        core_daily_grid = pd.Index([], name="date")
    low_frequency_frames = [
        macro_df[[feature_name]].dropna().copy()
        for feature_name in CORE_MACRO_LOW_FREQUENCY_FEATURE_NAMES
        if feature_name in macro_df.columns
    ]
    if low_frequency_frames:
        low_frequency_grid = pd.concat(low_frequency_frames, axis=1).index.unique().sort_values()
        core_bundle_grid = core_daily_grid.union(low_frequency_grid).sort_values()
    else:
        core_bundle_grid = core_daily_grid
    core_macro_df = macro_df[CORE_MACRO_FEATURE_NAMES].reindex(core_bundle_grid).copy()
    if not core_macro_df.empty:
        core_macro_df.index.name = "date"
    if empty_core_columns:
        _append_status(
            status_rows,
            run_context=run_context,
            category="macro",
            name="core_macro_features",
            source="internal_bundle",
            symbol=CORE_MACRO_BUNDLE_SYMBOL,
            status="failed",
            rows_count=len(macro_df),
            note=f"persist-stage | empty-core-columns={','.join(empty_core_columns)}",
        )
        return
    file_path = run_macro_dir / "core_macro_features.csv"
    try:
        if not skip_raw:
            _save_csv(core_macro_df, file_path)
        data_start, data_end = _date_bounds(core_macro_df)
        _append_status(
            status_rows,
            run_context=run_context,
            category="macro",
            name="core_macro_features",
            source="internal_bundle",
            symbol=CORE_MACRO_BUNDLE_SYMBOL,
            status="skipped" if skip_raw else "success",
            rows_count=len(core_macro_df),
            file_path="" if skip_raw else str(file_path.relative_to(PROJECT_ROOT)),
            data_start=data_start,
            data_end=data_end,
            note=(
                    f"proposal-core | columns={','.join(core_macro_df.columns)}"
                    " | daily-raw-grid-preserved | core-daily-grid=nasdaq,sox,vix,usd_krw"
                    " | low-frequency-dates-preserved=us_rate"
                    " | fill/align-in-preprocess"
                    + (" | raw-write-skipped" if skip_raw else "")
            ),
        )
    except Exception as error:
        _append_failure(
            status_rows,
            run_context=run_context,
            category="macro",
            name="core_macro_features",
            source="internal_bundle",
            symbol=CORE_MACRO_BUNDLE_SYMBOL,
            error=error,
            note="persist-stage",
        )


def persist_alpha_frames(
        frames: dict[str, pd.DataFrame],
        status_rows: list[dict[str, Any]],
        *,
        run_context: dict[str, Any],
        skip_raw: bool,
) -> None:
    for spec in KR_STOCKS:
        if spec.name not in frames:
            continue
        df = frames[spec.name]
        if df.empty:
            _append_status(
                status_rows,
                run_context=run_context,
                category="alpha",
                name=f"{spec.name}_alpha",
                source="PyKRX",
                symbol=spec.ticker,
                status="empty",
                rows_count=0,
                note="persist-stage | empty-frame | unit=net_buy_value",
            )
            continue
        file_path = _run_raw_dir(run_context["run_id"]) / "alpha" / f"{spec.name}_alpha.csv"
        try:
            if not skip_raw:
                _save_csv(df, file_path)
            data_start, data_end = _date_bounds(df)
            _append_status(
                status_rows,
                run_context=run_context,
                category="alpha",
                name=f"{spec.name}_alpha",
                source="PyKRX",
                symbol=spec.ticker,
                status="skipped" if skip_raw else "success",
                rows_count=len(df),
                file_path="" if skip_raw else str(file_path.relative_to(PROJECT_ROOT)),
                data_start=data_start,
                data_end=data_end,
                note=(
                    "proposal-core-domestic-only | unit=net_buy_value | raw-write-skipped"
                    if skip_raw
                    else "proposal-core-domestic-only | unit=net_buy_value"
                ),
            )
        except Exception as error:
            _append_failure(
                status_rows,
                run_context=run_context,
                category="alpha",
                name=f"{spec.name}_alpha",
                source="PyKRX",
                symbol=spec.ticker,
                error=error,
                note="persist-stage",
            )


def persist_financial_frames(
        frames: dict[str, pd.DataFrame],
        status_rows: list[dict[str, Any]],
        *,
        run_context: dict[str, Any],
        skip_raw: bool,
) -> None:
    run_financials_dir = _run_raw_dir(run_context["run_id"]) / "financials"
    for spec in FINANCIAL_SPECS:
        if spec.feature_name not in frames:
            continue
        df = frames[spec.feature_name]
        if df.empty:
            _append_status(
                status_rows,
                run_context=run_context,
                category="financials",
                name=spec.feature_name,
                source=spec.source,
                symbol=spec.provider_symbol,
                status="empty",
                rows_count=0,
                note=f"persist-stage | optional-extended | frequency={spec.frequency}",
            )
            continue
        file_path = run_financials_dir / spec.file_name
        try:
            if not skip_raw:
                _save_csv(df, file_path)
            data_start, data_end = _date_bounds(df)
            _append_status(
                status_rows,
                run_context=run_context,
                category="financials",
                name=spec.feature_name,
                source=spec.source,
                symbol=spec.provider_symbol,
                status="skipped" if skip_raw else "success",
                rows_count=len(df),
                file_path="" if skip_raw else str(file_path.relative_to(PROJECT_ROOT)),
                data_start=data_start,
                data_end=data_end,
                note=(
                    f"optional-extended | frequency={spec.frequency} | raw-write-skipped"
                    if skip_raw
                    else f"optional-extended | frequency={spec.frequency}"
                ),
            )
        except Exception as error:
            _append_failure(
                status_rows,
                run_context=run_context,
                category="financials",
                name=spec.feature_name,
                source=spec.source,
                symbol=spec.provider_symbol,
                error=error,
                note="persist-stage | optional-extended",
            )


def persist_nvidia_sec_companyfacts(
        df: pd.DataFrame | None,
        payload: dict[str, Any] | None,
        status_rows: list[dict[str, Any]],
        *,
        run_context: dict[str, Any],
        skip_raw: bool,
) -> None:
    if df is None or df.empty:
        return
    run_sec_dir = _run_raw_dir(run_context["run_id"]) / "external" / "sec"
    csv_path = run_sec_dir / "nvidia_sec_key_financials.csv"
    try:
        if not skip_raw:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        _append_status(
            status_rows,
            run_context=run_context,
            category="financials",
            name="nvidia_sec_companyfacts",
            source="SEC",
            symbol=NVIDIA_SEC_URL,
            status="skipped" if skip_raw else "success",
            rows_count=len(df),
            file_path="" if skip_raw else str(csv_path.relative_to(PROJECT_ROOT)),
            data_start=str(df["end"].min()) if "end" in df.columns else "",
            data_end=str(df["end"].max()) if "end" in df.columns else "",
            note=(
                "optional-extended | source_dependent | raw-json-not-persisted | raw-write-skipped"
                if skip_raw
                else "optional-extended | source_dependent | raw-json-not-persisted"
            ),
        )
    except Exception as error:
        _append_failure(
            status_rows,
            run_context=run_context,
            category="financials",
            name="nvidia_sec_companyfacts",
            source="SEC",
            symbol=NVIDIA_SEC_URL,
            error=error,
            note="persist-stage | optional-extended",
        )


def persist_external_archive_pages(
        pages: dict[str, str],
        status_rows: list[dict[str, Any]],
        *,
        run_context: dict[str, Any],
        skip_raw: bool,
) -> None:
    run_pages_dir = _run_raw_dir(run_context["run_id"]) / "external" / "public_pages"
    for spec in EXTERNAL_ARCHIVE_SPECS:
        if spec.name not in pages:
            continue
        html = pages[spec.name]
        file_path = run_pages_dir / spec.file_name
        try:
            if not skip_raw:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(html, encoding="utf-8")
            _append_status(
                status_rows,
                run_context=run_context,
                category="external_archive",
                name=spec.name,
                source=spec.source,
                symbol=spec.url,
                status="skipped" if skip_raw else "success",
                rows_count=1,
                file_path="" if skip_raw else str(file_path.relative_to(PROJECT_ROOT)),
                note=(
                    f"optional-{spec.priority} | html-archive | bytes={len(html)} | raw-write-skipped"
                    if skip_raw
                    else f"optional-{spec.priority} | html-archive | bytes={len(html)}"
                ),
            )
        except Exception as error:
            _append_failure(
                status_rows,
                run_context=run_context,
                category="external_archive",
                name=spec.name,
                source=spec.source,
                symbol=spec.url,
                error=error,
                note=f"persist-stage | optional-{spec.priority}",
            )


def write_metadata(
        status_rows: list[dict[str, Any]],
        run_context: dict[str, Any],
        *,
        update_canonical_catalog: bool,
        update_adopted_manifest: bool,
) -> None:
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    status_df = pd.DataFrame(status_rows)
    if not status_df.empty:
        summary_path = METADATA_ROOT / "collection_summary.csv"
        write_header = not summary_path.exists()
        status_df.to_csv(
            summary_path,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8-sig",
        )
    run_metadata_dir = METADATA_ROOT / "runs" / run_context["run_id"]
    run_metadata_dir.mkdir(parents=True, exist_ok=True)
    status_df.to_csv(
        run_metadata_dir / "collection_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    source_catalog_rows = []
    for spec in KR_STOCKS + US_STOCKS:
        source_catalog_rows.append(
            {
                "run_id": run_context["run_id"],
                "category": "stock_price",
                "name": spec.name,
                "source": spec.source,
                "symbol": spec.provider_symbol,
                "priority": "core",
                "rationale": "분석 대상 종목 가격과 라벨 생성의 기준",
            }
        )
    for spec in CORE_MACRO_SPECS:
        source_catalog_rows.append(
            {
                "run_id": run_context["run_id"],
                "category": "macro_series",
                "name": spec.feature_name,
                "source": spec.source,
                "symbol": spec.provider_symbol,
                "priority": "core",
                "rationale": spec.rationale,
            }
        )
    source_catalog_rows.append(
        {
            "run_id": run_context["run_id"],
            "category": "macro",
            "name": "core_macro_features",
            "source": "internal_bundle",
            "symbol": CORE_MACRO_BUNDLE_SYMBOL,
            "priority": "core",
            "rationale": "전처리 단계로 넘기는 공식 core macro bundle",
        }
    )
    for spec in EXTENDED_MACRO_SPECS:
        source_catalog_rows.append(
            {
                "run_id": run_context["run_id"],
                "category": "macro_series",
                "name": spec.feature_name,
                "source": spec.source,
                "symbol": spec.provider_symbol,
                "priority": "extended",
                "rationale": spec.rationale,
            }
        )
    for spec in KR_STOCKS:
        source_catalog_rows.append(
            {
                "run_id": run_context["run_id"],
                "category": "alpha",
                "name": f"{spec.name}_alpha",
                "source": "PyKRX",
                "symbol": spec.ticker,
                "priority": "extended",
                "rationale": "국내 종목 외국인/기관 순매수 거래대금 흐름 (KRX credential 필요)",
            }
        )
    for spec in FINANCIAL_SPECS:
        source_catalog_rows.append(
            {
                "run_id": run_context["run_id"],
                "category": "financials",
                "name": spec.feature_name,
                "source": spec.source,
                "symbol": spec.provider_symbol,
                "priority": "extended",
                "rationale": spec.rationale,
            }
        )
    source_catalog_rows.append(
        {
            "run_id": run_context["run_id"],
            "category": "financials",
            "name": "nvidia_sec_companyfacts",
            "source": "SEC",
            "symbol": NVIDIA_SEC_URL,
            "priority": "extended",
            "rationale": "NVIDIA 매출, 이익, 재고, R&D, CapEx 관련 company facts",
        }
    )
    for spec in EXTERNAL_ARCHIVE_SPECS:
        source_catalog_rows.append(
            {
                "run_id": run_context["run_id"],
                "category": "external_archive",
                "name": spec.name,
                "source": spec.source,
                "symbol": spec.url,
                "priority": spec.priority,
                "rationale": spec.rationale,
            }
        )

    source_catalog_df = pd.DataFrame(source_catalog_rows)
    if update_canonical_catalog:
        canonical_source_catalog_df = source_catalog_df.drop(columns=["run_id"])
        canonical_source_catalog_df.to_csv(
            METADATA_ROOT / "source_catalog.csv",
            index=False,
            encoding="utf-8-sig",
        )
    source_catalog_df.to_csv(
        run_metadata_dir / "source_catalog.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if update_adopted_manifest:
        manifest_df = pd.DataFrame(
            [
                {
                    "run_id": run_context["run_id"],
                    "run_start": run_context["run_start"],
                    "run_end": run_context["run_end"],
                    "raw_run_path": str((_run_raw_dir(run_context["run_id"])).relative_to(PROJECT_ROOT)),
                    "run_metadata_path": str(run_metadata_dir.relative_to(PROJECT_ROOT)),
                    "collection_summary_path": str((run_metadata_dir / "collection_summary.csv").relative_to(PROJECT_ROOT)),
                    "source_catalog_path": str((METADATA_ROOT / "source_catalog.csv").relative_to(PROJECT_ROOT)),
                    "adopted_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
        )
        manifest_df.to_csv(
            METADATA_ROOT / "current_raw_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )


def _core_artifact_failures(status_rows: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    debug_skip_raw = any(
        row.get("category") == "pipeline"
        and row.get("name") == "raw_output_contract"
        and row.get("status") == "failed"
        for row in status_rows
    )

    required_stock_names = {spec.name for spec in KR_STOCKS + US_STOCKS}

    latest_by_name: dict[str, dict[str, Any]] = {}
    for row in status_rows:
        latest_by_name[row["name"]] = row

    for name in sorted(required_stock_names):
        row = latest_by_name.get(name)
        if row is None:
            failures.append(f"missing-core-stock:{name}")
            continue
        if debug_skip_raw and row.get("status") == "skipped":
            continue
        if row.get("status") != "success":
            failures.append(f"missing-core-stock:{name}")

    core_macro_row = latest_by_name.get("core_macro_features")
    if core_macro_row is None:
        failures.append("missing-core-macro:core_macro_features")
    elif not (debug_skip_raw and core_macro_row.get("status") == "skipped") and core_macro_row.get(
            "status") != "success":
        failures.append("missing-core-macro:core_macro_features")

    latest_macro_series_by_name = {
        row["name"]: row
        for row in status_rows
        if row.get("category") == "macro_series"
    }
    for name in sorted(CORE_MACRO_FEATURE_NAMES):
        row = latest_macro_series_by_name.get(name)
        if row is None:
            failures.append(f"missing-core-macro-series:{name}")
            continue
        if debug_skip_raw and row.get("status") == "skipped":
            continue
        if row.get("status") != "success":
            failures.append(f"missing-core-macro-series:{name}")

    return failures


def _coverage_failure(
        row: dict[str, Any] | None,
        *,
        name: str,
        requested_start: pd.Timestamp,
        requested_end: pd.Timestamp,
        tolerance_days: int,
) -> str | None:
    if row is None:
        return None
    if row.get("status") not in {"success", "skipped"}:
        return None
    data_start = row.get("data_start")
    data_end = row.get("data_end")
    if not data_start or not data_end:
        return f"missing-coverage-bounds:{name}"
    actual_start = pd.Timestamp(data_start)
    actual_end = pd.Timestamp(data_end)
    if actual_start > requested_start + pd.Timedelta(days=tolerance_days):
        return f"truncated-start:{name}:{actual_start.date().isoformat()}"
    if actual_end < requested_end - pd.Timedelta(days=tolerance_days):
        return f"truncated-end:{name}:{actual_end.date().isoformat()}"
    return None


def _coverage_warnings(status_rows: list[dict[str, Any]], run_context: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    requested_start = pd.Timestamp(run_context["run_start"])
    requested_end = pd.Timestamp(run_context["run_end"])

    latest_by_name: dict[str, dict[str, Any]] = {}
    for row in status_rows:
        latest_by_name[row["name"]] = row

    for spec in KR_STOCKS + US_STOCKS:
        failure = _coverage_failure(
            latest_by_name.get(spec.name),
            name=spec.name,
            requested_start=requested_start,
            requested_end=requested_end,
            tolerance_days=DAILY_COVERAGE_TOLERANCE_DAYS,
        )
        if failure:
            warnings.append(failure)

    for feature_name in CORE_MACRO_DAILY_GRID_FEATURE_NAMES:
        failure = _coverage_failure(
            latest_by_name.get(feature_name),
            name=feature_name,
            requested_start=requested_start,
            requested_end=requested_end,
            tolerance_days=DAILY_COVERAGE_TOLERANCE_DAYS,
        )
        if failure:
            warnings.append(failure)

    for feature_name in CORE_MACRO_LOW_FREQUENCY_FEATURE_NAMES:
        failure = _coverage_failure(
            latest_by_name.get(feature_name),
            name=feature_name,
            requested_start=requested_start,
            requested_end=requested_end,
            tolerance_days=LOW_FREQUENCY_COVERAGE_TOLERANCE_DAYS,
        )
        if failure:
            warnings.append(failure)

    return warnings


def run_collection(args: argparse.Namespace) -> None:
    status_rows: list[dict[str, Any]] = []
    pipeline_failed = False
    run_context = {
        "run_id": f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}",
        "run_start": args.start,
        "run_end": args.end,
        "skip_raw": args.skip_raw,
        "skip_alpha": args.skip_alpha,
    }

    try:
        ensure_dirs()

        stock_frames = collect_stock_prices(args.start, args.end, status_rows, run_context)
        persist_stock_frames(
            stock_frames,
            status_rows,
            run_context=run_context,
            skip_raw=args.skip_raw,
        )

        macro_df = collect_macro_features(args.start, args.end, status_rows, run_context)
        persist_macro_frame(
            macro_df,
            status_rows,
            run_context=run_context,
            skip_raw=args.skip_raw,
        )

        if args.skip_alpha:
            for spec in KR_STOCKS:
                _append_status(
                    status_rows,
                    run_context=run_context,
                    category="alpha",
                    name=f"{spec.name}_alpha",
                    source="PyKRX",
                    symbol=spec.ticker,
                    status="skipped",
                    note="requested-by-flag | unit=net_buy_value",
                )
        else:
            alpha_frames = collect_alpha_features(args.start, args.end, status_rows, run_context)
            persist_alpha_frames(
                alpha_frames,
                status_rows,
                run_context=run_context,
                skip_raw=args.skip_raw,
            )

        financial_frames = collect_financial_features(status_rows, run_context)
        persist_financial_frames(
            financial_frames,
            status_rows,
            run_context=run_context,
            skip_raw=args.skip_raw,
        )

        nvidia_sec_df, nvidia_sec_payload = collect_nvidia_sec_companyfacts(status_rows, run_context)
        persist_nvidia_sec_companyfacts(
            nvidia_sec_df,
            nvidia_sec_payload,
            status_rows,
            run_context=run_context,
            skip_raw=args.skip_raw,
        )

        external_pages = collect_external_archive_pages(status_rows, run_context)
        persist_external_archive_pages(
            external_pages,
            status_rows,
            run_context=run_context,
            skip_raw=args.skip_raw,
        )

        if args.skip_raw:
            pipeline_failed = True
            _append_status(
                status_rows,
                run_context=run_context,
                category="pipeline",
                name="raw_output_contract",
                source="internal",
                symbol="-",
                status="failed",
                rows_count=0,
                note="skip-raw is allowed only for debug collection, not official raw baseline adoption",
            )

        core_failures = _core_artifact_failures(status_rows)
        if core_failures:
            pipeline_failed = True
            _append_status(
                status_rows,
                run_context=run_context,
                category="pipeline",
                name="core_artifact_validation",
                source="internal",
                symbol="-",
                status="failed",
                rows_count=0,
                note=" | ".join(core_failures),
            )
        coverage_warnings = _coverage_warnings(status_rows, run_context)
        if coverage_warnings:
            _append_status(
                status_rows,
                run_context=run_context,
                category="pipeline",
                name="coverage_validation",
                source="internal",
                symbol="-",
                status="warning",
                rows_count=0,
                note=" | ".join(coverage_warnings),
            )
    except Exception as error:
        if not isinstance(error, SystemExit):
            pipeline_failed = True
            _append_failure(
                status_rows,
                run_context=run_context,
                category="pipeline",
                name="run_collection",
                source="internal",
                symbol="-",
                error=error,
                note="unexpected-top-level-error",
            )
        raise
    finally:
        write_metadata(
            status_rows,
            run_context,
            update_canonical_catalog=True,
            update_adopted_manifest=not pipeline_failed and not args.skip_raw,
        )
        if pipeline_failed:
            print("데이터 수집 파이프라인이 실패 메타데이터를 남기고 중단됨")
        else:
            print("데이터 수집 파이프라인 완료")

    if pipeline_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_collection(parse_args())
