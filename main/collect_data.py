from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    import FinanceDataReader as fdr
except ImportError as exc:
    raise SystemExit(
        "FinanceDataReader is not installed. Run: pip install -r requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
START = "2011-01-01"
END = None
SEC_HEADERS = {
    "User-Agent": "semiconductor-stock-factor-project yunsunghwang@example.com"
}


@dataclass(frozen=True)
class SeriesSpec:
    priority: int
    part: str
    name: str
    symbol: str
    source: str
    frequency: str
    save_dir: str
    description: str


FDR_SPECS = [
    SeriesSpec(1, "stock_price", "samsung_electronics", "005930", "FDR", "daily", "prices", "Samsung Electronics stock price"),
    SeriesSpec(1, "stock_price", "sk_hynix", "000660", "FDR", "daily", "prices", "SK Hynix stock price"),
    SeriesSpec(1, "stock_price", "nvidia", "NVDA", "FDR", "daily", "prices", "NVIDIA stock price"),
    SeriesSpec(1, "semi_index", "vanEck_semiconductor_etf_smh", "SMH", "FDR", "daily", "semi_indices", "Semiconductor ETF proxy for SOX/global semi cycle"),
    SeriesSpec(1, "semi_index", "ishares_semiconductor_etf_soxx", "SOXX", "FDR", "daily", "semi_indices", "Semiconductor ETF proxy for SOX"),
    SeriesSpec(1, "semi_index", "phlx_semiconductor_index_sox", "SOX", "FDR", "daily", "semi_indices", "PHLX Semiconductor Index direct symbol trial"),
    SeriesSpec(1, "semi_index", "phlx_semiconductor_index_yahoo", "^SOX", "FDR", "daily", "semi_indices", "PHLX Semiconductor Index Yahoo style symbol trial"),
    SeriesSpec(1, "global_index", "kospi", "KS11", "FDR", "daily", "global_indices", "Korea market common factor"),
    SeriesSpec(1, "global_index", "kospi200", "KS200", "FDR", "daily", "global_indices", "Korea large-cap market factor"),
    SeriesSpec(1, "global_index", "kosdaq", "KQ11", "FDR", "daily", "global_indices", "Korea growth/tech sentiment"),
    SeriesSpec(1, "global_index", "nasdaq_composite", "IXIC", "FDR", "daily", "global_indices", "US technology beta"),
    SeriesSpec(1, "global_index", "sp500", "S&P500", "FDR", "daily", "global_indices", "US risk asset common factor"),
    SeriesSpec(1, "global_index", "dow_jones", "DJI", "FDR", "daily", "global_indices", "US cyclical large-cap sentiment"),
    SeriesSpec(1, "global_index", "russell2000", "RUT", "FDR", "daily", "global_indices", "US small-cap risk appetite"),
    SeriesSpec(1, "global_index", "vix", "VIX", "FDR", "daily", "global_indices", "Risk aversion and volatility"),
    SeriesSpec(1, "global_index", "shanghai_composite", "SSEC", "FDR", "daily", "global_indices", "China demand and Asia manufacturing proxy"),
    SeriesSpec(1, "global_index", "hang_seng", "HSI", "FDR", "daily", "global_indices", "China/Hong Kong risk sentiment"),
    SeriesSpec(1, "global_index", "nikkei225", "N225", "FDR", "daily", "global_indices", "Japan equipment/material supply-chain proxy"),
    SeriesSpec(1, "fx", "usd_krw", "USD/KRW", "FDR", "daily", "fx_rates", "KRW FX factor"),
    SeriesSpec(1, "fx", "usd_cny", "USD/CNY", "FDR", "daily", "fx_rates", "China FX and demand stress proxy"),
    SeriesSpec(1, "fx", "usd_jpy", "USD/JPY", "FDR", "daily", "fx_rates", "Japan FX and carry/risk proxy"),
    SeriesSpec(1, "fx", "cny_krw", "CNY/KRW", "FDR", "daily", "fx_rates", "Korea-China FX relationship"),
    SeriesSpec(1, "rate_dollar", "us_5y_treasury", "US5YT", "FDR", "daily", "rates", "US 5-year treasury yield"),
    SeriesSpec(1, "rate_dollar", "us_10y_treasury", "US10YT", "FDR", "daily", "rates", "US 10-year treasury yield"),
    SeriesSpec(1, "rate_dollar", "us_30y_treasury", "US30YT", "FDR", "daily", "rates", "US 30-year treasury yield"),
    SeriesSpec(1, "rate_dollar", "dollar_index", "^NYICDX", "FDR", "daily", "rates", "US dollar index"),
    SeriesSpec(1, "future", "wti_crude_oil", "CL=F", "FDR", "daily", "futures", "WTI crude oil futures"),
    SeriesSpec(1, "future", "brent_crude_oil", "BZ=F", "FDR", "daily", "futures", "Brent crude oil futures"),
    SeriesSpec(1, "future", "natural_gas", "NG=F", "FDR", "daily", "futures", "Natural gas futures"),
    SeriesSpec(1, "future", "gold", "GC=F", "FDR", "daily", "futures", "Gold futures"),
    SeriesSpec(1, "future", "silver", "SI=F", "FDR", "daily", "futures", "Silver futures"),
    SeriesSpec(1, "future", "copper", "HG=F", "FDR", "daily", "futures", "Copper futures"),
]

FRED_SYMBOLS = {
    "FEDFUNDS": "federal_funds_rate",
    "DGS10": "us_10y_treasury_fred",
    "DGS2": "us_2y_treasury_fred",
    "T10Y2Y": "us_10y_2y_spread",
    "VIXCLS": "vix_fred",
    "NASDAQCOM": "nasdaq_composite_fred",
    "DTWEXBGS": "broad_dollar_index",
    "INDPRO": "industrial_production",
    "IPG3344S": "semiconductor_electronic_component_industrial_production",
    "PCU334413334413P": "semiconductor_related_device_ppi",
    "A34SNO": "computers_electronic_products_new_orders",
}

FINANCIAL_SPECS = [
    ("samsung_electronics_finstate_y", "NAVER/FINSTATE-Y/005930", "annual"),
    ("samsung_electronics_finstate_q", "NAVER/FINSTATE-Q/005930", "quarterly"),
    ("sk_hynix_finstate_y", "NAVER/FINSTATE-Y/000660", "annual"),
    ("sk_hynix_finstate_q", "NAVER/FINSTATE-Q/000660", "quarterly"),
]

EXTERNAL_PUBLIC_SOURCES = [
    {
        "priority": 2,
        "part": "financials",
        "name": "nvidia_sec_companyfacts",
        "url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json",
        "save_dir": "sec",
        "description": "NVIDIA SEC company facts for revenue, income, inventory, capex-related line items",
    },
    {
        "priority": 2,
        "part": "semi_supply_demand",
        "name": "sia_market_data_page",
        "url": "https://www.semiconductors.org/data-resources/market-data/",
        "save_dir": "public_pages",
        "description": "SIA/WSTS semiconductor market data landing page availability check",
    },
    {
        "priority": 2,
        "part": "semi_equipment",
        "name": "semi_billings_page",
        "url": "https://www.semi.org/en/products-services/market-data/equipment/billings-report",
        "save_dir": "public_pages",
        "description": "SEMI equipment billings landing page availability check",
    },
    {
        "priority": 2,
        "part": "memory_prices",
        "name": "trendforce_dram_prices_page",
        "url": "https://www.trendforce.com/price",
        "save_dir": "public_pages",
        "description": "TrendForce/DRAMeXchange memory price page availability check",
    },
    {
        "priority": 2,
        "part": "foundry_revenue",
        "name": "tsmc_monthly_revenue_page",
        "url": "https://investor.tsmc.com/english/monthly-revenue",
        "save_dir": "public_pages",
        "description": "TSMC monthly revenue landing page availability check",
    },
    {
        "priority": 3,
        "part": "event",
        "name": "nvidia_investor_events_page",
        "url": "https://investor.nvidia.com/events-and-presentations/events-and-presentations/default.aspx",
        "save_dir": "public_pages",
        "description": "NVIDIA investor events and presentation page availability check",
    },
    {
        "priority": 3,
        "part": "event",
        "name": "chips_act_nist_page",
        "url": "https://www.nist.gov/chips",
        "save_dir": "public_pages",
        "description": "CHIPS Act event/policy data source availability check",
    },
]


def ensure_dirs() -> None:
    folders = [
        DATA_ROOT / "raw" / "fdr" / folder
        for folder in [
            "prices",
            "financials",
            "semi_indices",
            "global_indices",
            "fx_rates",
            "rates",
            "futures",
            "fred",
        ]
    ]
    folders += [
        DATA_ROOT / "raw" / "external" / "sec",
        DATA_ROOT / "raw" / "external" / "public_pages",
        DATA_ROOT / "processed",
        DATA_ROOT / "metadata",
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=True, encoding="utf-8-sig")


def dataframe_info(df: pd.DataFrame) -> dict[str, Any]:
    index_min = "" if df.empty else str(df.index.min())
    index_max = "" if df.empty else str(df.index.max())
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "start_date": index_min,
        "end_date": index_max,
        "column_names": "|".join(map(str, df.columns)),
    }


def record_success(records: list[dict[str, Any]], spec: dict[str, Any], path: Path, df: pd.DataFrame) -> None:
    info = dataframe_info(df)
    records.append(
        {
            **spec,
            **info,
            "status": "success",
            "file_path": str(path.relative_to(PROJECT_ROOT)),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def record_failure(records: list[dict[str, Any]], spec: dict[str, Any], error: Exception | str) -> None:
    records.append(
        {
            **spec,
            "status": "failed",
            "error_type": type(error).__name__ if isinstance(error, Exception) else "CollectionError",
            "error_message": str(error),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def collect_fdr_series(successes: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for item in FDR_SPECS:
        spec = item.__dict__
        try:
            df = fdr.DataReader(item.symbol, START, END)
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            df.index.name = "date"
            output_path = DATA_ROOT / "raw" / "fdr" / item.save_dir / f"{item.name}.csv"
            save_dataframe(df, output_path)
            frames[item.name] = df
            record_success(successes, spec, output_path, df)
            print(f"[OK] FDR {item.part}: {item.name}")
        except Exception as exc:
            record_failure(failures, spec, exc)
            print(f"[FAIL] FDR {item.part}: {item.name} - {exc}")
    return frames


def collect_fred(successes: list[dict[str, Any]], failures: list[dict[str, Any]]) -> pd.DataFrame | None:
    spec = {
        "priority": 1,
        "part": "fred",
        "name": "fred_macro_semiconductor_panel",
        "symbol": "FRED:" + ",".join(FRED_SYMBOLS.keys()),
        "source": "FDR/FRED",
        "frequency": "mixed",
        "description": "FRED macro, rate, market, semiconductor production/PPI/new-order indicators",
    }
    try:
        df = fdr.DataReader(spec["symbol"], START, END)
        if df is None or df.empty:
            raise ValueError("empty dataframe")
        df = df.rename(columns=FRED_SYMBOLS)
        df.index.name = "date"
        output_path = DATA_ROOT / "raw" / "fdr" / "fred" / "fred_macro_semiconductor_panel.csv"
        save_dataframe(df, output_path)
        for code, name in FRED_SYMBOLS.items():
            one = df[[name]].dropna()
            if not one.empty:
                save_dataframe(one, DATA_ROOT / "raw" / "fdr" / "fred" / f"{name}.csv")
        record_success(successes, spec, output_path, df)
        print("[OK] FRED panel")
        return df
    except Exception as exc:
        record_failure(failures, spec, exc)
        print(f"[FAIL] FRED panel - {exc}")
        return None


def collect_financials(successes: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    for name, symbol, frequency in FINANCIAL_SPECS:
        spec = {
            "priority": 1,
            "part": "financials",
            "name": name,
            "symbol": symbol,
            "source": "FDR/NAVER",
            "frequency": frequency,
            "description": "Domestic company financial statement snapshot",
        }
        try:
            df = fdr.SnapDataReader(symbol)
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            output_path = DATA_ROOT / "raw" / "fdr" / "financials" / f"{name}.csv"
            save_dataframe(df, output_path)
            record_success(successes, spec, output_path, df)
            print(f"[OK] financials: {name}")
        except Exception as exc:
            record_failure(failures, spec, exc)
            print(f"[FAIL] financials: {name} - {exc}")


def collect_external_pages(successes: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    session = requests.Session()
    for item in EXTERNAL_PUBLIC_SOURCES:
        spec = {
            "priority": item["priority"],
            "part": item["part"],
            "name": item["name"],
            "symbol": item["url"],
            "source": "external_public",
            "frequency": "source_dependent",
            "description": item["description"],
        }
        try:
            headers = SEC_HEADERS if "sec.gov" in item["url"] else {"User-Agent": SEC_HEADERS["User-Agent"]}
            response = session.get(item["url"], headers=headers, timeout=30)
            response.raise_for_status()
            target_dir = DATA_ROOT / "raw" / "external" / item["save_dir"]
            if item["name"] == "nvidia_sec_companyfacts":
                payload = response.json()
                output_path = target_dir / "nvidia_sec_companyfacts.json"
                output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                df = extract_nvidia_companyfacts(payload)
                csv_path = target_dir / "nvidia_sec_key_financials.csv"
                save_dataframe(df, csv_path)
                record_success(successes, spec, csv_path, df)
            else:
                output_path = target_dir / f"{item['name']}.html"
                output_path.write_text(response.text, encoding="utf-8")
                df = pd.DataFrame(
                    [{"url": item["url"], "status_code": response.status_code, "bytes": len(response.text)}]
                )
                record_success(successes, spec, output_path, df)
            print(f"[OK] external priority {item['priority']}: {item['name']}")
        except Exception as exc:
            record_failure(failures, spec, exc)
            print(f"[FAIL] external priority {item['priority']}: {item['name']} - {exc}")


def extract_nvidia_companyfacts(payload: dict[str, Any]) -> pd.DataFrame:
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
    for tag, friendly in tags.items():
        units = facts.get(tag, {}).get("units", {})
        for unit, values in units.items():
            for value in values:
                rows.append(
                    {
                        "metric": friendly,
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


def build_processed_panels(frames: dict[str, pd.DataFrame], fred: pd.DataFrame | None) -> None:
    close_series = []
    for name, df in frames.items():
        if "Close" in df.columns:
            close_series.append(df["Close"].rename(name))
        elif len(df.columns) == 1:
            close_series.append(df.iloc[:, 0].rename(name))
    if close_series:
        daily_panel = pd.concat(close_series, axis=1).sort_index()
        daily_panel.index.name = "date"
        save_dataframe(daily_panel, DATA_ROOT / "processed" / "daily_panel.csv")
        daily_panel.to_parquet(DATA_ROOT / "processed" / "daily_panel.parquet")
        daily_returns = daily_panel.pct_change(fill_method=None)
        save_dataframe(daily_returns, DATA_ROOT / "processed" / "daily_returns.csv")
        daily_returns.to_parquet(DATA_ROOT / "processed" / "daily_returns.parquet")
        monthly_panel = daily_panel.resample("ME").last()
        if fred is not None and not fred.empty:
            monthly_panel = monthly_panel.join(fred.resample("ME").last(), how="outer")
        save_dataframe(monthly_panel, DATA_ROOT / "processed" / "monthly_panel.csv")
        monthly_panel.to_parquet(DATA_ROOT / "processed" / "monthly_panel.parquet")


def write_metadata(successes: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    metadata_dir = DATA_ROOT / "metadata"
    success_df = pd.DataFrame(successes)
    failure_df = pd.DataFrame(failures)
    all_df = pd.concat([success_df, failure_df], ignore_index=True, sort=False)
    if not success_df.empty:
        success_df.to_csv(metadata_dir / "data_dictionary.csv", index=False, encoding="utf-8-sig")
    if not failure_df.empty:
        failure_df.to_csv(metadata_dir / "failed_collections.csv", index=False, encoding="utf-8-sig")
    all_df.to_csv(metadata_dir / "collection_summary.csv", index=False, encoding="utf-8-sig")
    write_source_catalog(metadata_dir)
    write_symbol_map(metadata_dir)


def write_source_catalog(metadata_dir: Path) -> None:
    rows = [item.__dict__ for item in FDR_SPECS]
    rows += [
        {
            "priority": 1,
            "part": "fred",
            "name": name,
            "symbol": f"FRED:{code}",
            "source": "FDR/FRED",
            "frequency": "mixed",
            "save_dir": "fred",
            "description": "FRED macro/semiconductor indicator",
        }
        for code, name in FRED_SYMBOLS.items()
    ]
    rows += [
        {
            "priority": 1,
            "part": "financials",
            "name": name,
            "symbol": symbol,
            "source": "FDR/NAVER",
            "frequency": frequency,
            "save_dir": "financials",
            "description": "Domestic financial statement",
        }
        for name, symbol, frequency in FINANCIAL_SPECS
    ]
    rows += [
        {
            "priority": item["priority"],
            "part": item["part"],
            "name": item["name"],
            "symbol": item["url"],
            "source": "external_public",
            "frequency": "source_dependent",
            "save_dir": item["save_dir"],
            "description": item["description"],
        }
        for item in EXTERNAL_PUBLIC_SOURCES
    ]
    pd.DataFrame(rows).to_csv(metadata_dir / "source_catalog.csv", index=False, encoding="utf-8-sig")


def write_symbol_map(metadata_dir: Path) -> None:
    rows = [
        {"asset": item.name, "symbol": item.symbol, "part": item.part, "source": item.source}
        for item in FDR_SPECS
    ]
    rows += [{"asset": name, "symbol": code, "part": "fred", "source": "FRED"} for code, name in FRED_SYMBOLS.items()]
    pd.DataFrame(rows).to_csv(metadata_dir / "symbol_map.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    frames = collect_fdr_series(successes, failures)
    collect_financials(successes, failures)
    fred = collect_fred(successes, failures)
    collect_external_pages(successes, failures)
    build_processed_panels(frames, fred)
    write_metadata(successes, failures)
    print(f"Done. success={len(successes)}, failed={len(failures)}")
    print(f"Metadata: {DATA_ROOT / 'metadata'}")


if __name__ == "__main__":
    main()
