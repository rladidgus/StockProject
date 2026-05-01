import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)


def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


def get_conn():
    return get_engine().connect()


def init_db():
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                date DATE NOT NULL, ticker VARCHAR(10) NOT NULL,
                open NUMERIC(12,4), high NUMERIC(12,4),
                low NUMERIC(12,4), close NUMERIC(12,4),
                volume BIGINT, return_pct NUMERIC(8,4),
                PRIMARY KEY (date, ticker)
            );
            CREATE TABLE IF NOT EXISTS macro_features (
                date DATE PRIMARY KEY,
                nasdaq NUMERIC(12,4), sox NUMERIC(12,4), vix NUMERIC(8,4),
                us_rate NUMERIC(6,4), usd_krw NUMERIC(10,4),
                nasdaq_lag1 NUMERIC(12,4), nasdaq_lag3 NUMERIC(12,4),
                nasdaq_lag5 NUMERIC(12,4), vix_lag1 NUMERIC(8,4),
                vix_lag3 NUMERIC(8,4), vix_lag5 NUMERIC(8,4),
                us_rate_lag1 NUMERIC(6,4), usd_krw_lag1 NUMERIC(10,4)
            );
            CREATE TABLE IF NOT EXISTS micro_features (
                date DATE NOT NULL, ticker VARCHAR(10) NOT NULL,
                revenue NUMERIC(20,2), operating_profit NUMERIC(20,2),
                eps NUMERIC(12,4),
                PRIMARY KEY (date, ticker)
            );
            CREATE TABLE IF NOT EXISTS alpha_features (
                date DATE NOT NULL, ticker VARCHAR(10) NOT NULL,
                foreign_net_buy NUMERIC(20,2), institution_net_buy NUMERIC(20,2),
                put_call_ratio NUMERIC(8,4), short_interest NUMERIC(8,4),
                PRIMARY KEY (date, ticker)
            );
            CREATE TABLE IF NOT EXISTS shap_results (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL, ticker VARCHAR(10) NOT NULL,
                feature_name VARCHAR(50) NOT NULL,
                shap_value NUMERIC(12,6), feature_value NUMERIC(12,6)
            );
            CREATE INDEX IF NOT EXISTS idx_shap_ticker_date
                ON shap_results (ticker, date);
            CREATE INDEX IF NOT EXISTS idx_stock_ticker
                ON stock_prices (ticker);
        """))
        conn.commit()
    print("PostgreSQL DB 초기화 완료")


if __name__ == "__main__":
    init_db()
