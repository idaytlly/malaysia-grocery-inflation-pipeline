"""
extract_and_load.py
===================
Phase 1 of the Malaysian Grocery Inflation Pipeline.

Flow: PriceCatcher (data.gov.my Parquet files)
        -> download (requests)
        -> clean / filter (pandas)
        -> load into a star schema (DuckDB)

Usage:
    python src/extract_and_load.py                         #last month, Johor
    python src/extract_and_load.py --month 2024-06 --state Selangor
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_URL = "https://storage.data.gov.my/pricecatcher"
PRICES_URL = BASE_URL + "/pricecatcher_{month}.parquet"     # one file per month
ITEMS_URL = BASE_URL +"/lookup_item.parquet"
PREMISES_URL = BASE_URL + "/lookup_premise.parquet"

PROJECT_ROOT =Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "warehouse" / "grocery_prices.duckdb"
DEFAULT_STATE = "Johor"

log = logging.getLogger("extract_and_load")

class PipelineError(Exception):
    """Raised for any expected, handled failure in the pipeline."""

# --------------------------------------------------------------------------- #
# Star schema DDL (kept as a list so each statement runs on its own)
# --------------------------------------------------------------------------- #
SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS dim_date (
        date_id      INTEGER PRIMARY KEY,      -- DDMMYYYY
        full_date    DATE NOT NULL,
        year         SMALLINT NOT NULL,
        quarter      SMALLINT NOT NULL,
        month        SMALLINT NOT NULL,
        month_name   VARCHAR NOT NULL,
        day          SMALLINT NOT NULL,
        day_of_week  SMALLINT NOT NULL,
        day_name     VARCHAR NOT NULL,
        is_weekend   BOOLEAN NOT NULL
    )""",
    """
    CREATE TABLE IF NOT EXISTS dim_items (
        item_code    INTEGER PRIMARY KEY,
        item_name    VARCHAR NOT NULL,
        unit         VARCHAR,
        item_group     VARCHAR,
        item_category  VARCHAR
    )""",
    """
    CREATE TABLE IF NOT EXISTS dim_locations (
        premise_code  INTEGER PRIMARY KEY,
        premise_name  VARCHAR,
        address       VARCHAR,
        premise_type  VARCHAR,
        state         VARCHAR NOT NULL,
        district      VARCHAR
    )
    """,
    # No FOREIGN KEY constraints on purpose: they slow bulk loads in DuckDB.
    # Referential integrity is checked in Phase 2 (orphan-key checks).
    """
    CREATE TABLE IF NOT EXISTS fact_daily_prices (
        date_id       INTEGER       NOT NULL,   -- -> dim_date
        item_code     INTEGER       NOT NULL,   -- -> dim_items
        premise_code  INTEGER       NOT NULL,   -- -> dim_locations
        price         DECIMAL(10,2) NOT NULL,
        PRIMARY KEY (date_id, item_code, premise_code)
    )
    """,
]

# --------------------------------------------------------------------------- #
# Extract
# --------------------------------------------------------------------------- #
def build_session() -> requests.Session:
    """HTTP session that retries transient failures with exponential backoff."""
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_parquet(session: requests.Session, url: str, dest: Path) -> pd.DataFrame:
    """Stream a Parquet file to disk (raw layer), then read it into pandas."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    try:
        with session.get(url, stream=True, timeout=(10, 120)) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        return pd.read_parquet(dest)
    except requests.RequestException as exc:
        raise PipelineError(f"Download failed for {url}: {exc}") from exc
    except (OSError, ValueError) as exc:  # corrupt / unreadable Parquet
        raise PipelineError(f"Could not read {dest.name}: {exc}") from exc

def extract(month: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch prices for one month plus the item and premise lookup tables."""
    session = build_session()
    prices = download_parquet(session, PRICES_URL.format(month=month),
                              RAW_DIR / f"pricecatcher_{month}.parquet")
    items = download_parquet(session, ITEMS_URL, RAW_DIR / "lookup_item.parquet")
    premises = download_parquet(session, PREMISES_URL, RAW_DIR / "lookup_premise.parquet")
    log.info("Extracted %d price rows, %d items, %d premises",
             len(prices), len(items), len(premises))
    return prices, items, premises

# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise PipelineError(
            f"{name} is missing expected columns {sorted(missing)}; "
            f"found {list(df.columns)}. The source schema may have changed."
        )


def clean_locations(premises: pd.DataFrame, state: str) -> pd.DataFrame:
    """Keep only premises in the target state; tidy text columns."""
    _require_columns(premises, {"premise_code", "premise", "address",
                                "premise_type", "state", "district"}, "lookup_premise")
    df = premises.copy()
    for col in ("premise", "address", "premise_type", "state", "district"):
        df[col] = df[col].astype("string").str.strip()

    df = df.dropna(subset=["premise_code", "state"])
    df = df.drop_duplicates(subset="premise_code", keep="last")

    in_state = df[df["state"].str.casefold() == state.casefold()]
    if in_state.empty:
        available = sorted(df["state"].dropna().unique())
        raise PipelineError(f"No premises found for state '{state}'. Available: {available}")

    in_state = in_state.rename(columns={"premise": "premise_name"})
    in_state["premise_code"] = in_state["premise_code"].astype("int64")
    log.info("Premises in %s: %d", state, len(in_state))
    return in_state[["premise_code", "premise_name", "address",
                     "premise_type", "state", "district"]]


def clean_items(items: pd.DataFrame) -> pd.DataFrame:
    """Standardise the item lookup."""
    _require_columns(items, {"item_code", "item", "unit",
                             "item_group", "item_category"}, "lookup_item")
    df = items.copy()
    for col in ("item", "unit", "item_group", "item_category"):
        df[col] = df[col].astype("string").str.strip()
    df = df.dropna(subset=["item_code", "item"])
    df = df.drop_duplicates(subset="item_code", keep="last")
    df["item_code"] = df["item_code"].astype("int64")
    return df.rename(columns={"item": "item_name"})[
        ["item_code", "item_name", "unit", "item_group", "item_category"]
    ]


def clean_prices(prices: pd.DataFrame, premise_codes: set[int],
                 item_codes: set[int]) -> pd.DataFrame:
    """Type-cast, drop invalid rows, and restrict to the target state's premises."""
    _require_columns(prices, {"date", "premise_code", "item_code", "price"}, "prices")
    df = prices[["date", "premise_code", "item_code", "price"]].copy()
    start = len(df)

    # Coerce bad values to NaN instead of crashing, then drop them
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna()
    df = df[df["price"] > 0]                      # zero/negative prices are invalid
    log.info("Dropped %d null/invalid rows", start - len(df))

    # Isolate state-level data (prices have no state column; join via premise)
    df = df[df["premise_code"].isin(premise_codes)]
    # Keep referential integrity with dim_items
    df = df[df["item_code"].isin(item_codes)]

    # One price per (day, item, premise); keep the last reported
    before = len(df)
    df = df.drop_duplicates(subset=["date", "item_code", "premise_code"], keep="last")
    log.info("Dropped %d duplicate rows", before - len(df))

    if df.empty:
        raise PipelineError("No price rows left after cleaning. Check the month/state.")

    df["date_id"] = df["date"].dt.strftime("%Y%m%d").astype("int32")
    df["premise_code"] = df["premise_code"].astype("int64")
    df["item_code"] = df["item_code"].astype("int64")
    df["price"] = df["price"].round(2)
    log.info("Clean fact rows: %d (%.1f%% of raw)", len(df), 100 * len(df) / start)
    return df[["date_id", "item_code", "premise_code", "price"]]


def build_dim_date(fact: pd.DataFrame) -> pd.DataFrame:
    """Derive the date dimension from the dates present in the fact data."""
    dates = pd.to_datetime(fact["date_id"].unique().astype(str), format="%Y%m%d")
    d = pd.DataFrame({"full_date": dates.sort_values()})
    d["date_id"] = d["full_date"].dt.strftime("%Y%m%d").astype("int32")
    d["year"] = d["full_date"].dt.year
    d["quarter"] = d["full_date"].dt.quarter
    d["month"] = d["full_date"].dt.month
    d["month_name"] = d["full_date"].dt.month_name()
    d["day"] = d["full_date"].dt.day
    d["day_of_week"] = d["full_date"].dt.dayofweek
    d["day_name"] = d["full_date"].dt.day_name()
    d["is_weekend"] = d["day_of_week"] >= 5
    d["full_date"] = d["full_date"].dt.date
    return d[["date_id", "full_date", "year", "quarter", "month", "month_name",
              "day", "day_of_week", "day_name", "is_weekend"]]


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load(db_path: Path, dim_date: pd.DataFrame, dim_items: pd.DataFrame,
         dim_locations: pd.DataFrame, fact: pd.DataFrame) -> None:
    """Create the schema if needed and load all tables in one transaction.

    Idempotent: dimensions are upserted (INSERT OR REPLACE); facts use
    INSERT OR IGNORE, so re-running the same month won't create duplicates.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        for stmt in SCHEMA_STATEMENTS:
            con.execute(stmt)

        # DuckDB can query pandas DataFrames directly once registered
        con.register("stg_date", dim_date)
        con.register("stg_items", dim_items)
        con.register("stg_locations", dim_locations)
        con.register("stg_fact", fact)

        con.execute("BEGIN TRANSACTION")
        con.execute("INSERT OR REPLACE INTO dim_date SELECT * FROM stg_date")
        con.execute("INSERT OR REPLACE INTO dim_items SELECT * FROM stg_items")
        con.execute("INSERT OR REPLACE INTO dim_locations SELECT * FROM stg_locations")
        con.execute("""
            INSERT OR IGNORE INTO fact_daily_prices
            SELECT date_id, item_code, premise_code, CAST(price AS DECIMAL(10,2))
            FROM stg_fact
        """)
        con.execute("COMMIT")

        for table in ("dim_date", "dim_items", "dim_locations", "fact_daily_prices"):
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            log.info("  %-18s %10d rows", table, n)
    except duckdb.Error as exc:
        try:
            con.execute("ROLLBACK")
        except duckdb.Error:
            pass
        raise PipelineError(f"DuckDB load failed: {exc}") from exc
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def previous_month() -> str:
    """Default target: the last fully completed month, as YYYY-MM."""
    last_day_prev = date.today().replace(day=1) - timedelta(days=1)
    return f"{last_day_prev:%Y-%m}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PriceCatcher extract & load pipeline")
    p.add_argument("--month", default=previous_month(), help="YYYY-MM (default: last month)")
    p.add_argument("--state", default=DEFAULT_STATE, help=f"State name (default: {DEFAULT_STATE})")
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="DuckDB file path")
    args = p.parse_args()
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", args.month):
        p.error("--month must look like 2024-06")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    args = parse_args()
    log.info("Starting pipeline: month=%s state=%s", args.month, args.state)
    try:
        prices_raw, items_raw, premises_raw = extract(args.month)

        dim_locations = clean_locations(premises_raw, args.state)
        dim_items = clean_items(items_raw)
        fact = clean_prices(prices_raw,
                            set(dim_locations["premise_code"]),
                            set(dim_items["item_code"]))
        dim_date = build_dim_date(fact)

        load(args.db, dim_date, dim_items, dim_locations, fact)
    except PipelineError as exc:
        log.error("Pipeline failed: %s", exc)
        return 1
    except Exception:
        log.exception("Unexpected error")
        return 2
    log.info("Pipeline finished successfully -> %s", args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())