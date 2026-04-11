"""
data_loader.py — Data Collection Layer
Fetches OHLCV stock data from Yahoo Finance via yfinance.
Handles caching so repeated runs don't hit the network unnecessarily.
"""

import os
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from config import DATA_DIR, VALID_PERIODS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for *ticker* covering *period*.

    Parameters
    ----------
    ticker : str
        Stock symbol, e.g. "AAPL", "TSLA", "RELIANCE.NS"
    period : str
        Time window accepted by yfinance: "1y", "2y", "5y", "max", etc.

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume, Ticker
        Index  : DatetimeIndex (sorted ascending, timezone-naive)

    Raises
    ------
    ValueError
        If the ticker is invalid or no data is returned.
    """
    ticker  = ticker.strip().upper()
    period  = period if period in VALID_PERIODS else "5y"

    # Check on-disk cache first
    cached = _load_cache(ticker, period)
    if cached is not None:
        logger.info("Loaded %s from cache (%d rows)", ticker, len(cached))
        return cached

    logger.info("Fetching %s | period=%s from Yahoo Finance…", ticker, period)
    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    except Exception as exc:
        raise ValueError(f"Network error fetching {ticker}: {exc}") from exc

    if raw.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. "
            "Check the symbol (e.g. use RELIANCE.NS for Indian stocks)."
        )

    df = _clean(raw, ticker)
    _save_cache(df, ticker, period)
    logger.info("Fetched %d rows for %s", len(df), ticker)
    return df


def get_ticker_info(ticker: str) -> dict:
    """Return basic company metadata (name, sector, currency)."""
    ticker = ticker.strip().upper()
    try:
        info = yf.Ticker(ticker).info
        return {
            "name"    : info.get("longName", ticker),
            "sector"  : info.get("sector", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "country" : info.get("country", "N/A"),
        }
    except Exception:
        return {"name": ticker, "sector": "N/A", "currency": "USD",
                "exchange": "N/A", "country": "N/A"}


def validate_ticker(ticker: str) -> bool:
    """Return True if *ticker* returns at least some historical data."""
    try:
        df = yf.download(ticker.strip().upper(), period="1mo",
                         progress=False, auto_adjust=True)
        return not df.empty
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Standardise raw yfinance output:
    - Flatten multi-level columns
    - Drop rows where Close is NaN
    - Ensure DatetimeIndex is timezone-naive and sorted
    - Add Ticker column for multi-stock scenarios
    """
    # yfinance sometimes returns MultiIndex columns when multiple tickers used
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(subset=["Close"], inplace=True)

    # Remove timezone so downstream libs (sklearn, keras) don't complain
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.sort_index(inplace=True)
    df["Ticker"] = ticker
    return df


def _cache_path(ticker: str, period: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{ticker}_{period}.parquet")


def _load_cache(ticker: str, period: str) -> pd.DataFrame | None:
    """Return cached DataFrame if it exists and is fresh (< 1 day old)."""
    path = _cache_path(ticker, period)
    if not os.path.exists(path):
        return None
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    if age > timedelta(hours=23):
        return None                      # stale — re-fetch
    return pd.read_parquet(path)


def _save_cache(df: pd.DataFrame, ticker: str, period: str) -> None:
    path = _cache_path(ticker, period)
    df.to_parquet(path)
    logger.info("Cached to %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test (run: python data_loader.py)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for sym in ["AAPL", "TSLA", "INVALID_XYZ"]:
        try:
            data = fetch_stock_data(sym, period="1y")
            info = get_ticker_info(sym)
            print(f"\n{sym} — {info['name']}")
            print(data.tail(3))
        except ValueError as e:
            print(f"\n{sym} — ERROR: {e}")
