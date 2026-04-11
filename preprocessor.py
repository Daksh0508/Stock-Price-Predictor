"""
preprocessor.py — Data Preprocessing Pipeline
Handles:
  1. Missing value imputation
  2. Technical indicator feature engineering (MA, EMA, RSI, MACD)
  3. MinMax normalisation (fitted only on training data to prevent data leakage)
  4. Time-series–aware train/test split
  5. Sliding-window sequence creation for LSTM/GRU
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import logging

from config import (
    MA_WINDOWS, EMA_WINDOWS, RSI_WINDOW,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    SEQUENCE_LENGTH, TEST_SPLIT,
    TARGET_COLUMN, MODEL_DIR
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Feature Engineering
# ──────────────────────────────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append technical analysis columns to *df* in-place and return it.

    Added features
    --------------
    MA_7, MA_21, MA_50        Simple moving averages
    EMA_12, EMA_26            Exponential moving averages
    RSI_14                    Relative Strength Index
    MACD, MACD_Signal         MACD line and signal line
    MACD_Hist                 MACD histogram (MACD - Signal)
    Daily_Return              Percentage daily change
    Volatility_21             21-day rolling standard deviation of returns
    """
    df = df.copy()
    close = df[TARGET_COLUMN]

    # ── Simple moving averages ────────────────────────────────────────────────
    for w in MA_WINDOWS:
        df[f"MA_{w}"] = close.rolling(window=w).mean()

    # ── Exponential moving averages ───────────────────────────────────────────
    for w in EMA_WINDOWS:
        df[f"EMA_{w}"] = close.ewm(span=w, adjust=False).mean()

    # ── RSI ───────────────────────────────────────────────────────────────────
    # Classic Wilder RSI: compare average gain vs average loss over 14 periods
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(alpha=1 / RSI_WINDOW, min_periods=RSI_WINDOW).mean()
    avg_l  = loss.ewm(alpha=1 / RSI_WINDOW, min_periods=RSI_WINDOW).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    df[f"RSI_{RSI_WINDOW}"] = 100 - (100 / (1 + rs))

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema_fast   = close.ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow   = close.ewm(span=MACD_SLOW,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["MACD"]        = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"]   = macd_line - signal_line

    # ── Daily return & volatility ─────────────────────────────────────────────
    df["Daily_Return"]   = close.pct_change()
    df["Volatility_21"]  = df["Daily_Return"].rolling(21).std()

    # ── Drop rows where indicators are still NaN (warm-up period) ────────────
    df.dropna(inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Train/Test Split (time-series aware — NO shuffling)
# ──────────────────────────────────────────────────────────────────────────────

def time_series_split(df: pd.DataFrame, test_size: float = TEST_SPLIT):
    """
    Split chronologically.  The last *test_size* fraction becomes the test set.

    Why not random split?
    ---------------------
    Random splits cause data leakage in time-series: the model would see
    future data during training, giving unrealistically high accuracy scores.
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]
    logger.info(
        "Split: train=%d rows (%s → %s), test=%d rows (%s → %s)",
        len(train_df), train_df.index[0].date(), train_df.index[-1].date(),
        len(test_df),  test_df.index[0].date(),  test_df.index[-1].date(),
    )
    return train_df, test_df


# ──────────────────────────────────────────────────────────────────────────────
# 3. Normalisation (fit on train only)
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA_7", "MA_21", "MA_50",
    "EMA_12", "EMA_26",
    "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist",
    "Daily_Return", "Volatility_21",
]

def build_scalers(train_df: pd.DataFrame):
    """
    Fit two independent MinMaxScalers:
      - feature_scaler : scales all FEATURE_COLUMNS to [0, 1]
      - target_scaler  : scales only the Close price (needed to invert predictions)

    Returns (feature_scaler, target_scaler, feature_columns_used)
    """
    # Keep only columns that actually exist in the dataframe
    available_features = [c for c in FEATURE_COLUMNS if c in train_df.columns]

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(train_df[available_features])

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(train_df[[TARGET_COLUMN]])

    return feature_scaler, target_scaler, available_features


def scale_data(df: pd.DataFrame, feature_scaler, feature_cols: list) -> np.ndarray:
    """Transform *df* using a pre-fitted *feature_scaler*."""
    return feature_scaler.transform(df[feature_cols])


def save_scalers(feature_scaler, target_scaler, ticker: str) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(feature_scaler, os.path.join(MODEL_DIR, f"{ticker}_feature_scaler.pkl"))
    joblib.dump(target_scaler,  os.path.join(MODEL_DIR, f"{ticker}_target_scaler.pkl"))
    logger.info("Scalers saved for %s", ticker)


def load_scalers(ticker: str):
    fs = joblib.load(os.path.join(MODEL_DIR, f"{ticker}_feature_scaler.pkl"))
    ts = joblib.load(os.path.join(MODEL_DIR, f"{ticker}_target_scaler.pkl"))
    return fs, ts


# ──────────────────────────────────────────────────────────────────────────────
# 4. Sliding-Window Sequence Creation (for LSTM / GRU)
# ──────────────────────────────────────────────────────────────────────────────

def create_sequences(scaled_data: np.ndarray,
                     close_col_idx: int,
                     seq_len: int = SEQUENCE_LENGTH):
    """
    Convert a 2D scaled array into (X, y) pairs using the sliding-window technique.

    How it works
    ------------
    For each position i from seq_len to len(data)-1:
        X[i] = scaled_data[i-seq_len : i]   ← shape (seq_len, n_features)
        y[i] = scaled_data[i, close_col_idx] ← the next Close price (scaled)

    Example with seq_len=3, data=[1,2,3,4,5]:
        X = [[1,2,3], [2,3,4]]
        y = [4, 5]
    """
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len : i])          # past N days
        y.append(scaled_data[i, close_col_idx])          # next day's Close
    return np.array(X), np.array(y)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Full Pipeline (convenience wrapper)
# ──────────────────────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, ticker: str):
    """
    Full preprocessing pipeline:
    1. Add technical indicators
    2. Chronological train/test split
    3. Fit scalers on training data
    4. Create LSTM sequences
    5. Persist scalers to disk

    Returns
    -------
    dict with keys:
        train_df, test_df           raw DataFrames (for plotting)
        X_train, y_train            LSTM training sequences
        X_test,  y_test             LSTM test sequences
        feature_scaler              fitted MinMaxScaler (features)
        target_scaler               fitted MinMaxScaler (Close only)
        feature_cols                list of feature column names
        close_col_idx               index of "Close" in feature_cols
    """
    # Step 1 — indicators
    df_feat = add_technical_indicators(df)

    # Step 2 — split
    train_df, test_df = time_series_split(df_feat)

    # Step 3 — scalers (fit on train only)
    feature_scaler, target_scaler, feature_cols = build_scalers(train_df)
    save_scalers(feature_scaler, target_scaler, ticker)

    close_col_idx = feature_cols.index(TARGET_COLUMN)

    # Step 4 — scale
    train_scaled = scale_data(train_df, feature_scaler, feature_cols)
    test_scaled  = scale_data(test_df,  feature_scaler, feature_cols)

    # Step 5 — sequences
    X_train, y_train = create_sequences(train_scaled, close_col_idx)
    X_test,  y_test  = create_sequences(test_scaled,  close_col_idx)

    logger.info(
        "Sequences — train: %s/%s  test: %s/%s",
        X_train.shape, y_train.shape, X_test.shape, y_test.shape
    )

    return {
        "train_df"       : train_df,
        "test_df"        : test_df,
        "X_train"        : X_train,
        "y_train"        : y_train,
        "X_test"         : X_test,
        "y_test"         : y_test,
        "feature_scaler" : feature_scaler,
        "target_scaler"  : target_scaler,
        "feature_cols"   : feature_cols,
        "close_col_idx"  : close_col_idx,
    }
