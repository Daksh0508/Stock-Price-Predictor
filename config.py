"""
config.py — Central configuration
All tuneable parameters live here so nothing is hardcoded.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
DEFAULT_TICKER   = "AAPL"
DEFAULT_PERIOD   = "5y"          # yfinance period string: 1y, 2y, 5y, max
VALID_PERIODS    = ["6mo", "1y", "2y", "5y", "10y", "max"]

# ── Feature engineering ───────────────────────────────────────────────────────
MA_WINDOWS       = [7, 21, 50]   # Simple moving average windows (days)
EMA_WINDOWS      = [12, 26]      # Exponential moving average windows
RSI_WINDOW       = 14            # RSI look-back period
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9

# ── Model ─────────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 60            # Past N days used as input (sliding window)
PREDICTION_DAYS  = 7             # Future days to forecast
TEST_SPLIT       = 0.20          # 20% of data held out for testing (time-based)
TARGET_COLUMN    = "Close"

# LSTM hyper-parameters
LSTM_UNITS       = [128, 64]     # Units per LSTM layer
DROPOUT_RATE     = 0.2
EPOCHS           = 50
BATCH_SIZE       = 32
LEARNING_RATE    = 0.001
EARLY_STOP_PATIENCE = 10

# ── Paths ──────────────────────────────────────────────────────────────────────
import os
BASE_DIR     = os.path.dirname(__file__)
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
EXPORTS_DIR  = os.path.join(BASE_DIR, "exports")

# ── Evaluation ────────────────────────────────────────────────────────────────
METRICS = ["RMSE", "MAE", "R2"]

# ── Signals ───────────────────────────────────────────────────────────────────
BUY_SIGNAL_THRESHOLD  =  0.02   # +2% predicted rise → BUY
SELL_SIGNAL_THRESHOLD = -0.02   # -2% predicted fall → SELL
