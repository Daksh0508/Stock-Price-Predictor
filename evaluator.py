"""
evaluator.py — Model Evaluation & Prediction
Handles:
  - RMSE, MAE, R² calculation
  - Inverse-transforming scaled predictions back to real prices
  - Generating multi-day forecasts
  - Buy/Sell signal generation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from config import (
    SEQUENCE_LENGTH, PREDICTION_DAYS,
    TARGET_COLUMN, BUY_SIGNAL_THRESHOLD, SELL_SIGNAL_THRESHOLD
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE and R² given actual and predicted price arrays.
    Both arrays must be in the ORIGINAL price scale (after inverse transform).
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

    results = {"RMSE": round(rmse, 4),
               "MAE" : round(mae,  4),
               "R2"  : round(r2,   4),
               "MAPE": round(mape, 4)}
    logger.info("Metrics: %s", results)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 2. Inverse Transform
# ──────────────────────────────────────────────────────────────────────────────

def inverse_transform_predictions(
    y_scaled: np.ndarray,
    target_scaler,
) -> np.ndarray:
    """
    Reverse MinMax scaling on a 1-D prediction array.
    target_scaler was fitted on the Close column only (shape n×1).
    """
    return target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


# ──────────────────────────────────────────────────────────────────────────────
# 3. LSTM Prediction on Test Set
# ──────────────────────────────────────────────────────────────────────────────

def predict_test_set(model, X_test: np.ndarray, target_scaler) -> np.ndarray:
    """Run model on test sequences and return prices in original scale."""
    y_scaled = model.predict(X_test, verbose=0).flatten()
    return inverse_transform_predictions(y_scaled, target_scaler)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Multi-Day Future Forecast
# ──────────────────────────────────────────────────────────────────────────────

def forecast_future(
    model,
    last_sequence: np.ndarray,   # shape (seq_len, n_features)
    feature_scaler,
    target_scaler,
    feature_cols: list,
    n_days: int = PREDICTION_DAYS,
) -> np.ndarray:
    """
    Iteratively predict *n_days* into the future using the sliding-window method.

    How it works
    ------------
    1. Start with the last known window of *seq_len* days.
    2. Predict day seq_len+1.
    3. Append that prediction into the window (shift by 1).
    4. Repeat n_days times.

    Only the Close column is predicted; other features are held at their
    last known values (a practical approximation for short horizons).
    """
    close_col_idx = feature_cols.index(TARGET_COLUMN)
    current_seq   = last_sequence.copy()    # (seq_len, n_features)
    predictions   = []

    for _ in range(n_days):
        # Model expects shape (1, seq_len, n_features)
        x_input   = current_seq[np.newaxis, ...]
        pred_scaled = model.predict(x_input, verbose=0)[0, 0]
        predictions.append(pred_scaled)

        # Build next row: copy last row, update only the Close column
        next_row  = current_seq[-1].copy()
        next_row[close_col_idx] = pred_scaled

        # Slide window forward by one day
        current_seq = np.vstack([current_seq[1:], next_row])

    # Convert scaled predictions back to real prices
    return inverse_transform_predictions(np.array(predictions), target_scaler)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Buy / Sell Signal Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_signals(actual_prices: np.ndarray, predicted_prices: np.ndarray) -> pd.Series:
    """
    Simple signal rule based on predicted return:
      predicted_return = (predicted[t] - actual[t-1]) / actual[t-1]
      > +2%  → BUY
      < -2%  → SELL
      else   → HOLD

    Returns a Pandas Series of {"BUY", "SELL", "HOLD"}.
    """
    signals = []
    for i in range(len(predicted_prices)):
        base  = actual_prices[i - 1] if i > 0 else actual_prices[0]
        ret   = (predicted_prices[i] - base) / base if base != 0 else 0
        if   ret >  BUY_SIGNAL_THRESHOLD:  signals.append("BUY")
        elif ret <  SELL_SIGNAL_THRESHOLD: signals.append("SELL")
        else:                               signals.append("HOLD")
    return pd.Series(signals)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Full Evaluation Report
# ──────────────────────────────────────────────────────────────────────────────

def evaluation_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler,
    test_df: pd.DataFrame,
    feature_scaler,
    feature_cols: list,
    ticker: str,
) -> dict:
    """
    Compute all evaluation artefacts in one call.

    Returns dict with:
        metrics          RMSE / MAE / R² / MAPE
        actual_prices    array of true Close prices
        predicted_prices array of predicted Close prices
        test_dates       DatetimeIndex aligned with predictions
        signals          Buy/Sell/Hold per prediction step
        forecast_prices  n_days future forecast
        forecast_dates   pandas DatetimeIndex for forecast
    """
    # Actual prices from test set
    actual_prices = test_df[TARGET_COLUMN].values[SEQUENCE_LENGTH:]

    # LSTM predictions
    predicted_prices = predict_test_set(model, X_test, target_scaler)

    # Align dates (sequences start SEQUENCE_LENGTH rows into test_df)
    test_dates = test_df.index[SEQUENCE_LENGTH:]

    metrics = compute_metrics(actual_prices, predicted_prices)
    signals = generate_signals(actual_prices, predicted_prices)

    # Future forecast starting from the last test window
    close_col_idx = feature_cols.index(TARGET_COLUMN)
    last_seq  = feature_scaler.transform(test_df[feature_cols])[-SEQUENCE_LENGTH:]
    forecast  = forecast_future(model, last_seq, feature_scaler, target_scaler,
                                feature_cols, n_days=PREDICTION_DAYS)

    # Create business-day dates for the forecast
    last_date      = test_df.index[-1]
    forecast_dates = pd.bdate_range(start=last_date, periods=PREDICTION_DAYS + 1)[1:]

    logger.info("Evaluation complete for %s — %s", ticker, metrics)

    return {
        "metrics"          : metrics,
        "actual_prices"    : actual_prices,
        "predicted_prices" : predicted_prices,
        "test_dates"       : test_dates,
        "signals"          : signals,
        "forecast_prices"  : forecast,
        "forecast_dates"   : forecast_dates,
        "ticker"           : ticker,
    }
