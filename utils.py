"""
utils.py — Shared Utilities
Plotly chart builders + CSV export + sentiment helper.
All charts return plotly.graph_objects.Figure so the dashboard can display them.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging

from config import EXPORTS_DIR, TARGET_COLUMN

logger = logging.getLogger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────
GREEN  = "#26A69A"
RED    = "#EF5350"
BLUE   = "#1976D2"
ORANGE = "#FB8C00"
PURPLE = "#7B1FA2"
GRAY   = "#9E9E9E"
BG     = "#FAFAFA"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Candlestick Chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Interactive OHLC candlestick chart with volume subplot.
    Includes MA_21 and MA_50 overlaid if present.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{ticker} — Candlestick", "Volume"),
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        name="OHLC",
    ), row=1, col=1)

    # Moving averages
    for col, colour, label in [
        ("MA_21", BLUE,   "MA 21"),
        ("MA_50", ORANGE, "MA 50"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], line=dict(color=colour, width=1.2),
                name=label, opacity=0.8,
            ), row=1, col=1)

    # Volume bars
    colours = [GREEN if c >= o else RED
               for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colours, opacity=0.6, name="Volume",
    ), row=2, col=1)

    fig.update_layout(
        height=520, xaxis_rangeslider_visible=False,
        paper_bgcolor=BG, plot_bgcolor=BG,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2. Actual vs Predicted
# ──────────────────────────────────────────────────────────────────────────────

def plot_prediction(
    test_dates,
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    forecast_dates=None,
    forecast_prices: np.ndarray = None,
    ticker: str = "",
    signals: pd.Series = None,
) -> go.Figure:
    """
    Chart showing:
      - Full historical Close price (light gray)
      - Test-set actual prices (blue)
      - Model predictions (orange)
      - 7-day future forecast (green dashed)
      - Buy/Sell markers (optional)
    """
    fig = go.Figure()

    # Actual prices in test window
    fig.add_trace(go.Scatter(
        x=test_dates, y=actual_prices,
        line=dict(color=BLUE, width=2),
        name="Actual price",
    ))

    # Model predictions
    fig.add_trace(go.Scatter(
        x=test_dates, y=predicted_prices,
        line=dict(color=ORANGE, width=2, dash="dot"),
        name="Predicted price",
    ))

    # Future forecast
    if forecast_dates is not None and forecast_prices is not None:
        # Connecting line from last actual to first forecast
        fig.add_trace(go.Scatter(
            x=[test_dates[-1], forecast_dates[0]],
            y=[actual_prices[-1], forecast_prices[0]],
            line=dict(color=GREEN, width=1.5, dash="dot"),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_prices,
            line=dict(color=GREEN, width=2.5, dash="dash"),
            name="7-day forecast",
            fill="tozeroy", fillcolor="rgba(38,166,154,0.07)",
        ))

    # Buy/Sell markers
    if signals is not None:
        _add_signals(fig, test_dates, actual_prices, signals)

    fig.update_layout(
        title=f"{ticker} — Actual vs Predicted Close Price",
        xaxis_title="Date", yaxis_title="Price (USD)",
        height=480, paper_bgcolor=BG, plot_bgcolor=BG,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _add_signals(fig, dates, prices, signals: pd.Series):
    buy_x  = [d for d, s in zip(dates, signals) if s == "BUY"]
    buy_y  = [p for p, s in zip(prices, signals) if s == "BUY"]
    sell_x = [d for d, s in zip(dates, signals) if s == "SELL"]
    sell_y = [p for p, s in zip(prices, signals) if s == "SELL"]

    if buy_x:
        fig.add_trace(go.Scatter(
            x=buy_x, y=buy_y, mode="markers",
            marker=dict(symbol="triangle-up", color=GREEN, size=10),
            name="Buy signal",
        ))
    if sell_x:
        fig.add_trace(go.Scatter(
            x=sell_x, y=sell_y, mode="markers",
            marker=dict(symbol="triangle-down", color=RED, size=10),
            name="Sell signal",
        ))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Technical Indicators Panel
# ──────────────────────────────────────────────────────────────────────────────

def plot_technicals(df: pd.DataFrame, ticker: str) -> go.Figure:
    """3-row chart: Close + MAs, RSI, MACD histogram."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=("Price & Moving Averages", "RSI (14)", "MACD"),
    )

    # Price
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"],
                             line=dict(color=BLUE, width=1.5), name="Close"), row=1, col=1)
    for col, c, lbl in [("MA_7", ORANGE, "MA 7"), ("MA_21", PURPLE, "MA 21"),
                        ("MA_50", GRAY,   "MA 50")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col],
                                     line=dict(color=c, width=1), name=lbl), row=1, col=1)

    # RSI
    rsi_col = "RSI_14"
    if rsi_col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[rsi_col],
                                 line=dict(color=PURPLE, width=1.2), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=RED,   row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=GREEN, row=2, col=1)

    # MACD
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
                                 line=dict(color=BLUE,  width=1.2), name="MACD"),   row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"],
                                 line=dict(color=ORANGE, width=1.2), name="Signal"), row=3, col=1)
        hist_col = [GREEN if v >= 0 else RED for v in df.get("MACD_Hist", [])]
        if "MACD_Hist" in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"],
                                 marker_color=hist_col, name="Histogram",
                                 opacity=0.6), row=3, col=1)

    fig.update_layout(
        height=580, paper_bgcolor=BG, plot_bgcolor=BG,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. Training Loss Curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_history(history) -> go.Figure:
    """Plot train and validation loss over epochs."""
    epochs = list(range(1, len(history.history["loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history.history["loss"],
                             name="Training loss",   line=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=epochs, y=history.history["val_loss"],
                             name="Validation loss", line=dict(color=ORANGE, dash="dot")))
    fig.update_layout(
        title="LSTM Training Loss", xaxis_title="Epoch", yaxis_title="MSE Loss",
        height=320, paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5. CSV Export
# ──────────────────────────────────────────────────────────────────────────────

def export_predictions_csv(
    test_dates,
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    signals: pd.Series,
    forecast_dates=None,
    forecast_prices: np.ndarray = None,
    ticker: str = "STOCK",
) -> str:
    """
    Write predictions to a CSV file and return the file path.
    Suitable for download via Streamlit's st.download_button.
    """
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    test_df = pd.DataFrame({
        "Date"           : test_dates,
        "Actual_Price"   : actual_prices,
        "Predicted_Price": predicted_prices,
        "Signal"         : signals.values if signals is not None else "N/A",
        "Type"           : "test",
    })

    if forecast_dates is not None and forecast_prices is not None:
        fcast_df = pd.DataFrame({
            "Date"           : forecast_dates,
            "Actual_Price"   : np.nan,
            "Predicted_Price": forecast_prices,
            "Signal"         : "FORECAST",
            "Type"           : "forecast",
        })
        combined = pd.concat([test_df, fcast_df], ignore_index=True)
    else:
        combined = test_df

    path = os.path.join(EXPORTS_DIR, f"{ticker}_predictions.csv")
    combined.to_csv(path, index=False)
    logger.info("Exported predictions to %s", path)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 6. Sentiment Analysis (Bonus Feature)
# ──────────────────────────────────────────────────────────────────────────────

def get_sentiment_score(ticker: str) -> dict:
    """
    Stub for sentiment analysis.  Replace API_KEY with a real NewsAPI key.
    Falls back gracefully so the dashboard doesn't crash without a key.
    """
    try:
        from newsapi import NewsApiClient
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        api_key = os.getenv("NEWS_API_KEY", "")
        if not api_key:
            return {"score": 0, "label": "N/A", "articles": []}

        client    = NewsApiClient(api_key=api_key)
        response  = client.get_everything(q=ticker, language="en",
                                          sort_by="relevancy", page_size=10)
        analyser  = SentimentIntensityAnalyzer()
        scores    = []
        articles  = []

        for a in response.get("articles", []):
            text   = (a.get("title") or "") + " " + (a.get("description") or "")
            score  = analyser.polarity_scores(text)["compound"]
            scores.append(score)
            articles.append({
                "title"    : a.get("title", ""),
                "score"    : round(score, 3),
                "url"      : a.get("url", ""),
                "published": a.get("publishedAt", ""),
            })

        avg = float(np.mean(scores)) if scores else 0
        label = "Positive" if avg > 0.05 else "Negative" if avg < -0.05 else "Neutral"
        return {"score": round(avg, 3), "label": label, "articles": articles}

    except ImportError:
        return {"score": 0, "label": "N/A (install newsapi-python & vaderSentiment)",
                "articles": []}
    except Exception as exc:
        logger.warning("Sentiment fetch failed: %s", exc)
        return {"score": 0, "label": "Unavailable", "articles": []}
