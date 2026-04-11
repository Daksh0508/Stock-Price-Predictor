import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ────────────────────────────────────────────────────────────────
from data_loader  import fetch_stock_data, get_ticker_info, validate_ticker
from preprocessor import prepare_data, add_technical_indicators, FEATURE_COLUMNS
from models       import build_lstm, build_gru, train_lstm, BaselineModels
from models       import save_deep_model, load_deep_model
from evaluator    import evaluation_report, compute_metrics, inverse_transform_predictions
from utils        import (
    plot_candlestick, plot_prediction, plot_technicals,
    plot_training_history, export_predictions_csv, get_sentiment_score
)
from config import (
    DEFAULT_TICKER, VALID_PERIODS, SEQUENCE_LENGTH, MODEL_DIR
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(ticker, period):
    return fetch_stock_data(ticker, period)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_info(ticker):
    return get_ticker_info(ticker)


def model_exists(ticker, model_type="lstm"):
    return os.path.exists(os.path.join(MODEL_DIR, f"{ticker}_{model_type}.keras"))


def scaler_exists(ticker):
    return os.path.exists(os.path.join(MODEL_DIR, f"{ticker}_target_scaler.pkl"))


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Controls")

    # Stock input
    st.subheader("Stock Selection")
    popular = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN",
               "NVDA", "META", "RELIANCE.NS", "TCS.NS", "INFY.NS"]
    ticker_input = st.text_input("Enter ticker symbol", value=DEFAULT_TICKER,
                                 placeholder="AAPL, TSLA, RELIANCE.NS…").upper().strip()
    st.caption("💡 Indian stocks: add .NS (e.g. RELIANCE.NS)")

    period = st.selectbox("Historical period", VALID_PERIODS, index=2)

    st.divider()

    # Model config
    st.subheader("Model Settings")
    model_type   = st.radio("Model", ["LSTM", "GRU"], horizontal=True)
    show_baseline = st.checkbox("Compare with baselines", value=False)
    show_signals  = st.checkbox("Show Buy/Sell signals", value=True)
    show_sentiment= st.checkbox("Sentiment analysis", value=False)

    st.divider()

    # Train button
    train_btn = st.button("Train Model", type="primary", use_container_width=True)

    st.divider()
    st.caption("Built with Streamlit · yfinance · TensorFlow · Plotly")


# ──────────────────────────────────────────────────────────────────────────────
# Main header
# ──────────────────────────────────────────────────────────────────────────────

st.title("Stock Price Prediction System")
st.markdown("*LSTM-powered forecasting with interactive visualisation*")

# ── Validate and fetch ─────────────────────────────────────────────────────
if ticker_input:
    with st.spinner(f"Loading {ticker_input}…"):
        try:
            df_raw  = cached_fetch(ticker_input, period)
            info    = cached_info(ticker_input)
        except ValueError as e:
            st.error(f"❌ {e}")
            st.stop()

    # ── Company header ────────────────────────────────────────────────────
    col_name, col_px, col_chg, col_vol, col_mkt = st.columns(5)
    latest  = df_raw["Close"].iloc[-1]
    prev    = df_raw["Close"].iloc[-2]
    change  = latest - prev
    pct_chg = change / prev * 100
    delta_colour = "normal" if change >= 0 else "inverse"

    col_name.metric("Company",  info["name"][:25])
    col_px.metric("Price",      f"{info['currency']} {latest:.2f}", f"{change:+.2f}")
    col_chg.metric("Change",    f"{pct_chg:+.2f}%", delta_color=delta_colour)
    col_vol.metric("Volume",    f"{int(df_raw['Volume'].iloc[-1]):,}")
    col_mkt.metric("Sector",    info.get("sector", "N/A"))

    st.divider()

    # ──────────────────────────────────────────────────────────────────────
    # Tabs
    # ──────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Market Overview",
        "Technical Analysis",
        "Predictions",
        "Model Metrics",
        "Sentiment",
    ])

    # ── TAB 1: Market Overview ─────────────────────────────────────────────
    with tab1:
        st.subheader(f"{ticker_input} — Price History")

        df_feat = add_technical_indicators(df_raw.copy())
        fig_candle = plot_candlestick(df_feat, ticker_input)
        st.plotly_chart(fig_candle, use_container_width=True)

        with st.expander("Raw Data"):
            st.dataframe(df_raw.tail(50).sort_index(ascending=False),
                         use_container_width=True)

    # ── TAB 2: Technical Analysis ──────────────────────────────────────────
    with tab2:
        st.subheader("Technical Indicators")
        df_feat = add_technical_indicators(df_raw.copy())
        fig_tech = plot_technicals(df_feat, ticker_input)
        st.plotly_chart(fig_tech, use_container_width=True)

        # RSI interpretation
        rsi_last = df_feat["RSI_14"].iloc[-1]
        if rsi_last > 70:
            st.warning(f"RSI = {rsi_last:.1f} — Overbought zone (>70)")
        elif rsi_last < 30:
            st.info(f"RSI = {rsi_last:.1f} — Oversold zone (<30)")
        else:
            st.success(f"RSI = {rsi_last:.1f} — Neutral zone (30–70)")

    # ── TAB 3: Predictions ─────────────────────────────────────────────────
    with tab3:
        st.subheader("LSTM Predictions & Forecast")

        # ── Training ──────────────────────────────────────────────────────
        if train_btn:
            st.info("Training model… this may take a few minutes.")
            progress = st.progress(0, text="Preprocessing data…")

            data = prepare_data(df_raw.copy(), ticker_input)
            progress.progress(20, text="Building model…")

            n_features  = data["X_train"].shape[2]
            val_split   = int(len(data["X_train"]) * 0.1)
            X_val = data["X_train"][-val_split:]
            y_val = data["y_train"][-val_split:]
            X_tr  = data["X_train"][:-val_split]
            y_tr  = data["y_train"][:-val_split]

            mt = model_type.lower()
            m  = build_lstm(SEQUENCE_LENGTH, n_features) if mt == "lstm" \
                 else build_gru(SEQUENCE_LENGTH, n_features)

            progress.progress(40, text=f"Training {model_type}…")

            history = train_lstm(m, X_tr, y_tr, X_val, y_val,
                                 ticker_input, model_type=mt)
            save_deep_model(m, ticker_input, mt)

            # Cache in session so we don't reload from disk
            st.session_state["model"]   = m
            st.session_state["data"]    = data
            st.session_state["history"] = history
            st.session_state["ticker"]  = ticker_input

            progress.progress(80, text="Evaluating…")
            results = evaluation_report(
                m, data["X_test"], data["y_test"],
                data["target_scaler"], data["test_df"],
                data["feature_scaler"], data["feature_cols"], ticker_input
            )
            st.session_state["results"] = results
            progress.progress(100, text="Done!")
            st.success("Model trained successfully!")

        # ── Load from disk if already trained ─────────────────────────────
        if "results" not in st.session_state:
            mt = model_type.lower()
            if model_exists(ticker_input, mt) and scaler_exists(ticker_input):
                with st.spinner("Loading saved model…"):
                    from preprocessor import load_scalers
                    m    = load_deep_model(ticker_input, mt)
                    data = prepare_data(df_raw.copy(), ticker_input)
                    results = evaluation_report(
                        m, data["X_test"], data["y_test"],
                        data["target_scaler"], data["test_df"],
                        data["feature_scaler"], data["feature_cols"], ticker_input
                    )
                    st.session_state.update({
                        "model": m, "data": data,
                        "results": results, "ticker": ticker_input
                    })
            else:
                st.info("No trained model found.  Click ** Train Model** in the sidebar.")
                st.stop()

        results = st.session_state["results"]
        data    = st.session_state["data"]

        # ── Forecast table ────────────────────────────────────────────────
        st.markdown("####  7-Day Price Forecast")
        forecast_df = pd.DataFrame({
            "Date" : results["forecast_dates"].date,
            "Forecasted Price": [f"{info['currency']} {p:.2f}"
                                  for p in results["forecast_prices"]],
            "Direction": ["▲" if p > results["actual_prices"][-1] else "▼"
                          for p in results["forecast_prices"]],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        # ── Prediction chart ──────────────────────────────────────────────
        fig_pred = plot_prediction(
            test_dates      = results["test_dates"],
            actual_prices   = results["actual_prices"],
            predicted_prices= results["predicted_prices"],
            forecast_dates  = results["forecast_dates"],
            forecast_prices = results["forecast_prices"],
            ticker          = ticker_input,
            signals         = results["signals"] if show_signals else None,
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # ── CSV download ──────────────────────────────────────────────────
        csv_path = export_predictions_csv(
            results["test_dates"], results["actual_prices"],
            results["predicted_prices"], results["signals"],
            results["forecast_dates"], results["forecast_prices"],
            ticker_input,
        )
        with open(csv_path, "rb") as f:
            st.download_button(
                label="Download Predictions CSV",
                data=f,
                file_name=os.path.basename(csv_path),
                mime="text/csv",
            )

        # ── Baseline comparison ───────────────────────────────────────────
        if show_baseline:
            st.markdown("#### Baseline Model Comparison")
            bl = BaselineModels(ticker_input)
            bl.fit_all(data["X_train"], data["y_train"])
            bl_preds = bl.predict_all(data["X_test"])

            cmp_rows = []
            for bname, bpreds_scaled in bl_preds.items():
                bp = data["target_scaler"].inverse_transform(
                    bpreds_scaled.reshape(-1, 1)).flatten()
                actual = results["actual_prices"]
                m = compute_metrics(actual, bp[:len(actual)])
                cmp_rows.append({"Model": bname.replace("_", " ").title(), **m})

            # Add LSTM row
            lstm_m = results["metrics"]
            cmp_rows.append({"Model": "LSTM (Deep Learning)", **lstm_m})

            st.dataframe(pd.DataFrame(cmp_rows).set_index("Model"),
                         use_container_width=True)

    # ── TAB 4: Model Metrics ───────────────────────────────────────────────
    with tab4:
        st.subheader("Model Performance")
        if "results" in st.session_state:
            results = st.session_state["results"]
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("RMSE", f"{results['metrics']['RMSE']:.4f}")
            m_col2.metric("MAE",  f"{results['metrics']['MAE']:.4f}")
            m_col3.metric("R²",   f"{results['metrics']['R2']:.4f}")
            m_col4.metric("MAPE", f"{results['metrics']['MAPE']:.2f}%")

            if "history" in st.session_state:
                fig_loss = plot_training_history(st.session_state["history"])
                st.plotly_chart(fig_loss, use_container_width=True)

            with st.expander("How to interpret these metrics"):
                st.markdown("""
| Metric | What it means | Good range |
|--------|---------------|------------|
| **RMSE** | Root Mean Squared Error — average error in price units. Lower = better. | < 5% of avg price |
| **MAE** | Mean Absolute Error — average absolute error. More robust to outliers. | < 3% of avg price |
| **R²** | Coefficient of determination. 1.0 = perfect, 0.0 = no better than mean. | > 0.90 |
| **MAPE** | Mean Absolute Percentage Error — relative error in %. | < 5% |
                """)
        else:
            st.info("Train the model first to see metrics.")

    # ── TAB 5: Sentiment ───────────────────────────────────────────────────
    with tab5:
        st.subheader("News Sentiment Analysis")
        if show_sentiment:
            with st.spinner("Fetching news…"):
                sentiment = get_sentiment_score(ticker_input)

            score = sentiment["score"]
            label = sentiment["label"]
            col_s, col_l = st.columns([1, 3])
            col_s.metric("Sentiment Score", f"{score:.3f}", label)

            if label == "Positive":
                col_l.success(f"Market sentiment for {ticker_input} is **Positive**")
            elif label == "Negative":
                col_l.error(f"Market sentiment for {ticker_input} is **Negative**")
            else:
                col_l.info(f"Market sentiment for {ticker_input} is **Neutral**")

            if sentiment["articles"]:
                st.markdown("**Recent News Headlines:**")
                for art in sentiment["articles"][:5]:
                    emoji = "🟢" if art["score"] > 0.05 else "🔴" if art["score"] < -0.05 else "🟡"
                    st.markdown(f"{emoji} [{art['title']}]({art['url']}) — score: `{art['score']}`")
            else:
                st.warning("No articles found. Add a `NEWS_API_KEY` to `.env` for live sentiment.")
        else:
            st.info("Enable 'Sentiment analysis' in the sidebar to see this panel.")

else:
    st.info(" Enter a stock ticker in the sidebar to get started.")
