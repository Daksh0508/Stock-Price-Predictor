# 📈 Stock Price Prediction System

> An end-to-end machine learning system for stock price forecasting with an interactive Streamlit dashboard.

---

## 📌 Project Overview

This project demonstrates a complete ML pipeline applied to financial time-series data:

- **Data collection** from Yahoo Finance (live, cached)
- **Feature engineering**: MA, EMA, RSI, MACD, volatility
- **Deep learning models**: LSTM and GRU (TensorFlow/Keras)
- **Baseline models**: Linear Regression, Decision Tree, Random Forest
- **Evaluation**: RMSE, MAE, R², MAPE with visual comparison
- **Interactive dashboard**: Streamlit with Plotly charts
- **Bonus**: News sentiment analysis, Buy/Sell signals, CSV export

---

## 🗂 Project Structure

```
stock_prediction/
│
├── app.py              ← Streamlit dashboard (entry point)
├── trainer.py          ← CLI training script
├── config.py           ← All hyperparameters & paths
├── data_loader.py      ← Yahoo Finance data fetching & caching
├── preprocessor.py     ← Feature engineering & MinMax scaling & sequences
├── models.py           ← LSTM, GRU, baseline model definitions
├── evaluator.py        ← Metrics, forecasting, buy/sell signals
├── utils.py            ← Plotly charts, CSV export, sentiment
│
├── data/               ← Cached parquet files (auto-created)
├── models/             ← Saved .keras and .pkl model files
├── exports/            ← Prediction CSV exports
├── notebooks/          ← Jupyter notebooks for exploration
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone / Download

```bash
git clone https://github.com/yourname/stock-prediction.git
cd stock-prediction
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Set News API key for sentiment analysis

Create a `.env` file in the project root:

```
NEWS_API_KEY=your_newsapi_key_here
```

Get a free key at [newsapi.org](https://newsapi.org).

---

## 🚀 Running the Project

### Option A — Dashboard (recommended)

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

1. Enter a ticker in the sidebar (e.g. `AAPL`, `TSLA`, `RELIANCE.NS`)
2. Choose historical period
3. Click **🚀 Train Model**
4. Explore the 5 tabs: Market Overview, Technical Analysis, Predictions, Metrics, Sentiment

### Option B — CLI Training

```bash
# Train LSTM on Apple stock (5 years)
python trainer.py --ticker AAPL --period 5y

# Train both LSTM and GRU, plus baseline comparison
python trainer.py --ticker TSLA --period 2y --gru --baselines

# Custom epochs and batch size
python trainer.py --ticker MSFT --period 5y --epochs 30 --batch 64
```

---

## 🧠 Model Architecture

### LSTM (Primary Model)

```
Input  →  (batch, 60, 16)         ← 60-day window, 16 features
LSTM₁  →  128 units, return_seq   ← Learns temporal patterns
Dropout →  0.20
LSTM₂  →  64  units
Dropout →  0.20
Dense  →  25  units, ReLU
Dense  →  1   unit               ← Next-day Close price (scaled)
```

### Why LSTM for stock prices?

| Method | Limitation |
|--------|-----------|
| Linear Regression | Assumes linear relationship, ignores order |
| Random Forest | Treats days as independent, no memory |
| LSTM | Has a cell state that persists information across time steps |

The LSTM's **forget gate** decides what past information to discard; the **input gate** decides what new information to store. This "memory" is ideal for sequential data like prices.

---

## 📊 Features Engineered

| Feature | Description |
|---------|-------------|
| MA_7, MA_21, MA_50 | Simple moving averages (trend) |
| EMA_12, EMA_26 | Exponential moving averages (weighted trend) |
| RSI_14 | Relative Strength Index (momentum, 0–100) |
| MACD, MACD_Signal | Moving Average Convergence Divergence |
| MACD_Hist | MACD histogram (crossover signal) |
| Daily_Return | % price change day-over-day |
| Volatility_21 | 21-day rolling standard deviation of returns |

---

## 📈 Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| RMSE | √(mean((y − ŷ)²)) | Error in price units |
| MAE | mean(|y − ŷ|) | Avg absolute error |
| R² | 1 - SS_res/SS_tot | Variance explained (1.0 = perfect) |
| MAPE | mean(|y−ŷ|/y)×100 | Relative % error |

---

## 🌐 Deployment

### Streamlit Cloud (free)

1. Push your project to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as the entry point
4. Add `NEWS_API_KEY` in Secrets if needed
5. Deploy ✅

### Render.com

1. Create a new **Web Service**
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Heroku

```bash
heroku create your-app-name
heroku buildpacks:set heroku/python
echo "web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0" > Procfile
git push heroku main
```

---

## 🧪 Testing Edge Cases

| Scenario | Handling |
|----------|---------|
| Invalid ticker (e.g. `XYZABC`) | `validate_ticker()` returns False → user sees clear error |
| No data for period | `fetch_stock_data()` raises `ValueError` with helpful message |
| Model not trained | Dashboard prompts user to train first |
| API rate limit | On-disk parquet cache (23-hour TTL) avoids repeated calls |
| Short history | SEQUENCE_LENGTH guard ensures at least 60 rows exist |

---

## 🏆 Bonus Features

- [x] Real-time-ish data (yfinance, refreshed every hour)
- [x] News sentiment analysis (VADER + NewsAPI)
- [x] Buy/Sell/Hold signals based on predicted return
- [x] Model comparison dashboard (LSTM vs baselines)
- [x] Export predictions as CSV

---

## 📚 References

- Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory*
- [yfinance documentation](https://python-yfinance.readthedocs.io/)
- [TensorFlow Keras LSTM guide](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- [Streamlit documentation](https://docs.streamlit.io)

---

## 👤 Author

Built as a final-year computer science project.  
Feel free to fork, extend, and deploy!
