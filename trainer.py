import argparse
import sys
import logging
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from data_loader import fetch_stock_data, get_ticker_info, validate_ticker
from preprocessor import prepare_data
from models import build_lstm, build_gru, train_lstm, BaselineModels, save_deep_model
from evaluator import evaluation_report, compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",    default="AAPL")
    p.add_argument("--period",    default="5y")
    p.add_argument("--epochs",    default=50, type=int)
    p.add_argument("--batch",     default=32, type=int)
    p.add_argument("--gru",       action="store_true")
    p.add_argument("--baselines", action="store_true")
    return p.parse_args()


def print_metrics(results, name):
    m = results["metrics"]
    print(f"\n--- {name} ---")
    print(f"  RMSE : {m['RMSE']:.4f}")
    print(f"  MAE  : {m['MAE']:.4f}")
    print(f"  R2   : {m['R2']:.4f}")
    print(f"  MAPE : {m['MAPE']:.2f}%")


def main():
    args   = parse_args()
    ticker = args.ticker.strip().upper()
    seq    = config.SEQUENCE_LENGTH

    print(f"\nValidating ticker: {ticker}")
    if not validate_ticker(ticker):
        print(f"Invalid ticker {ticker}")
        sys.exit(1)

    info = get_ticker_info(ticker)
    print(f"{info['name']} | {info['sector']}")

    print(f"\nFetching {args.period} of data...")
    df = fetch_stock_data(ticker, period=args.period)
    print(f"   {len(df)} rows  |  {df.index[0].date()} to {df.index[-1].date()}")

    print("\nPreprocessing...")
    data = prepare_data(df, ticker)

    nf    = data["X_train"].shape[2]
    val_n = int(len(data["X_train"]) * 0.1)
    X_val = data["X_train"][-val_n:]
    y_val = data["y_train"][-val_n:]
    X_tr  = data["X_train"][:-val_n]
    y_tr  = data["y_train"][:-val_n]

    print(f"   Features={nf}  Train={len(X_tr)}  Val={len(X_val)}  Test={len(data['X_test'])}")

    print("\nTraining LSTM...")
    model = build_lstm(seq, nf)
    train_lstm(model, X_tr, y_tr, X_val, y_val, ticker, "lstm")
    save_deep_model(model, ticker, "lstm")

    results = evaluation_report(
        model, data["X_test"], data["y_test"],
        data["target_scaler"], data["test_df"],
        data["feature_scaler"], data["feature_cols"], ticker
    )
    print_metrics(results, "LSTM")

    print("\n7-Day Forecast:")
    for d, p in zip(results["forecast_dates"], results["forecast_prices"]):
        print(f"   {d.date()}  ->  {info.get('currency', 'USD')} {p:.2f}")

    print(f"\nDone! Model saved to models/{ticker}_lstm.keras")
    print("Run: streamlit run app.py")


if __name__ == "__main__":
    main()