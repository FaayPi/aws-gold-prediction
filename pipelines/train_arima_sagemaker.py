"""
SageMaker-kompatibles Training für ARIMA (GC=F).
Liest Trainingsdaten aus dem SageMaker-Channel "training", nutzt 3y Rolling Window
und speichert das Modell (Serie + Order) als Pickle.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROLLING_YEARS = 3  # auf die letzten 3 Jahre beschränken
ARIMA_ORDER = (3, 1, 3)


def load_series(data_path: Path) -> pd.Series:
    df = pd.read_csv(data_path)
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    elif "Date" in df.columns:
        df = df.set_index("Date")
    else:
        df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Rolling Window
    cutoff = df.index.max() - pd.Timedelta(days=ROLLING_YEARS * 365)
    df = df[df.index >= cutoff]
    return df["close"]


def split_series(series: pd.Series, train_ratio: float = 0.8):
    n = len(series)
    train_end = int(n * train_ratio)
    return series.iloc[:train_end], series.iloc[train_end:]


def evaluate(true, pred):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    return rmse, mae


def rolling_forecast(train: pd.Series, test: pd.Series, order, horizon: int):
    history = train.copy()
    preds = []
    test_copy = test.copy()
    for i in range(len(test_copy)):
        if len(history) < 10:
            break
        try:
            model = ARIMA(history, order=order)
            res = model.fit()
            fc = res.forecast(steps=horizon)
            pred = fc.iloc[-1] if horizon > 1 else fc.iloc[0]
            preds.append(pred)
            history = pd.concat([history, test_copy.iloc[i:i+1]])
        except Exception as e:
            print(f"Warning: ARIMA forecast failed at step {i}: {e}")
            break
    return preds


def train_and_eval(series: pd.Series, order=ARIMA_ORDER):
    train, test = split_series(series, 0.8)

    # next day
    preds_day = rolling_forecast(train, test, order, horizon=1)
    rmse_day = mae_day = float("inf")
    if len(preds_day) > 0:
        test_day = test.iloc[:len(preds_day)]
        rmse_day, mae_day = evaluate(test_day, preds_day)

    # next week (5 trading days)
    preds_week = rolling_forecast(train, test, order, horizon=5)
    rmse_week = mae_week = float("inf")
    if len(preds_week) > 0:
        test_week = test.iloc[:len(preds_week)]
        rmse_week, mae_week = evaluate(test_week, preds_week)

    return (rmse_day, mae_day), (rmse_week, mae_week)


def save_model(model_dir: Path, series: pd.Series, order):
    model_path = model_dir / "arima_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"series": series, "order": order}, f)
    print(f"✅ Model saved to {model_path}")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "training"))
    parser.add_argument("--data-file", type=str, default="gold_GCF_10y_1d.csv")
    parsed = parser.parse_args(args=args)

    model_dir = Path(parsed.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(parsed.train) / parsed.data_file
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    print(f"📊 Loading data from: {data_path}")
    series = load_series(data_path)
    print(f"   Samples after rolling window: {len(series)}")

    print("\n🚂 Training & evaluating ARIMA...")
    (rmse_day, mae_day), (rmse_week, mae_week) = train_and_eval(series, order=ARIMA_ORDER)
    print(f"ARIMA results:")
    print(f"  Next day   -> RMSE: {rmse_day:.3f}, MAE: {mae_day:.3f}")
    print(f"  Next week  -> RMSE: {rmse_week:.3f}, MAE: {mae_week:.3f}")

    print("\n💾 Saving model...")
    save_model(model_dir, series, ARIMA_ORDER)


def model_fn(model_dir):
    # For SageMaker inference: load serialized content
    with open(Path(model_dir) / "arima_model.pkl", "rb") as f:
        payload = pickle.load(f)
    return payload


def predict_fn(input_data, model):
    """
    input_data: pd.DataFrame with a datetime index and a 'close' column (recent history)
    model: dict with 'series' and 'order'
    """
    series = model["series"]
    order = model["order"]
    # append incoming history if provided
    if isinstance(input_data, pd.DataFrame) and "close" in input_data.columns:
        s = input_data["close"]
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        series = pd.concat([series, s])
    model_arima = ARIMA(series, order=order)
    res = model_arima.fit()
    # Forecast next day
    fc1 = res.forecast(steps=1).iloc[0]
    # Forecast next 5 days
    fc5 = res.forecast(steps=5).iloc[-1]
    return {"next_day": float(fc1), "next_week": float(fc5)}


if __name__ == "__main__":
    main()

