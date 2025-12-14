"""
Train LSTM models for next-day and next-week forecasts on Gold Futures (GC=F).
Input: gold_GCF_3y_1d.csv
Requires: tensorflow (or keras) installed.

Uses the *optimized* feature engineering pipeline (reduced ~25 features to avoid overfitting).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Try to import root_mean_squared_error (newer sklearn), fallback to manual calculation
try:
    from sklearn.metrics import root_mean_squared_error
    USE_RMSE_FUNC = True
except ImportError:
    USE_RMSE_FUNC = False

try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM
except ImportError as e:
    raise SystemExit("tensorflow/keras nicht installiert. Bitte `pip install tensorflow` ausführen.") from e

from pipelines.feature_engineering_optimized import create_optimized_features, get_feature_columns

# Try 10y data first, fallback to 3y
DATA_PATH = Path("gold_GCF_10y_1d.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path("gold_GCF_3y_1d.csv")
ROLLING_YEARS = 3  # keep LSTM on recent window to reduce regime drift


def build_sequences(df: pd.DataFrame, target: str, seq_len: int = 30):
    feature_cols = get_feature_columns(df, exclude_targets=True)
    data = df[feature_cols].values
    y = df[target].values

    Xs, ys = [], []
    for i in range(len(df) - seq_len):
        Xs.append(data[i : i + seq_len])
        ys.append(y[i + seq_len])
    Xs, ys = np.array(Xs), np.array(ys)
    return Xs, ys


def split_sequences(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate(y_true, y_pred):
    if USE_RMSE_FUNC:
        rmse = root_mean_squared_error(y_true, y_pred)
    else:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def build_lstm(input_shape):
    model = Sequential(
        [
            LSTM(64, return_sequences=False, input_shape=input_shape),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_and_eval(target="y_day"):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run fetch_gold.py first.")

    df = pd.read_csv(DATA_PATH)
    # Handle different column names
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    elif "Date" in df.columns:
        df = df.set_index("Date")
    else:
        df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)
    if ROLLING_YEARS is not None:
        cutoff = df.index.max() - pd.Timedelta(days=ROLLING_YEARS * 365)
        df = df[df.index >= cutoff]
    
    # Use optimized feature engineering pipeline (reduced feature set)
    print("🔧 Creating features with optimized pipeline...")
    df = create_optimized_features(df)
    print(f"   Created {len(get_feature_columns(df))} features")

    seq_len = 30
    X, y = build_sequences(df, target=target, seq_len=seq_len)

    # Scale features across all timesteps
    n_samples, n_steps, n_feats = X.shape
    scaler = StandardScaler()
    X_reshaped = X.reshape(n_samples * n_steps, n_feats)
    X_scaled = scaler.fit_transform(X_reshaped).reshape(n_samples, n_steps, n_feats)

    X_train, y_train, X_val, y_val, X_test, y_test = split_sequences(X_scaled, y)

    model = build_lstm(input_shape=(seq_len, n_feats))
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=1,
    )
    preds = model.predict(X_test).ravel()
    rmse, mae = evaluate(y_test, preds)
    return rmse, mae


def main():
    rmse_day, mae_day = train_and_eval(target="y_day")
    rmse_week, mae_week = train_and_eval(target="y_week")

    print("LSTM results:")
    print(f"  Next day   -> RMSE: {rmse_day:.3f}, MAE: {mae_day:.3f}")
    print(f"  Next week  -> RMSE: {rmse_week:.3f}, MAE: {mae_week:.3f}")


if __name__ == "__main__":
    main()
