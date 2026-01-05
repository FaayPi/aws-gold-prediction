"""
SageMaker-compatible training script for LightGBM (GC=F).
Reads training data from the "training" channel, uses 3-year rolling window,
feature engineering (optimized) and saves model + scaler + feature list.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except ImportError as e:
    raise SystemExit("lightgbm not installed. Please run `pip install lightgbm`.") from e

from pipelines.feature_engineering_optimized import create_optimized_features, get_feature_columns

ROLLING_YEARS = 3


def select_top_features(df: pd.DataFrame, target: str, top_n: int = 12) -> list[str]:
    feature_cols = get_feature_columns(df, exclude_targets=True)
    corr = df[feature_cols + [target]].corr()[target].drop(target)
    corr = corr.dropna().abs().sort_values(ascending=False)
    if corr.empty:
        return feature_cols
    return corr.head(top_n).index.tolist()


def split(df: pd.DataFrame, target: str, feature_cols: list[str], test_size=0.2):
    X = df[feature_cols]
    y = df[target]
    split_idx = int(len(df) * (1 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def eval_model(model, X_test, y_test, scaler=None):
    Xt = scaler.transform(X_test) if scaler else X_test
    pred = model.predict(Xt)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, pred)
    return rmse, mae


def train_model(X_train, y_train, params, scaler=None):
    Xt = scaler.fit_transform(X_train) if scaler else X_train
    model = lgb.LGBMRegressor(
        **params,
        objective="regression",
        metric="rmse",
        random_state=42,
        verbose=-1,
    )
    model.fit(Xt, y_train)
    return model


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "training"))
    parser.add_argument("--data-file", type=str, default="gold_GCF_10y_1d.csv")
    args_parsed = parser.parse_args(args=args)

    model_dir = Path(args_parsed.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args_parsed.train) / args_parsed.data_file
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    print(f"📊 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
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

    print("🔧 Creating optimized features...")
    df = create_optimized_features(df)
    print(f"   Created {len(get_feature_columns(df))} features")

    # Top features per target
    feats_day = select_top_features(df, target="y_day", top_n=12)
    feats_week = select_top_features(df, target="y_week", top_n=12)
    print(f"   Top features (day): {feats_day[:5]} ...")
    print(f"   Top features (week): {feats_week[:5]} ...")

    # Hyperparameters for LightGBM
    params_base = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }

    scaler_day = StandardScaler()
    X_train, X_test, y_train, y_test = split(df, target="y_day", feature_cols=feats_day)
    model_day = train_model(X_train, y_train, params_base, scaler=scaler_day)
    rmse_day, mae_day = eval_model(model_day, X_test, y_test, scaler=scaler_day)

    scaler_week = StandardScaler()
    X_train, X_test, y_train, y_test = split(df, target="y_week", feature_cols=feats_week)
    model_week = train_model(X_train, y_train, params_base, scaler=scaler_week)
    rmse_week, mae_week = eval_model(model_week, X_test, y_test, scaler=scaler_week)

    print("LightGBM results:")
    print(f"  Next day   -> RMSE: {rmse_day:.3f}, MAE: {mae_day:.3f}")
    print(f"  Next week  -> RMSE: {rmse_week:.3f}, MAE: {mae_week:.3f}")

    # Save artifacts
    joblib.dump(model_day, model_dir / "model_day.pkl")
    joblib.dump(model_week, model_dir / "model_week.pkl")
    joblib.dump(scaler_day, model_dir / "scaler_day.pkl")
    joblib.dump(scaler_week, model_dir / "scaler_week.pkl")
    joblib.dump(feats_day, model_dir / "features_day.pkl")
    joblib.dump(feats_week, model_dir / "features_week.pkl")
    print(f"✅ Models saved to {model_dir}")


def model_fn(model_dir):
    # Load artifacts for inference
    model_day = joblib.load(Path(model_dir) / "model_day.pkl")
    model_week = joblib.load(Path(model_dir) / "model_week.pkl")
    scaler_day = joblib.load(Path(model_dir) / "scaler_day.pkl")
    scaler_week = joblib.load(Path(model_dir) / "scaler_week.pkl")
    feats_day = joblib.load(Path(model_dir) / "features_day.pkl")
    feats_week = joblib.load(Path(model_dir) / "features_week.pkl")
    return {
        "model_day": model_day,
        "model_week": model_week,
        "scaler_day": scaler_day,
        "scaler_week": scaler_week,
        "feats_day": feats_day,
        "feats_week": feats_week,
    }


def predict_fn(input_data, model_bundle):
    """
    input_data: pd.DataFrame with feature columns present in feats_day/feats_week.
    model_bundle: dict from model_fn.
    """
    df = input_data.copy()
    # ensure datetime index if possible
    if not isinstance(df.index, pd.DatetimeIndex) and "datetime" in df.columns:
        df = df.set_index("datetime")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Use optimized features
    df_feats = create_optimized_features(df)

    # Day prediction
    feats_day = model_bundle["feats_day"]
    scaler_day = model_bundle["scaler_day"]
    model_day = model_bundle["model_day"]
    X_day = df_feats[feats_day]
    Xd = scaler_day.transform(X_day)
    pred_day = model_day.predict(Xd)

    # Week prediction
    feats_week = model_bundle["feats_week"]
    scaler_week = model_bundle["scaler_week"]
    model_week = model_bundle["model_week"]
    X_week = df_feats[feats_week]
    Xw = scaler_week.transform(X_week)
    pred_week = model_week.predict(Xw)

    return {
        "next_day": float(pred_day[-1]) if len(pred_day) > 0 else None,
        "next_week": float(pred_week[-1]) if len(pred_week) > 0 else None,
    }


if __name__ == "__main__":
    main()

