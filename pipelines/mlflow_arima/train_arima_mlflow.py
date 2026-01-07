"""
MLflow-integrated training for ARIMA (GC=F).
Tracks parameters, metrics, models and enables experiment comparisons.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import json
import inspect
from pathlib import Path
from datetime import datetime

# Print immediately after basic imports
print("🚀 Starting ARIMA training...", flush=True)

# Load libraries with progress indicators
print("   Loading numpy...", flush=True)
import numpy as np

print("   Loading pandas...", flush=True)
import pandas as pd

print("   Loading sklearn...", flush=True)
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("   Loading statsmodels (this may take a moment)...", flush=True)
from statsmodels.tsa.arima.model import ARIMA

print("   Loading MLflow...", flush=True)
import mlflow
import mlflow.pyfunc

print("   ✓ All libraries loaded", flush=True)

ROLLING_YEARS = 3
ARIMA_ORDER = (3, 1, 3)


def load_series(data_path: Path) -> pd.Series:
    """Load and prepare the time series."""
    df = pd.read_csv(data_path)
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    elif "Date" in df.columns:
        df = df.set_index("Date")
    else:
        df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    cutoff = df.index.max() - pd.Timedelta(days=ROLLING_YEARS * 365)
    df = df[df.index >= cutoff]
    return df["close"]


def split_series(series: pd.Series, train_ratio: float = 0.8):
    n = len(series)
    train_end = int(n * train_ratio)
    return series.iloc[:train_end], series.iloc[train_end:]


def evaluate(true, pred):
    """Calculate RMSE, MAE and MAPE."""
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return rmse, mae, mape


def rolling_forecast(train: pd.Series, test: pd.Series, order, horizon: int):
    history = train.copy()
    preds = []
    test_copy = test.copy()
    total_steps = len(test_copy)
    
    print(f"   Rolling forecast: {total_steps} steps, horizon={horizon}", flush=True)
    
    for i in range(total_steps):
        if len(history) < 10:
            break
        try:
            # Progress update every 10 steps or at start/end
            if i % 10 == 0 or i == total_steps - 1:
                progress = (i + 1) / total_steps * 100
                print(f"   Step {i+1}/{total_steps} ({progress:.1f}%) - Fitting ARIMA{order}...", end='\r', flush=True)
            
            model = ARIMA(history, order=order)
            res = model.fit()
            fc = res.forecast(steps=horizon)
            pred = fc.iloc[-1] if horizon > 1 else fc.iloc[0]
            preds.append(pred)
            history = pd.concat([history, test_copy.iloc[i:i+1]])
        except Exception as e:
            print(f"\n   ⚠️  Warning: ARIMA forecast failed at step {i}: {e}", flush=True)
            break
    
    print(f"\n   ✓ Rolling forecast completed: {len(preds)} predictions", flush=True)
    return preds


def train_and_eval(series: pd.Series, order=ARIMA_ORDER):
    train, test = split_series(series, 0.8)
    
    print(f"   Train size: {len(train)}, Test size: {len(test)}", flush=True)

    # Next day predictions
    print("\n   📅 Evaluating Next Day predictions...", flush=True)
    preds_day = rolling_forecast(train, test, order, horizon=1)
    rmse_day = mae_day = mape_day = float("inf")
    if len(preds_day) > 0:
        test_day = test.iloc[:len(preds_day)]
        rmse_day, mae_day, mape_day = evaluate(test_day, preds_day)
        print(f"   ✓ Day evaluation: RMSE={rmse_day:.3f}, MAE={mae_day:.3f}", flush=True)

    # Next week predictions (5 trading days)
    print("\n   📅 Evaluating Next Week predictions...", flush=True)
    preds_week = rolling_forecast(train, test, order, horizon=5)
    rmse_week = mae_week = mape_week = float("inf")
    if len(preds_week) > 0:
        test_week = test.iloc[:len(preds_week)]
        rmse_week, mae_week, mape_week = evaluate(test_week, preds_week)
        print(f"   ✓ Week evaluation: RMSE={rmse_week:.3f}, MAE={mae_week:.3f}", flush=True)

    return {
        'day': (rmse_day, mae_day, mape_day),
        'week': (rmse_week, mae_week, mape_week)
    }


def main(args=None):
    parser = argparse.ArgumentParser(description="Train ARIMA model with MLflow tracking")
    parser.add_argument("--model-dir", type=str, default="models/model_arima")
    parser.add_argument("--data-file", type=str, default="gold_GCF_10y_1d.csv")
    parser.add_argument("--experiment-name", type=str, default="arima_gold_price")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--p", type=int, default=3, help="ARIMA p parameter")
    parser.add_argument("--d", type=int, default=1, help="ARIMA d parameter")
    parser.add_argument("--q", type=int, default=3, help="ARIMA q parameter")
    parser.add_argument("--rolling-years", type=int, default=3)
    parsed = parser.parse_args(args=args)

    print(f"   Parameters: ARIMA({parsed.p}, {parsed.d}, {parsed.q}), rolling_years={parsed.rolling_years}", flush=True)
    print(f"   Data file: {parsed.data_file}", flush=True)
    print(f"   Experiment: {parsed.experiment_name}", flush=True)
    
    # MLflow Setup
    print("\n📡 Connecting to MLflow...", flush=True)
    mlflow.set_experiment(parsed.experiment_name)
    print("   ✓ MLflow connected", flush=True)
    
    # Run name mit Timestamp falls nicht angegeben
    run_name = parsed.run_name or f"arima_{parsed.p}_{parsed.d}_{parsed.q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"   Starting run: {run_name}", flush=True)
    with mlflow.start_run(run_name=run_name):
        # Log Parameters
        mlflow.log_param("arima_p", parsed.p)
        mlflow.log_param("arima_d", parsed.d)
        mlflow.log_param("arima_q", parsed.q)
        mlflow.log_param("rolling_years", parsed.rolling_years)
        mlflow.log_param("train_test_split", 0.8)
        
        # Load data
        data_path = Path(parsed.data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        print(f"\n📊 Loading data from: {data_path}", flush=True)
        series = load_series(data_path)
        print(f"   Samples after rolling window: {len(series)}", flush=True)
        
        mlflow.log_param("data_samples", len(series))
        mlflow.log_param("data_start_date", str(series.index.min()))
        mlflow.log_param("data_end_date", str(series.index.max()))
        
        # Train and evaluate
        print("\n🚂 Training & evaluating ARIMA...", flush=True)
        order = (parsed.p, parsed.d, parsed.q)
        results = train_and_eval(series, order=order)
        
        # Log Metrics
        rmse_day, mae_day, mape_day = results['day']
        rmse_week, mae_week, mape_week = results['week']
        
        mlflow.log_metric("rmse_next_day", rmse_day)
        mlflow.log_metric("mae_next_day", mae_day)
        mlflow.log_metric("mape_next_day", mape_day)
        mlflow.log_metric("rmse_next_week", rmse_week)
        mlflow.log_metric("mae_next_week", mae_week)
        mlflow.log_metric("mape_next_week", mape_week)
        
        print(f"\n📈 ARIMA Results:", flush=True)
        print(f"  Next Day  -> RMSE: {rmse_day:.3f}, MAE: {mae_day:.3f}, MAPE: {mape_day:.2f}%", flush=True)
        print(f"  Next Week -> RMSE: {rmse_week:.3f}, MAE: {mae_week:.3f}, MAPE: {mape_week:.2f}%", flush=True)
        
        # Save model locally
        model_dir = Path(parsed.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "arima_model.pkl"
        
        model_data = {
            "series": series,
            "order": order,
            "rmse_day": rmse_day,
            "mae_day": mae_day,
            "mape_day": mape_day,
            "rmse_week": rmse_week,
            "mae_week": mae_week,
            "mape_week": mape_week,
            "trained_date": datetime.now().isoformat()
        }
        
        # Train final model on ALL data for production use
        print("\n🎯 Training final production model on all data...", flush=True)
        print(f"   Fitting ARIMA{order} on {len(series)} samples...", flush=True)
        
        # #region agent log
        import statsmodels
        log_data = {
            "sessionId": "debug-session",
            "runId": "post-fix",
            "hypothesisId": "A",
            "location": "train_arima_mlflow.py:218",
            "message": "Checking statsmodels version and ARIMA.fit signature",
            "data": {
                "statsmodels_version": statsmodels.__version__,
                "arima_class": str(ARIMA),
                "arima_module": str(ARIMA.__module__)
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        with open("/Users/feepieper/Desktop/capstone_projects/ML_model_gold_price_predictions/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_data) + "\n")
        # #endregion
        
        final_model = ARIMA(series, order=order)
        
        # #region agent log
        try:
            fit_signature = inspect.signature(final_model.fit)
            fit_params = list(fit_signature.parameters.keys())
        except Exception as e:
            fit_params = f"Error getting signature: {e}"
        log_data = {
            "sessionId": "debug-session",
            "runId": "post-fix",
            "hypothesisId": "B",
            "location": "train_arima_mlflow.py:219",
            "message": "ARIMA.fit() method signature",
            "data": {
                "fit_parameters": fit_params if isinstance(fit_params, list) else [fit_params],
                "attempting_disp": False,
                "fix_applied": True
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        with open("/Users/feepieper/Desktop/capstone_projects/ML_model_gold_price_predictions/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_data) + "\n")
        # #endregion
        
        # #region agent log
        log_data = {
            "sessionId": "debug-session",
            "runId": "post-fix",
            "hypothesisId": "C",
            "location": "train_arima_mlflow.py:219",
            "message": "Before calling fit() without disp parameter",
            "data": {
                "series_length": len(series),
                "order": str(order),
                "fix_applied": True
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        with open("/Users/feepieper/Desktop/capstone_projects/ML_model_gold_price_predictions/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_data) + "\n")
        # #endregion
        
        try:
            # Note: disp parameter removed in statsmodels 0.14+ - verbosity controlled via logging
            final_model_fitted = final_model.fit()
        except TypeError as e:
            # #region agent log
            log_data = {
                "sessionId": "debug-session",
                "runId": "post-fix",
                "hypothesisId": "D",
                "location": "train_arima_mlflow.py:219",
                "message": "TypeError caught - fit() parameter issue (unexpected)",
                "data": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_args": list(e.args) if hasattr(e, 'args') and e.args else None,
                    "fix_applied": True
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
            with open("/Users/feepieper/Desktop/capstone_projects/ML_model_gold_price_predictions/.cursor/debug.log", "a") as f:
                f.write(json.dumps(log_data) + "\n")
            # #endregion
            raise
        
        # #region agent log
        log_data = {
            "sessionId": "debug-session",
            "runId": "post-fix",
            "hypothesisId": "E",
            "location": "train_arima_mlflow.py:219",
            "message": "After successful fit() call - fix verified",
            "data": {
                "fitted_model_type": str(type(final_model_fitted)),
                "fix_applied": True,
                "success": True
            },
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        with open("/Users/feepieper/Desktop/capstone_projects/ML_model_gold_price_predictions/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_data) + "\n")
        # #endregion
        
        print(f"   ✓ Final model trained on {len(series)} samples", flush=True)
        
        # Update model_data with fitted model
        model_data["fitted_model"] = final_model_fitted
        model_data["final_series"] = series
        
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"\n💾 Model saved to {model_path}", flush=True)
        
        # Define ARIMA Model Wrapper for MLflow
        class ARIMAModel(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                import pickle
                with open(context.artifacts["model_path"], "rb") as f:
                    self.model_data = pickle.load(f)
                self.fitted_model = self.model_data["fitted_model"]
                self.order = self.model_data["order"]
            
            def predict(self, context, model_input):
                """
                Predict next day or next week gold price.
                
                model_input can be:
                - DataFrame with 'horizon' column (1 for next day, 5 for next week)
                - Integer (number of steps to forecast)
                - Empty DataFrame (defaults to next day)
                """
                import pandas as pd
                import numpy as np
                
                # Default: predict next day
                if model_input is None or len(model_input) == 0:
                    horizon = 1
                elif isinstance(model_input, (int, np.integer)):
                    horizon = int(model_input)
                elif isinstance(model_input, pd.DataFrame):
                    if 'horizon' in model_input.columns:
                        horizon = int(model_input['horizon'].iloc[0])
                    else:
                        horizon = 1
                else:
                    horizon = 1
                
                # Get forecast
                forecast = self.fitted_model.forecast(steps=horizon)
                
                # Return as DataFrame for consistency
                if horizon == 1:
                    return pd.DataFrame({'predicted_price': [forecast.iloc[0]]})
                else:
                    return pd.DataFrame({'predicted_price': forecast.values})
        
        # Log model in MLflow format
        print("\n📦 Logging model to MLflow...", flush=True)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ARIMAModel(),
            artifacts={"model_path": str(model_path)}
        )
        
        # Log additional artifacts (optional)
        mlflow.log_artifact(str(data_path), "data")
        
        # Log tags
        mlflow.set_tag("model_type", "ARIMA")
        mlflow.set_tag("target", "gold_price")
        mlflow.set_tag("ticker", "GC=F")
        
        print(f"\n✅ MLflow Run completed!", flush=True)
        print(f"   Run ID: {mlflow.active_run().info.run_id}", flush=True)
        print(f"   Experiment: {parsed.experiment_name}", flush=True)
        print(f"   View UI: http://localhost:5000", flush=True)


if __name__ == "__main__":
    main()

