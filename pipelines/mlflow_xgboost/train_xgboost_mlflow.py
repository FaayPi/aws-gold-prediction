"""
MLflow-integrated training for XGBoost (GC=F).
Tracks parameters, metrics, models and enables experiment comparisons.
"""

from __future__ import annotations

import argparse
import os
import pickle
import joblib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.pyfunc
import mlflow.sklearn

# Import feature engineering
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_engineering_optimized import create_optimized_features, get_feature_columns

ROLLING_YEARS = 3


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and prepare the time series data."""
    df = pd.read_csv(data_path)
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    elif "Date" in df.columns:
        df = df.set_index("Date")
    else:
        df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Apply rolling window
    cutoff = df.index.max() - pd.Timedelta(days=ROLLING_YEARS * 365)
    df = df[df.index >= cutoff]
    
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split data into train and test sets."""
    n = len(df)
    train_end = int(n * train_ratio)
    return df.iloc[:train_end], df.iloc[train_end:]


def evaluate(true, pred):
    """Calculate RMSE, MAE and MAPE."""
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return rmse, mae, mape


def train_xgboost_model(X_train, y_train, X_test, y_test, hyperparams=None):
    """Train XGBoost model and return metrics."""
    if hyperparams is None:
        hyperparams = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    # Add eval_metric and verbose to hyperparams (XGBoost 2.0+ API)
    hyperparams['eval_metric'] = 'rmse'
    hyperparams['verbose'] = True
    
    print(f"   Training with {hyperparams['n_estimators']} estimators, max_depth={hyperparams['max_depth']}...")
    model = XGBRegressor(**hyperparams)
    
    # Use eval_set for better progress tracking
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set
    )
    
    print("   ✓ Training completed!")
    
    # Predictions
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    # Metrics
    rmse_train, mae_train, mape_train = evaluate(y_train, pred_train)
    rmse_test, mae_test, mape_test = evaluate(y_test, pred_test)
    
    return {
        'model': model,
        'train_metrics': (rmse_train, mae_train, mape_train),
        'test_metrics': (rmse_test, mae_test, mape_test)
    }


def main(args=None):
    parser = argparse.ArgumentParser(description="Train XGBoost model with MLflow tracking")
    parser.add_argument("--model-dir", type=str, default="models/model_xgb")
    parser.add_argument("--data-file", type=str, default="gold_GCF_10y_1d.csv")
    parser.add_argument("--experiment-name", type=str, default="xgboost_gold_price")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of boosting rounds")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Column sample ratio")
    parser.add_argument("--rolling-years", type=int, default=3)
    parsed = parser.parse_args(args=args)

    # MLflow Setup
    mlflow.set_experiment(parsed.experiment_name)
    
    # Run name with timestamp if not provided
    run_name = parsed.run_name or f"xgboost_{parsed.n_estimators}_{parsed.max_depth}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log Parameters
        mlflow.log_param("n_estimators", parsed.n_estimators)
        mlflow.log_param("max_depth", parsed.max_depth)
        mlflow.log_param("learning_rate", parsed.learning_rate)
        mlflow.log_param("subsample", parsed.subsample)
        mlflow.log_param("colsample_bytree", parsed.colsample_bytree)
        mlflow.log_param("rolling_years", parsed.rolling_years)
        mlflow.log_param("train_test_split", 0.8)
        
        # Load data
        data_path = Path(parsed.data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        print(f"📊 Loading data from: {data_path}")
        df = load_data(data_path)
        print(f"   Samples after rolling window: {len(df)}")
        
        mlflow.log_param("data_samples", len(df))
        mlflow.log_param("data_start_date", str(df.index.min()))
        mlflow.log_param("data_end_date", str(df.index.max()))
        
        # Feature Engineering
        print("\n🔧 Creating features...")
        df_features = create_optimized_features(df)
        feature_cols = get_feature_columns(df_features)
        print(f"   Features created: {len(feature_cols)}")
        mlflow.log_param("n_features", len(feature_cols))
        
        # Split data
        train_data, test_data = split_data(df_features, 0.8)
        
        # Prepare features and targets for Day prediction
        print("\n🚂 Training XGBoost model for Next Day prediction...")
        print(f"   Training samples: {len(train_data)}, Test samples: {len(test_data)}")
        X_train_day = train_data[feature_cols]
        y_train_day = train_data['y_day']
        X_test_day = test_data[feature_cols]
        y_test_day = test_data['y_day']
        
        # Scale features
        print("   Scaling features...")
        scaler_day = StandardScaler()
        X_train_day_scaled = scaler_day.fit_transform(X_train_day)
        X_test_day_scaled = scaler_day.transform(X_test_day)
        
        # Train model
        hyperparams_day = {
            'n_estimators': parsed.n_estimators,
            'max_depth': parsed.max_depth,
            'learning_rate': parsed.learning_rate,
            'subsample': parsed.subsample,
            'colsample_bytree': parsed.colsample_bytree,
            'random_state': 42,
            'verbose': True  # Add verbose
        }
        
        result_day = train_xgboost_model(
            X_train_day_scaled, y_train_day,
            X_test_day_scaled, y_test_day,
            hyperparams_day
        )
        
        rmse_day, mae_day, mape_day = result_day['test_metrics']
        print(f"   ✓ Day model completed - RMSE: {rmse_day:.3f}, MAE: {mae_day:.3f}")
        
        # Prepare features and targets for Week prediction
        print("\n🚂 Training XGBoost model for Next Week prediction...")
        print(f"   Training samples: {len(train_data)}, Test samples: {len(test_data)}")
        X_train_week = train_data[feature_cols]
        y_train_week = train_data['y_week']
        X_test_week = test_data[feature_cols]
        y_test_week = test_data['y_week']
        
        # Scale features
        print("   Scaling features...")
        scaler_week = StandardScaler()
        X_train_week_scaled = scaler_week.fit_transform(X_train_week)
        X_test_week_scaled = scaler_week.transform(X_test_week)
        
        # Train model
        result_week = train_xgboost_model(
            X_train_week_scaled, y_train_week,
            X_test_week_scaled, y_test_week,
            hyperparams_day
        )
        
        rmse_week, mae_week, mape_week = result_week['test_metrics']
        print(f"   ✓ Week model completed - RMSE: {rmse_week:.3f}, MAE: {mae_week:.3f}")
        
        # Log Metrics
        mlflow.log_metric("rmse_next_day", rmse_day)
        mlflow.log_metric("mae_next_day", mae_day)
        mlflow.log_metric("mape_next_day", mape_day)
        mlflow.log_metric("rmse_next_week", rmse_week)
        mlflow.log_metric("mae_next_week", mae_week)
        mlflow.log_metric("mape_next_week", mape_week)
        
        print(f"\n📈 XGBoost Results:")
        print(f"  Next Day  -> RMSE: {rmse_day:.3f}, MAE: {mae_day:.3f}, MAPE: {mape_day:.2f}%")
        print(f"  Next Week -> RMSE: {rmse_week:.3f}, MAE: {mae_week:.3f}, MAPE: {mape_week:.2f}%")
        
        # Save models locally
        model_dir = Path(parsed.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Day model
        joblib.dump(result_day['model'], model_dir / "model_day.pkl")
        joblib.dump(scaler_day, model_dir / "scaler_day.pkl")
        joblib.dump(feature_cols, model_dir / "features_day.pkl")
        
        # Save Week model
        joblib.dump(result_week['model'], model_dir / "model_week.pkl")
        joblib.dump(scaler_week, model_dir / "scaler_week.pkl")
        joblib.dump(feature_cols, model_dir / "features_week.pkl")
        
        print(f"\n💾 Models saved to {model_dir}")
        
        # Define XGBoost Model Wrapper for MLflow
        class XGBoostModel(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                import joblib
                self.model_day = joblib.load(context.artifacts["model_day_path"])
                self.model_week = joblib.load(context.artifacts["model_week_path"])
                self.scaler_day = joblib.load(context.artifacts["scaler_day_path"])
                self.scaler_week = joblib.load(context.artifacts["scaler_week_path"])
                self.features_day = joblib.load(context.artifacts["features_day_path"])
                self.features_week = joblib.load(context.artifacts["features_week_path"])
            
            def predict(self, context, model_input):
                """
                Predict next day or next week gold price.
                
                model_input should be a DataFrame with feature columns and 'horizon' column:
                - horizon=1 for next day
                - horizon=5 for next week
                """
                import pandas as pd
                import numpy as np
                
                if model_input is None or len(model_input) == 0:
                    raise ValueError("model_input cannot be empty")
                
                # Determine horizon
                if 'horizon' in model_input.columns:
                    horizon = int(model_input['horizon'].iloc[0])
                else:
                    horizon = 1  # Default to next day
                
                # Select model and scaler
                if horizon == 1:
                    model = self.model_day
                    scaler = self.scaler_day
                    features = self.features_day
                else:
                    model = self.model_week
                    scaler = self.scaler_week
                    features = self.features_week
                
                # Extract features
                X = model_input[features]
                X_scaled = scaler.transform(X)
                
                # Predict
                predictions = model.predict(X_scaled)
                
                return pd.DataFrame({'predicted_price': predictions})
        
        # Log models in MLflow format
        print("\n📦 Logging models to MLflow...")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=XGBoostModel(),
            artifacts={
                "model_day_path": str(model_dir / "model_day.pkl"),
                "model_week_path": str(model_dir / "model_week.pkl"),
                "scaler_day_path": str(model_dir / "scaler_day.pkl"),
                "scaler_week_path": str(model_dir / "scaler_week.pkl"),
                "features_day_path": str(model_dir / "features_day.pkl"),
                "features_week_path": str(model_dir / "features_week.pkl")
            }
        )
        
        # Log additional artifacts
        mlflow.log_artifact(str(data_path), "data")
        
        # Log tags
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("target", "gold_price")
        mlflow.set_tag("ticker", "GC=F")
        
        print(f"\n✅ MLflow Run completed!")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")
        print(f"   Experiment: {parsed.experiment_name}")
        print(f"   View UI: http://localhost:5000")


if __name__ == "__main__":
    main()

