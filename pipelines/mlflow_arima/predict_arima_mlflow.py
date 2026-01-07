"""
Load a registered ARIMA model from MLflow Model Registry and make predictions.
"""

import mlflow
import pandas as pd
from pathlib import Path
import argparse

MODEL_NAME = "arima_gold_price_production"


def load_model_from_registry(model_name=MODEL_NAME, use_alias=True):
    """Load the Production model from MLflow Model Registry using alias."""
    try:
        if use_alias:
            # New method: Use alias (MLflow 2.0+)
            model_uri = f"models:/{model_name}@Production"
        else:
            # Old method: Use stage (deprecated)
            model_uri = f"models:/{model_name}/Production"
        
        model = mlflow.pyfunc.load_model(model_uri)
        
        print(f"✅ Model loaded: {model_name} (Production alias)")
        print(f"   Model URI: {model_uri}")
        
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"   Make sure model '{model_name}' is registered and has 'Production' alias")
        print(f"   You can set the alias with: python pipelines/mlflow_arima/promote_arima_model_mlflow.py")
        return None


def predict_next_day(model, historical_data=None):
    """
    Make prediction for the next day using ARIMA model.
    
    Note: ARIMA models use the data they were trained on for forecasting.
    The historical_data parameter is optional and mainly for logging purposes.
    """
    import pandas as pd
    
    # Create input DataFrame with horizon=1 (next day)
    input_df = pd.DataFrame({'horizon': [1]})
    
    # Make prediction using MLflow model
    prediction_result = model.predict(input_df)
    
    # Extract prediction value
    if isinstance(prediction_result, pd.DataFrame):
        predicted_price = prediction_result['predicted_price'].iloc[0]
    elif hasattr(prediction_result, 'iloc'):
        predicted_price = prediction_result.iloc[0]
    elif isinstance(prediction_result, (list, tuple, pd.Series)):
        predicted_price = prediction_result[0]
    else:
        predicted_price = float(prediction_result)
    
    return predicted_price


def predict_next_week(model, historical_data=None):
    """
    Make prediction for the next week (5 trading days) using ARIMA model.
    
    Note: ARIMA models use the data they were trained on for forecasting.
    The historical_data parameter is optional and mainly for logging purposes.
    """
    import pandas as pd
    
    # Create input DataFrame with horizon=5 (next week)
    input_df = pd.DataFrame({'horizon': [5]})
    
    # Make prediction using MLflow model
    prediction_result = model.predict(input_df)
    
    # Extract prediction value (last value of the 5-day forecast)
    if isinstance(prediction_result, pd.DataFrame):
        predicted_price = prediction_result['predicted_price'].iloc[-1]
    elif hasattr(prediction_result, 'iloc'):
        predicted_price = prediction_result.iloc[-1]
    elif isinstance(prediction_result, (list, tuple, pd.Series)):
        predicted_price = prediction_result[-1]
    else:
        predicted_price = float(prediction_result)
    
    return predicted_price


def main():
    parser = argparse.ArgumentParser(description="Load ARIMA model from MLflow and make predictions")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                       help="Model name in registry")
    parser.add_argument("--use-alias", action="store_true", default=True,
                       help="Use alias instead of stage (default: True)")
    parser.add_argument("--data-file", type=str, default=None,
                       help="Historical data file (optional, for logging purposes)")
    parser.add_argument("--horizon", type=int, default=1,
                       help="Prediction horizon: 1 for next day, 5 for next week")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model_from_registry(args.model_name, args.use_alias)
    
    if model is None:
        return
    
    # Load historical data if provided (for logging/info purposes)
    historical_data = None
    if args.data_file:
        data_path = Path(args.data_file)
        if data_path.exists():
            df = pd.read_csv(data_path)
            
            # Set index
            if "Date" in df.columns:
                df = df.set_index("Date")
            elif "datetime" in df.columns:
                df = df.set_index("datetime")
            else:
                df = df.set_index(df.columns[0])
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            historical_data = df
            print(f"\n📊 Loaded data from: {data_path}")
            print(f"   Data shape: {df.shape}")
            print(f"   Date range: {df.index.min()} to {df.index.max()}")
        else:
            print(f"⚠️  Data file not found: {data_path}")
            print("   Continuing without historical data (ARIMA uses trained data for prediction)")
    
    # Make prediction based on horizon
    if args.horizon == 1:
        prediction = predict_next_day(model, historical_data)
        print(f"\n🔮 Next Day Prediction: ${prediction:.2f}/oz")
    elif args.horizon == 5:
        prediction = predict_next_week(model, historical_data)
        print(f"\n🔮 Next Week Prediction: ${prediction:.2f}/oz")
    else:
        print(f"❌ Invalid horizon: {args.horizon}. Use 1 for next day or 5 for next week")
        return
    
    if historical_data is None:
        print("\n💡 Note: ARIMA model uses the data it was trained on for forecasting.")
        print("   Historical data file is optional and only used for logging purposes.")


if __name__ == "__main__":
    main()
