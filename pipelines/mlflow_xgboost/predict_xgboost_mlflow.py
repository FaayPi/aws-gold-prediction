"""
Load a registered XGBoost model from MLflow Model Registry and make predictions.
"""

import mlflow
import pandas as pd
from pathlib import Path
import argparse
import sys

# Import feature engineering
sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_engineering_optimized import create_optimized_features, get_feature_columns

MODEL_NAME = "xgboost_gold_price_production"


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
        print(f"   You can set the alias with: python pipelines/mlflow_xgboost/promote_xgboost_model_mlflow.py")
        return None


def predict_next_day(model, historical_data):
    """Make prediction for the next day."""
    # Create features from historical data
    df_features = create_optimized_features(historical_data)
    
    # Get the last row (most recent data)
    last_row = df_features.iloc[[-1]].copy()
    
    # Add horizon column
    last_row['horizon'] = 1
    
    # Make prediction
    prediction = model.predict(last_row)
    
    return prediction.iloc[0]['predicted_price'] if hasattr(prediction, 'iloc') else prediction[0]


def predict_next_week(model, historical_data):
    """Make prediction for the next week."""
    # Create features from historical data
    df_features = create_optimized_features(historical_data)
    
    # Get the last row (most recent data)
    last_row = df_features.iloc[[-1]].copy()
    
    # Add horizon column
    last_row['horizon'] = 5
    
    # Make prediction
    prediction = model.predict(last_row)
    
    return prediction.iloc[0]['predicted_price'] if hasattr(prediction, 'iloc') else prediction[0]


def main():
    parser = argparse.ArgumentParser(description="Load XGBoost model from MLflow and make predictions")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                       help="Model name in registry")
    parser.add_argument("--use-alias", action="store_true", default=True,
                       help="Use alias instead of stage (default: True)")
    parser.add_argument("--data-file", type=str, default=None,
                       help="Historical data file for prediction")
    parser.add_argument("--horizon", type=int, default=1,
                       help="Prediction horizon: 1 for next day, 5 for next week")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model_from_registry(args.model_name, args.use_alias)
    
    if model is None:
        return
    
    # If data file provided, load and make prediction
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
            
            print(f"\n📊 Loaded data from: {data_path}")
            print(f"   Data shape: {df.shape}")
            print(f"   Date range: {df.index.min()} to {df.index.max()}")
            
            # Make prediction
            if args.horizon == 1:
                prediction = predict_next_day(model, df)
                print(f"\n🔮 Next Day Prediction: ${prediction:.2f}/oz")
            elif args.horizon == 5:
                prediction = predict_next_week(model, df)
                print(f"\n🔮 Next Week Prediction: ${prediction:.2f}/oz")
            else:
                print(f"❌ Invalid horizon: {args.horizon}. Use 1 for next day or 5 for next week")
        else:
            print(f"❌ Data file not found: {data_path}")
    else:
        print("\n💡 To make predictions, provide --data-file argument")
        print("   Example: python predict_xgboost_mlflow.py --data-file gold_GCF_10y_1d.csv --horizon 1")


if __name__ == "__main__":
    main()

