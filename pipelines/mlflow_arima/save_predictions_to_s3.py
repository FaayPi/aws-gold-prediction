"""
Daily script to generate ARIMA predictions and save them to S3.
This script should run daily (e.g., via EventBridge/Lambda or cron job)
after the model has been retrained.

The predictions are saved as a simple JSON file in S3, which the Lambda API
can read without needing to load the model or any ML dependencies.
"""

import json
import pickle
import boto3
from pathlib import Path
from datetime import datetime
import argparse

MODEL_NAME = "arima_gold_price_production"
MODEL_VERSION = "2"
S3_BUCKET = "gold-ml-mlflow-artifacts"
S3_MODEL_KEY = "models/arima_model.pkl"  # Model in S3 (from extract_model_for_lambda.py)
S3_PREDICTIONS_KEY = "predictions/latest.json"  # Predictions output


def load_model_from_s3(bucket=S3_BUCKET, key=S3_MODEL_KEY):
    """
    Load ARIMA model directly from S3 pickle file.
    This avoids all MLflow version compatibility issues.
    """
    try:
        s3_client = boto3.client('s3')
        
        print(f"📥 Loading model from S3: s3://{bucket}/{key}")
        
        # Load pickle from S3
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        model_data = pickle.loads(obj['Body'].read())
        
        # Extract the fitted model
        if isinstance(model_data, dict) and "fitted_model" in model_data:
            fitted_model = model_data["fitted_model"]
            order = model_data.get("order", "N/A")
            print(f"✅ Model loaded from S3")
            print(f"   ARIMA order: {order}")
            print(f"   Model type: {type(fitted_model)}")
            return fitted_model
        else:
            # Fallback: assume it's already the fitted model
            print(f"✅ Model loaded from S3 (direct model)")
            print(f"   Model type: {type(model_data)}")
            return model_data
            
    except Exception as e:
        print(f"❌ Error loading model from S3: {e}")
        print(f"   Make sure the model exists at: s3://{bucket}/{key}")
        print(f"   You can create it by running: python pipelines/mlflow_arima/extract_model_for_lambda.py")
        raise


def load_model_from_local(mlruns_path="mlruns"):
    """
    Load ARIMA model from local MLflow runs directory.
    Fallback if S3 model doesn't exist.
    """
    try:
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        model_version = client.get_model_version(MODEL_NAME, MODEL_VERSION)
        run_id = model_version.run_id
        
        print(f"📥 Loading model from local MLflow: run_id={run_id}")
        
        # Find the pickle file in local mlruns directory
        mlruns_dir = Path(mlruns_path)
        model_path = mlruns_dir / "1" / "models" / f"m-{run_id[:32]}" / "artifacts" / "artifacts" / "arima_model.pkl"
        
        if not model_path.exists():
            # Try alternative path
            model_path = mlruns_dir / "1" / run_id / "artifacts" / "model" / "artifacts" / "arima_model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model pickle not found in {mlruns_dir}")
        
        print(f"   Found model at: {model_path}")
        
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict) and "fitted_model" in model_data:
            fitted_model = model_data["fitted_model"]
            print(f"✅ Model loaded from local MLflow")
            return fitted_model
        else:
            print(f"✅ Model loaded from local MLflow (direct model)")
            return model_data
            
    except Exception as e:
        print(f"❌ Error loading model from local MLflow: {e}")
        raise


def generate_predictions(fitted_model):
    """
    Generate predictions for tomorrow (t+1) and day after tomorrow (t+2).
    
    Args:
        fitted_model: The fitted ARIMA model (from statsmodels)
    
    Returns:
        dict: {
            "tomorrow": float,
            "day_after_tomorrow": float
        }
    """
    # Predict tomorrow (horizon=1)
    forecast_tomorrow = fitted_model.forecast(steps=1)
    price_tomorrow = float(forecast_tomorrow.iloc[0])
    
    # Predict day after tomorrow (horizon=2)
    forecast_day2 = fitted_model.forecast(steps=2)
    price_day2 = float(forecast_day2.iloc[1])
    
    return {
        "tomorrow": price_tomorrow,
        "day_after_tomorrow": price_day2
    }


def save_predictions_to_s3(predictions, bucket=S3_BUCKET, key=S3_PREDICTIONS_KEY):
    """Save predictions as JSON to S3."""
    s3_client = boto3.client('s3')
    
    # Create JSON payload
    payload = {
        "as_of": datetime.utcnow().isoformat() + "Z",
        "model_name": MODEL_NAME,
        "model_type": "ARIMA",
        "predictions": predictions,
        "note": "Predictions generated daily. Model is retrained separately."
    }
    
    # Convert to JSON string
    json_str = json.dumps(payload, indent=2)
    
    # Upload to S3
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json_str.encode('utf-8'),
        ContentType='application/json'
    )
    
    print(f"✅ Predictions saved to S3: s3://{bucket}/{key}")
    print(f"   Tomorrow: ${predictions['tomorrow']:.2f}/oz")
    print(f"   Day after tomorrow: ${predictions['day_after_tomorrow']:.2f}/oz")


def main():
    parser = argparse.ArgumentParser(description="Generate ARIMA predictions and save to S3")
    parser.add_argument("--model-bucket", type=str, default=S3_BUCKET,
                       help="S3 bucket for model")
    parser.add_argument("--model-key", type=str, default=S3_MODEL_KEY,
                       help="S3 key for model pickle file")
    parser.add_argument("--predictions-bucket", type=str, default=S3_BUCKET,
                       help="S3 bucket for predictions")
    parser.add_argument("--predictions-key", type=str, default=S3_PREDICTIONS_KEY,
                       help="S3 key for predictions JSON")
    parser.add_argument("--use-local", action="store_true",
                       help="Load model from local MLflow instead of S3")
    
    args = parser.parse_args()
    
    print("🚀 Generating ARIMA predictions...")
    print(f"   S3 destination: s3://{args.predictions_bucket}/{args.predictions_key}")
    
    # Load model (try S3 first, then local as fallback)
    print("\n📥 Loading model...")
    try:
        if args.use_local:
            fitted_model = load_model_from_local()
        else:
            fitted_model = load_model_from_s3(bucket=args.model_bucket, key=args.model_key)
    except Exception as e:
        print(f"\n⚠️  Failed to load from S3, trying local MLflow...")
        try:
            fitted_model = load_model_from_local()
        except Exception as e2:
            print(f"❌ Failed to load model from both S3 and local: {e2}")
            raise
    
    # Generate predictions
    print("\n🔮 Generating predictions...")
    predictions = generate_predictions(fitted_model)
    
    # Save to S3
    print("\n💾 Saving predictions to S3...")
    save_predictions_to_s3(predictions, bucket=args.predictions_bucket, key=args.predictions_key)
    
    print("\n✅ Done!")
    print(f"\n📝 Next steps:")
    print(f"   1. Lambda function should read from: s3://{args.predictions_bucket}/{args.predictions_key}")
    print(f"   2. Set Lambda environment variable: PREDICTIONS_S3_URI=s3://{args.predictions_bucket}/{args.predictions_key}")


if __name__ == "__main__":
    main()