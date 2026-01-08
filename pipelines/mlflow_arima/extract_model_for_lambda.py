"""
Extract ARIMA model from MLflow and save as simple pickle for Lambda.
This script loads the pickle file directly from MLflow artifacts,
avoiding Python version incompatibility issues.
"""

import pickle
import boto3
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import tempfile

MODEL_NAME = "arima_gold_price_production"
MODEL_VERSION = "2"  # Explizit Version 2 verwenden
S3_BUCKET = "gold-ml-mlflow-artifacts"
S3_KEY = "models/arima_model.pkl"  # Einfaches pickle für Lambda


def extract_and_save_model():
    """Extract ARIMA model from MLflow artifacts and save as simple pickle."""
    
    print("📥 Loading MLflow model version from registry...")
    
    client = MlflowClient()
    
    # Get model version info
    model_version = client.get_model_version(MODEL_NAME, MODEL_VERSION)
    run_id = model_version.run_id
    print(f"   Run ID: {run_id}")
    print(f"   Model Version: {MODEL_VERSION}")
    
    # Download the model artifact directly (pickle file)
    print("🔍 Downloading model pickle from MLflow artifacts...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download the entire model directory
        model_uri = f"runs:/{run_id}/model"
        local_model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=tmpdir
        )
        
        print(f"✅ Model artifacts downloaded to: {local_model_path}")
        
        # The pickle file should be in the artifacts subdirectory
        # Based on training script, it's saved as "model_path" in artifacts
        artifacts_path = Path(local_model_path) / "artifacts"
        
        # Look for the pickle file
        pickle_files = list(artifacts_path.glob("**/*.pkl"))
        if not pickle_files:
            # Try looking in the model directory itself
            pickle_files = list(Path(local_model_path).glob("**/*.pkl"))
        
        if not pickle_files:
            raise FileNotFoundError(f"Could not find pickle file in {local_model_path}")
        
        # Use the first pickle file found (should be the model)
        model_pickle_path = pickle_files[0]
        print(f"   Found pickle file: {model_pickle_path}")
        
        # Load the pickle file directly
        print("📦 Loading model from pickle file...")
        with open(model_pickle_path, "rb") as f:
            model_data = pickle.load(f)
        
        print(f"✅ Model loaded from pickle")
        print(f"   Keys in model_data: {list(model_data.keys())}")
        
        # Extract the fitted model
        if "fitted_model" not in model_data:
            raise ValueError("Model pickle does not contain 'fitted_model' key")
        
        fitted_model = model_data["fitted_model"]
        order = model_data.get("order", "N/A")
        series = model_data.get("final_series")
        
        print(f"✅ ARIMA model extracted:")
        print(f"   Order: {order}")
        print(f"   Model type: {type(fitted_model)}")
        print(f"   Series length: {len(series) if series is not None else 'N/A'}")
        
        # Create a simple model_data dict for Lambda (only what we need)
        lambda_model_data = {
            "fitted_model": fitted_model,
            "order": order,
            "final_series": series,
        }
        
        # Save locally first
        local_path = Path("arima_model_lambda.pkl")
        print(f"\n💾 Saving model to {local_path}...")
        with open(local_path, "wb") as f:
            pickle.dump(lambda_model_data, f)
    
    print(f"✅ Model saved locally: {local_path}")
    print(f"   File size: {local_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Upload to S3
    print(f"\n📤 Uploading to S3: s3://{S3_BUCKET}/{S3_KEY}...")
    s3_client = boto3.client('s3')
    s3_client.upload_file(str(local_path), S3_BUCKET, S3_KEY)
    
    print(f"✅ Model uploaded to S3!")
    print(f"   S3 URI: s3://{S3_BUCKET}/{S3_KEY}")
    print(f"\n📝 Next steps:")
    print(f"   1. Set Lambda environment variable:")
    print(f"      S3_MODEL_URI=s3://{S3_BUCKET}/{S3_KEY}")
    print(f"   2. Test Lambda function")


if __name__ == "__main__":
    extract_and_save_model()

