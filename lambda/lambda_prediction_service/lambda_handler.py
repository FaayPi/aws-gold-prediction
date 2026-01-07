"""
Lambda handler for gold price prediction service using Flask and MLflow.
"""

import json
import os
import sys
from datetime import datetime
from io import BytesIO

import boto3
import mlflow
import pandas as pd
from mangum import Mangum
from flask import Flask, jsonify, request

# Initialize Flask app
app = Flask(__name__)

# Initialize S3 client
s3_client = boto3.client('s3')

# Environment variables
BUCKET_NAME = os.environ.get('BUCKET_NAME')
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://your-mlflow-server:5000')
MODEL_NAME = os.environ.get('MODEL_NAME', 'xgboost_gold_price_production')
MODEL_STAGE = os.environ.get('MODEL_STAGE', 'Production')
S3_DATA_PREFIX = os.environ.get('S3_DATA_PREFIX', 'processed/daily/')

# Global model variable (loaded once per Lambda container)
model = None


def load_model():
    """Load MLflow model (cached per Lambda container)."""
    global model
    if model is None:
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            # Load model from registry
            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"✅ Model loaded: {MODEL_NAME} (Stage: {MODEL_STAGE})")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    return model


def get_latest_gold_price_from_s3():
    """
    Fetch the latest gold price CSV from S3.
    Returns pandas DataFrame with historical data.
    """
    try:
        # List objects in S3 to find the latest file
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=S3_DATA_PREFIX
        )
        
        if 'Contents' not in response:
            raise ValueError(f"No files found in s3://{BUCKET_NAME}/{S3_DATA_PREFIX}")
        
        # Get the most recent file
        objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
        latest_key = objects[0]['Key']
        
        print(f"📥 Reading latest data from: s3://{BUCKET_NAME}/{latest_key}")
        
        # Read CSV from S3
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=latest_key)
        df = pd.read_csv(BytesIO(obj['Body'].read()))
        
        # Set datetime index if available
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        
        df = df.sort_index()
        
        return df
    
    except Exception as e:
        print(f"❌ Error reading from S3: {e}")
        raise


def create_features_for_prediction(df):
    """
    Create features from historical data for prediction.
    This should match your feature engineering pipeline.
    """
    # Import your feature engineering module
    # Note: You'll need to include this in your Lambda package
    try:
        from feature_engineering_optimized import create_optimized_features
        df_features = create_optimized_features(df)
        return df_features.iloc[[-1]].copy()  # Return last row
    except ImportError:
        # Fallback: basic feature engineering
        # Adjust based on your actual feature requirements
        df_features = df.copy()
        if 'close' in df_features.columns:
            df_features['price_change'] = df_features['close'].diff()
            df_features['price_change_pct'] = df_features['close'].pct_change()
            df_features['rolling_mean_7'] = df_features['close'].rolling(7).mean()
            df_features['rolling_std_7'] = df_features['close'].rolling(7).std()
        return df_features.iloc[[-1]].copy()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model_name': MODEL_NAME,
        'model_stage': MODEL_STAGE
    })


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Main prediction endpoint.
    Returns predictions for tomorrow and next week.
    """
    try:
        # Load model
        model = load_model()
        
        # Get latest data from S3
        historical_data = get_latest_gold_price_from_s3()
        
        # Create features
        features = create_features_for_prediction(historical_data)
        
        # Predict tomorrow (horizon=1)
        features_tomorrow = features.copy()
        features_tomorrow['horizon'] = 1
        prediction_tomorrow = model.predict(features_tomorrow)
        price_tomorrow = float(prediction_tomorrow.iloc[0]['predicted_price']) if hasattr(prediction_tomorrow, 'iloc') else float(prediction_tomorrow[0])
        
        # Predict next week (horizon=5)
        features_week = features.copy()
        features_week['horizon'] = 5
        prediction_week = model.predict(features_week)
        price_week = float(prediction_week.iloc[0]['predicted_price']) if hasattr(prediction_week, 'iloc') else float(prediction_week[0])
        
        # Get current price
        current_price = float(historical_data['close'].iloc[-1]) if 'close' in historical_data.columns else None
        
        return jsonify({
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'current_price': current_price,
            'predictions': {
                'tomorrow': {
                    'price': price_tomorrow,
                    'change': price_tomorrow - current_price if current_price else None,
                    'change_pct': ((price_tomorrow - current_price) / current_price * 100) if current_price else None
                },
                'next_week': {
                    'price': price_week,
                    'change': price_week - current_price if current_price else None,
                    'change_pct': ((price_week - current_price) / current_price * 100) if current_price else None
                }
            },
            'data_date': historical_data.index[-1].isoformat() if hasattr(historical_data.index[-1], 'isoformat') else str(historical_data.index[-1])
        })
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information."""
    return jsonify({
        'service': 'Gold Price Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/predict': 'GET/POST - Get predictions for tomorrow and next week',
            '/': 'GET - API information'
        }
    })


# Create Mangum handler for Lambda
handler = Mangum(app)


def lambda_handler(event, context):
    """AWS Lambda handler entry point."""
    return handler(event, context)