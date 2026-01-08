"""
Lambda handler for gold price prediction service using Flask.
Simplified version: Reads pre-computed predictions from S3 JSON file.

The predictions are generated daily by a separate script (save_predictions_to_s3.py)
and stored as a simple JSON file in S3. This Lambda function only reads and serves
those predictions - no model loading, no NumPy/Statsmodels dependencies needed!
"""

import json
import os
from datetime import datetime

import boto3
from flask import Flask, jsonify, request

# Initialize Flask app
app = Flask(__name__)

# Initialize S3 client
s3_client = boto3.client('s3')

# Environment variables
PREDICTIONS_S3_URI = os.environ.get('PREDICTIONS_S3_URI')  # e.g. s3://bucket/predictions/latest.json
MODEL_NAME = os.environ.get('MODEL_NAME', 'arima_gold_price_production')  # For logging only

# Cache for predictions (loaded once per Lambda container)
cached_predictions = None
cache_timestamp = None


def load_predictions_from_s3():
    """
    Load pre-computed predictions from S3 JSON file.
    Returns the full predictions payload.
    """
    global cached_predictions, cache_timestamp
    
    # Return cached predictions if available (within same Lambda container)
    if cached_predictions is not None:
        return cached_predictions
    
    try:
        if not PREDICTIONS_S3_URI:
            raise ValueError(
                "PREDICTIONS_S3_URI environment variable not set. "
                "Please set it to the S3 path of your predictions JSON file "
                "(e.g., s3://bucket/predictions/latest.json)"
            )
        
        # Parse S3 URI
        if not PREDICTIONS_S3_URI.startswith('s3://'):
            raise ValueError(f"PREDICTIONS_S3_URI must start with 's3://', got: {PREDICTIONS_S3_URI}")
        
        parts = PREDICTIONS_S3_URI.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        if not key:
            raise ValueError(f"Invalid S3 URI format: {PREDICTIONS_S3_URI}")
        
        print(f"📥 Loading predictions from: s3://{bucket}/{key}")
        
        # Load JSON from S3
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        predictions_data = json.loads(obj['Body'].read().decode('utf-8'))
        
        print(f"✅ Predictions loaded successfully from S3")
        print(f"   Generated at: {predictions_data.get('as_of', 'N/A')}")
        print(f"   Tomorrow: ${predictions_data['predictions']['tomorrow']:.2f}/oz")
        print(f"   Day after tomorrow: ${predictions_data['predictions']['day_after_tomorrow']:.2f}/oz")
        
        # Cache predictions
        cached_predictions = predictions_data
        cache_timestamp = datetime.utcnow()
        
        return predictions_data
        
    except Exception as e:
        print(f"❌ Error loading predictions from S3: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model_name': MODEL_NAME,
        'model_type': 'ARIMA',
        'predictions_source': 'S3 (JSON)',
        'predictions_uri': PREDICTIONS_S3_URI if PREDICTIONS_S3_URI else 'Not configured'
    })


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Main prediction endpoint.
    Returns pre-computed predictions for tomorrow and day after tomorrow.
    
    Process:
    1. Load predictions from S3 JSON file (generated daily by separate script)
    2. Return predictions in API format
    
    Note: Predictions are generated daily by save_predictions_to_s3.py script.
    This Lambda function only reads and serves those predictions.
    """
    try:
        # Load predictions from S3
        predictions_data = load_predictions_from_s3()
        
        # Extract predictions
        predictions = predictions_data['predictions']
        
        return jsonify({
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'generated_at': predictions_data.get('as_of'),
            'model_type': predictions_data.get('model_type', 'ARIMA'),
            'model_name': predictions_data.get('model_name', MODEL_NAME),
            'predictions': {
                'tomorrow': {
                    'price': predictions['tomorrow'],
                    'horizon_days': 1
                },
                'day_after_tomorrow': {
                    'price': predictions['day_after_tomorrow'],
                    'horizon_days': 2
                }
            },
            'note': predictions_data.get('note', 'Predictions generated daily. Model is retrained separately.')
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
        'version': '2.0.0',
        'description': 'Simplified API that reads pre-computed predictions from S3',
        'endpoints': {
            '/health': 'GET - Health check',
            '/predict': 'GET/POST - Get predictions for tomorrow and day after tomorrow',
            '/': 'GET - API information'
        },
        'note': 'Predictions are generated daily by a separate script and stored in S3 as JSON.'
    })


def lambda_handler(event, context):
    """AWS Lambda Handler Entry Point - Direct WSGI Implementation"""
    
    # Konvertiere API Gateway Event zu WSGI Environment
    environ = {
        'REQUEST_METHOD': event.get('httpMethod', 'GET'),
        'SCRIPT_NAME': '',
        'PATH_INFO': event.get('path', '/'),
        'QUERY_STRING': event.get('queryStringParameters', '') or '',
        'CONTENT_TYPE': event.get('headers', {}).get('Content-Type', ''),
        'CONTENT_LENGTH': str(len(event.get('body', '') or '')),
        'SERVER_NAME': 'localhost',
        'SERVER_PORT': '80',
        'wsgi.version': (1, 0),
        'wsgi.url_scheme': 'https',
        'wsgi.input': None,
        'wsgi.errors': None,
        'wsgi.multithread': False,
        'wsgi.multiprocess': False,
        'wsgi.run_once': False,
    }
    
    # Füge Headers hinzu
    for key, value in event.get('headers', {}).items():
        key = key.upper().replace('-', '_')
        if key not in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
            key = 'HTTP_' + key
        environ[key] = value
    
    # WSGI Response
    response = {'status': 200, 'headers': [], 'body': b''}
    
    def start_response(status, response_headers):
        response['status'] = int(status.split()[0])
        response['headers'] = response_headers
    
    # Rufe Flask-App auf
    body_iter = app(environ, start_response)
    body = b''.join(body_iter)
    
    # Konvertiere zu API Gateway Response
    headers = {k: v for k, v in response['headers']}
    
    return {
        'statusCode': response['status'],
        'headers': headers,
        'body': body.decode('utf-8'),
        'isBase64Encoded': False
    }
