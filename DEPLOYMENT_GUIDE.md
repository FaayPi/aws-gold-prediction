# Step-by-Step Guide: Deploying MLflow Gold Price Prediction Model with AWS Lambda + API Gateway + Flask

## Overview

This guide walks you through deploying your MLflow-staged gold price prediction model to production using:
- **AWS Lambda** for serverless compute
- **API Gateway** for REST API endpoints
- **Flask** (via Mangum adapter) for web framework compatibility
- **S3** for reading daily gold price data
- **MLflow** for model loading

The system will:
1. Read current gold price from S3 bucket
2. Load the production MLflow model
3. Predict gold prices for tomorrow and next week
4. Serve predictions via a Flask web application accessible through API Gateway

---

## Prerequisites

- AWS Account with appropriate permissions
- MLflow model registered and staged in "Production"
- S3 bucket with daily gold price data
- AWS CLI configured locally
- Python 3.9+ environment

---

## Step 1: Prepare Lambda Deployment Package

### 1.1 Create Lambda Function Directory Structure

Create a new directory for your Lambda function:

```bash
mkdir -p lambda_prediction_service
cd lambda_prediction_service
```

### 1.2 Create Lambda Handler with Flask

Create `lambda_handler.py`:

```python
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
```

### 1.3 Create Requirements File

Create `requirements.txt` for Lambda:

```txt
flask==3.0.0
mangum==0.17.0
mlflow>=3.8.1
boto3>=1.42.0
pandas>=2.2.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
```

### 1.4 Copy Feature Engineering Module

Copy your feature engineering module to the Lambda directory:

```bash
cp ../pipelines/feature_engineering_optimized.py .
```

---

## Step 2: Build Lambda Deployment Package

### 2.1 Install Dependencies in a Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt -t .

# Note: Some packages may need to be installed separately due to size
# You might need to use Lambda layers for large dependencies like MLflow
```

### 2.2 Create Deployment Package

```bash
# Remove unnecessary files
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type d -name "*.dist-info" -exec rm -r {} +
find . -type d -name "*.egg-info" -exec rm -r {} +

# Create zip file (excluding venv)
zip -r lambda_prediction_service.zip . -x "venv/*" "*.pyc" "__pycache__/*" "*.git*"
```

**Note:** If the package exceeds 50MB, consider using Lambda Layers for dependencies.

---

## Step 3: Create Lambda Layer (Optional but Recommended)

For large dependencies like MLflow, create a Lambda Layer:

### 3.1 Create Layer Directory

```bash
mkdir -p lambda_layer/python
cd lambda_layer/python
```

### 3.2 Install Layer Dependencies

```bash
pip install mlflow pandas numpy scikit-learn xgboost -t .
```

### 3.3 Package Layer

```bash
cd ..
zip -r mlflow_layer.zip python/
```

### 3.4 Upload Layer to AWS

```bash
aws lambda publish-layer-version \
    --layer-name mlflow-gold-prediction-layer \
    --zip-file fileb://mlflow_layer.zip \
    --compatible-runtimes python3.9 python3.10 python3.11
```

Note the Layer ARN from the output.

---

## Step 4: Create Lambda Function

### 4.1 Create IAM Role for Lambda

Create `lambda-role-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::YOUR-BUCKET-NAME/*",
        "arn:aws:s3:::YOUR-BUCKET-NAME"
      ]
    }
  ]
}
```

Create the role:

```bash
aws iam create-role \
    --role-name lambda-gold-prediction-role \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }'

aws iam put-role-policy \
    --role-name lambda-gold-prediction-role \
    --policy-name lambda-s3-policy \
    --policy-document file://lambda-role-policy.json
```

Get the Role ARN:

```bash
aws iam get-role --role-name lambda-gold-prediction-role --query 'Role.Arn' --output text
```

### 4.2 Create Lambda Function

```bash
aws lambda create-function \
    --function-name gold-price-prediction-api \
    --runtime python3.11 \
    --role arn:aws:iam::YOUR-ACCOUNT-ID:role/lambda-gold-prediction-role \
    --handler lambda_handler.lambda_handler \
    --zip-file fileb://lambda_prediction_service.zip \
    --timeout 60 \
    --memory-size 512 \
    --environment Variables="{
        BUCKET_NAME=your-s3-bucket-name,
        MLFLOW_TRACKING_URI=http://your-mlflow-server:5000,
        MODEL_NAME=xgboost_gold_price_production,
        MODEL_STAGE=Production,
        S3_DATA_PREFIX=processed/daily/
    }"
```

### 4.3 Add Layer (if created)

```bash
aws lambda update-function-configuration \
    --function-name gold-price-prediction-api \
    --layers arn:aws:lambda:REGION:ACCOUNT-ID:layer:mlflow-gold-prediction-layer:1
```

---

## Step 5: Set Up API Gateway

### 5.1 Create REST API

```bash
aws apigateway create-rest-api \
    --name gold-price-prediction-api \
    --description "API for gold price predictions"
```

Note the API ID from the output.

### 5.2 Get Root Resource ID

```bash
API_ID="your-api-id"
aws apigateway get-resources --rest-api-id $API_ID
```

Note the root resource ID (usually starts with the API ID).

### 5.3 Create Resources and Methods

```bash
ROOT_RESOURCE_ID="your-root-resource-id"
REGION="your-aws-region"
ACCOUNT_ID="your-account-id"

# Create /predict resource
aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $ROOT_RESOURCE_ID \
    --path-part predict

# Get the /predict resource ID
PREDICT_RESOURCE_ID=$(aws apigateway get-resources \
    --rest-api-id $API_ID \
    --query "items[?path=='/predict'].id" \
    --output text)

# Create GET method
aws apigateway put-method \
    --rest-api-id $API_ID \
    --resource-id $PREDICT_RESOURCE_ID \
    --http-method GET \
    --authorization-type NONE

# Create POST method
aws apigateway put-method \
    --rest-api-id $API_ID \
    --resource-id $PREDICT_RESOURCE_ID \
    --http-method POST \
    --authorization-type NONE

# Set up Lambda integration
LAMBDA_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:gold-price-prediction-api"

aws apigateway put-integration \
    --rest-api-id $API_ID \
    --resource-id $PREDICT_RESOURCE_ID \
    --http-method GET \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri "arn:aws:apigateway:${REGION}:lambda:path/2015-03-31/functions/${LAMBDA_ARN}/invocations"

aws apigateway put-integration \
    --rest-api-id $API_ID \
    --resource-id $PREDICT_RESOURCE_ID \
    --http-method POST \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri "arn:aws:apigateway:${REGION}:lambda:path/2015-03-31/functions/${LAMBDA_ARN}/invocations"
```

### 5.4 Grant API Gateway Permission to Invoke Lambda

```bash
aws lambda add-permission \
    --function-name gold-price-prediction-api \
    --statement-id apigateway-invoke \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*/*"
```

### 5.5 Deploy API

```bash
# Create deployment
aws apigateway create-deployment \
    --rest-api-id $API_ID \
    --stage-name prod

# Your API URL will be:
# https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod/predict
```

---

## Step 6: Create Simple Flask Frontend (Optional)

If you want a web interface, create a simple HTML page that calls your API:

### 6.1 Create `index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gold Price Predictions</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            padding: 30px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        h1 {
            text-align: center;
        }
        .prediction-card {
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
        }
        .price {
            font-size: 2em;
            font-weight: bold;
            color: #ffd700;
        }
        .change {
            font-size: 1.2em;
        }
        .positive { color: #4ade80; }
        .negative { color: #f87171; }
        button {
            background: #4ade80;
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
        }
        button:hover {
            background: #22c55e;
        }
        .loading {
            text-align: center;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>💰 Gold Price Predictions</h1>
        <div id="current-price" class="prediction-card">
            <h2>Current Price</h2>
            <div class="price" id="current-price-value">Loading...</div>
        </div>
        
        <div id="tomorrow" class="prediction-card">
            <h2>Tomorrow's Prediction</h2>
            <div class="price" id="tomorrow-price">-</div>
            <div class="change" id="tomorrow-change">-</div>
        </div>
        
        <div id="next-week" class="prediction-card">
            <h2>Next Week's Prediction</h2>
            <div class="price" id="week-price">-</div>
            <div class="change" id="week-change">-</div>
        </div>
        
        <button onclick="fetchPredictions()">Refresh Predictions</button>
        <div id="last-updated" style="text-align: center; margin-top: 20px; opacity: 0.7;"></div>
    </div>

    <script>
        const API_URL = 'YOUR_API_GATEWAY_URL'; // Replace with your API Gateway URL
        
        function formatPrice(price) {
            return `$${price.toFixed(2)}/oz`;
        }
        
        function formatChange(change, changePct) {
            const sign = change >= 0 ? '+' : '';
            return `${sign}$${change.toFixed(2)} (${sign}${changePct.toFixed(2)}%)`;
        }
        
        async function fetchPredictions() {
            const loadingEl = document.getElementById('current-price-value');
            loadingEl.textContent = 'Loading...';
            
            try {
                const response = await fetch(`${API_URL}/predict`);
                const data = await response.json();
                
                if (data.success) {
                    // Current price
                    document.getElementById('current-price-value').textContent = 
                        formatPrice(data.current_price);
                    
                    // Tomorrow prediction
                    const tomorrowPrice = data.predictions.tomorrow.price;
                    const tomorrowChange = data.predictions.tomorrow.change;
                    const tomorrowChangePct = data.predictions.tomorrow.change_pct;
                    
                    document.getElementById('tomorrow-price').textContent = formatPrice(tomorrowPrice);
                    const tomorrowChangeEl = document.getElementById('tomorrow-change');
                    tomorrowChangeEl.textContent = formatChange(tomorrowChange, tomorrowChangePct);
                    tomorrowChangeEl.className = `change ${tomorrowChange >= 0 ? 'positive' : 'negative'}`;
                    
                    // Next week prediction
                    const weekPrice = data.predictions.next_week.price;
                    const weekChange = data.predictions.next_week.change;
                    const weekChangePct = data.predictions.next_week.change_pct;
                    
                    document.getElementById('week-price').textContent = formatPrice(weekPrice);
                    const weekChangeEl = document.getElementById('week-change');
                    weekChangeEl.textContent = formatChange(weekChange, weekChangePct);
                    weekChangeEl.className = `change ${weekChange >= 0 ? 'positive' : 'negative'}`;
                    
                    // Last updated
                    document.getElementById('last-updated').textContent = 
                        `Last updated: ${new Date(data.timestamp).toLocaleString()}`;
                } else {
                    throw new Error(data.error || 'Unknown error');
                }
            } catch (error) {
                console.error('Error fetching predictions:', error);
                document.getElementById('current-price-value').textContent = 'Error loading data';
            }
        }
        
        // Load predictions on page load
        fetchPredictions();
        
        // Auto-refresh every 5 minutes
        setInterval(fetchPredictions, 5 * 60 * 1000);
    </script>
</body>
</html>
```

### 6.2 Host Frontend

You can host this HTML file on:
- **S3 Static Website Hosting**
- **AWS Amplify**
- **CloudFront + S3**
- Any static hosting service

---

## Step 7: Testing

### 7.1 Test Lambda Function Directly

```bash
aws lambda invoke \
    --function-name gold-price-prediction-api \
    --payload '{"httpMethod": "GET", "path": "/predict"}' \
    response.json

cat response.json
```

### 7.2 Test API Gateway Endpoint

```bash
# Replace with your API Gateway URL
curl https://YOUR-API-ID.execute-api.REGION.amazonaws.com/prod/predict
```

### 7.3 Expected Response

```json
{
  "success": true,
  "timestamp": "2025-01-15T10:30:00.000000",
  "current_price": 2050.50,
  "predictions": {
    "tomorrow": {
      "price": 2055.75,
      "change": 5.25,
      "change_pct": 0.26
    },
    "next_week": {
      "price": 2062.30,
      "change": 11.80,
      "change_pct": 0.58
    }
  },
  "data_date": "2025-01-14T00:00:00"
}
```

---

## Step 8: Monitoring and Troubleshooting

### 8.1 View Lambda Logs

```bash
aws logs tail /aws/lambda/gold-price-prediction-api --follow
```

### 8.2 Common Issues and Solutions

**Issue: Model loading timeout**
- **Solution:** Increase Lambda timeout (max 15 minutes) and memory
- **Solution:** Use Lambda Layers to reduce cold start time

**Issue: MLflow connection error**
- **Solution:** Ensure MLflow tracking URI is accessible from Lambda (use public IP or VPC configuration)
- **Solution:** Consider using MLflow model artifacts stored in S3 directly

**Issue: Package too large**
- **Solution:** Use Lambda Layers for dependencies
- **Solution:** Optimize dependencies (remove unused packages)

**Issue: S3 access denied**
- **Solution:** Check IAM role permissions
- **Solution:** Verify bucket name and prefix in environment variables

---

## Step 9: Optimization Tips

### 9.1 Reduce Cold Starts

- Use Lambda Provisioned Concurrency
- Optimize package size
- Use Lambda Layers for common dependencies

### 9.2 Improve Performance

- Cache model in global variable (already implemented)
- Use connection pooling for S3
- Consider caching predictions with TTL

### 9.3 Cost Optimization

- Set up CloudWatch alarms for costs
- Use appropriate memory allocation
- Consider reserved concurrency limits

---

## Step 10: Security Best Practices

1. **Enable API Key Authentication** (optional but recommended)
2. **Use VPC** if MLflow is in private network
3. **Enable CloudWatch Logs Encryption**
4. **Use IAM roles** with least privilege
5. **Enable API Gateway throttling** to prevent abuse
6. **Use HTTPS only** (enabled by default with API Gateway)

---

## Next Steps

1. Set up **CloudWatch alarms** for monitoring
2. Implement **API rate limiting**
3. Add **authentication/authorization** if needed
4. Set up **CI/CD pipeline** for deployments
5. Create **automated tests** for the API
6. Set up **custom domain** for API Gateway

---

## Additional Resources

- [AWS Lambda Python Documentation](https://docs.aws.amazon.com/lambda/latest/dg/lambda-python.html)
- [API Gateway REST API](https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-rest-api.html)
- [Mangum Documentation](https://mangum.io/)
- [MLflow Model Serving](https://www.mlflow.org/docs/latest/models.html)

---

## Summary

You now have:
✅ Lambda function with Flask serving MLflow predictions  
✅ API Gateway REST API endpoint  
✅ Integration with S3 for daily gold price data  
✅ Predictions for tomorrow and next week  
✅ Optional web frontend  

Your API is accessible at:
`https://YOUR-API-ID.execute-api.REGION.amazonaws.com/prod/predict`

