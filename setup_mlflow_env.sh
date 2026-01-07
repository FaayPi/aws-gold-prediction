#!/bin/bash
# Setup script for MLflow with AWS S3 credentials
# This script exports AWS credentials for MLflow to use S3 as artifact store

# Use user1 profile (which has valid credentials)
export AWS_PROFILE=user1

# Load AWS credentials from AWS CLI configuration (user1 profile)
export AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile user1)
export AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile user1)
export AWS_DEFAULT_REGION=eu-central-1
export AWS_REGION=eu-central-1

# Set MLflow tracking URI (if not already set)
if [ -z "$MLFLOW_TRACKING_URI" ]; then
    export MLFLOW_TRACKING_URI="http://localhost:5002"
fi

echo "✅ AWS Credentials loaded from user1 profile"
echo "   Region: $AWS_DEFAULT_REGION"
echo "   MLflow URI: $MLFLOW_TRACKING_URI"
echo ""
echo "You can now run MLflow training scripts."
echo "Example:"
echo "  python pipelines/mlflow_arima/train_arima_mlflow.py --data-file gold_GCF_10y_1d.csv"

