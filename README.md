# Gold Price Prediction with MLOps on AWS

**Live Application**: [http://35.158.241.11:8080](http://35.158.241.11:8080)

## Overview

This project implements an end-to-end machine learning solution for predicting Gold prices using MLOps best practices and AWS cloud services. Following the project brief, I've built a production-ready system that leverages state-of-the-art ML technologies, cloud deployment, and operational excellence.

Gold has been a globally recognized store of value for centuries, and predicting its price fluctuations is crucial for traders, investors, and financial institutions. This project addresses the business need for accurate daily price forecasts while demonstrating modern MLOps workflows.

## Architecture Overview

The system follows a serverless architecture pattern, separating concerns between frontend, backend, and data storage:

```
┌─────────────┐
│     User    │
│  (Browser)  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐         ┌──────────────┐         ┌─────────────┐
│   AWS Lightsail │────────▶│ API Gateway  │────────▶│   Lambda    │
│  (Flask Web App)│         │   (HTTP API) │         │ (Flask API) │
│  Port 8080      │         └──────────────┘         └──────┬──────┘
└─────────────────┘                                         │
                                                            ▼
                                                     ┌─────────────┐
                                                     │     S3      │
                                                     │(Predictions)│
                                                     └─────────────┘
```

## Infrastructure & Setup

### Local Development Environment

The following components were developed and tested locally:

- **Data Collection**: `pipelines/fetch_gold.py` - Downloads historical Gold Futures (GC=F) data from Yahoo Finance using `yfinance`
- **Model Training**: `pipelines/mlflow_arima/train_arima_mlflow.py` - Trains ARIMA(3,1,3) models with rolling window evaluation
- **MLflow Tracking**: Local MLflow server tracks experiments, parameters, metrics, and model versions
- **Model Evaluation**: Rolling forecast methodology evaluates next-day and next-week predictions (RMSE, MAE, MAPE)
- **Prediction Generation**: `pipelines/mlflow_arima/save_predictions_to_s3.py` - Generates daily forecasts and uploads to S3

### AWS Cloud Infrastructure

The production system runs entirely on AWS, orchestrated as follows:

#### **AWS Lambda + API Gateway** (Backend - Serverless)
- **Lambda Function**: Hosts a Flask API (`lambda/flask_deployment/lambda_handler.py`) that serves predictions
- **Architecture**: Simplified design where Lambda reads pre-computed predictions from S3 JSON files (no model loading in Lambda)
- **API Gateway**: Provides RESTful HTTP endpoint for the Lambda function
- **Benefits**: Pay-per-use, automatic scaling, zero server management

#### **AWS Lightsail** (Frontend)
- **Web Application**: Flask web app (`web_app/app.py`) displays predictions in a user-friendly interface
- **Deployment**: Runs on a Lightsail instance at IP `35.158.241.11` on port 8080
- **Why Lightsail**: Simplified VPS with fixed pricing, perfect for hosting the frontend web application

#### **Amazon S3** (Storage)
- **Predictions Storage**: Daily predictions stored as JSON files in S3 bucket
- **Model Artifacts**: Trained models can be stored in S3 for version control
- **Data Source**: Historical gold price data stored as CSV files

### Infrastructure Orchestration

The AWS cloud orchestration follows this workflow:

1. **Daily Prediction Generation**: 
   - A scheduled script (can be triggered via EventBridge/Lambda or cron) runs `save_predictions_to_s3.py`
   - Loads the latest ARIMA model from S3 or MLflow
   - Generates predictions for tomorrow and day after tomorrow
   - Saves predictions as JSON to S3 (`s3://bucket/predictions/latest.json`)

2. **Model Serving**:
   - User accesses web app on Lightsail
   - Web app makes HTTP request to API Gateway endpoint
   - API Gateway invokes Lambda function
   - Lambda reads predictions from S3 and returns JSON response
   - Web app renders predictions to user

3. **Model Retraining** (Separate Workflow):
   - Models are retrained locally or on EC2/SageMaker
   - MLflow tracks experiments and registers best models
   - Production models are extracted and uploaded to S3
   - Model promotion workflow ensures only validated models reach production

## Key Technologies

- **ML Frameworks**: ARIMA (statsmodels), XGBoost, LightGBM, scikit-learn
- **MLOps**: MLflow for experiment tracking and model registry
- **Cloud Services**: AWS Lambda, API Gateway, Lightsail, S3
- **Web Framework**: Flask (both for Lambda API and Lightsail web app)
- **Data Processing**: pandas, numpy, yfinance
- **Deployment**: Serverless architecture with Lambda, containerization-ready

## Project Structure

The repository is organized into logical modules following MLOps best practices:

```
├── pipelines/         # Training and data pipelines (ARIMA, XGBoost workflows)
├── experiments/       # Experimental scripts and model comparison
├── lambda/            # AWS Lambda deployment code
├── web_app/           # Flask web application (frontend)
├── models/            # Local model storage (trained models)
├── mlruns/            # MLflow tracking data (experiments, runs)
└── requirements.txt   # Project dependencies
```

## Model Comparison

Four different machine learning models were trained and evaluated on the same test dataset using a rolling window methodology with 3 years of historical data (80% train, 20% test):

### Models Evaluated

1. **ARIMA (3,1,3)** - Time series model, univariate approach
2. **XGBoost** - Gradient boosting ensemble with engineered features
3. **Random Forest** - Bagging ensemble, robust to overfitting
4. **LightGBM** - Fast gradient boosting, efficient for large datasets

### Performance Results

#### Next Day Predictions (+1 day)

| Model | RMSE (USD) | MAE (USD) | MAPE (%) | RMSE (%) |
|-------|------------|-----------|----------|----------|
| **ARIMA** | **$50.32** | **$36.07** | **0.98%** | **1.39%** |
| Random Forest | $375.28 | $254.85 | 6.66% | 10.61% |
| LightGBM | $392.41 | $297.27 | 7.91% | 11.09% |
| XGBoost | $395.71 | $309.88 | 8.30% | 11.18% |

#### Next Week Predictions (+5 trading days)

| Model | RMSE (USD) | MAE (USD) | MAPE (%) | RMSE (%) |
|-------|------------|-----------|----------|----------|
| **ARIMA** | **$100.61** | **$77.18** | **2.07%** | **2.75%** |
| LightGBM | $379.93 | $249.74 | 6.49% | 10.74% |
| Random Forest | $403.89 | $288.49 | 7.59% | 11.42% |
| XGBoost | $417.12 | $294.11 | 7.73% | 11.79% |

### Key Findings

- **ARIMA Dominance**: ARIMA significantly outperforms all tree-based models for both short-term (next day) and medium-term (next week) predictions, achieving less than 1% MAPE for next-day forecasts.
- **Why ARIMA Works Best**: Gold prices exhibit strong temporal autocorrelation and trend patterns that ARIMA models capture effectively through its autoregressive and moving average components.
- **Tree-Based Models**: Despite extensive feature engineering, gradient boosting models (XGBoost, LightGBM) and Random Forest show similar performance (~6-8% MAPE), suggesting that engineered features don't capture temporal dependencies as well as direct time series modeling.
- **Production Choice**: **ARIMA is deployed in production** due to superior accuracy, interpretability, and suitability for time series forecasting tasks.


## Live Application

Access the production web application at: **http://35.158.241.11:8080**

The application displays:
- Tomorrow's predicted gold price (+1 day)
- Day after tomorrow's predicted price (+2 days)
- Model metadata and last update timestamp

## Future Enhancements

- Automated CI/CD pipelines for model retraining
- Real-time model monitoring with CloudWatch
- Alerting system for significant price changes
- Model interpretability with SHAP/LIME
- Multi-model ensemble predictions

---

*This project demonstrates end-to-end MLOps practices from data collection to cloud deployment, following industry best practices for production machine learning systems.*

