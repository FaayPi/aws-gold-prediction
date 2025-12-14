# Technical Overview: Gold Price Prediction ML Project

## 🎯 Projekt-Ziel
Entwicklung eines Machine Learning Systems zur Vorhersage von Goldpreisen (GC=F - Gold Futures) mit mehreren ML-Modellen und einer vollständigen AWS-basierten Data Pipeline.

---

## 🛠️ Technische Tools & Technologien

### **Cloud Infrastructure (AWS)**
- **AWS Lambda**: Serverless Data Collection & Preprocessing
  - `lamda_data_collection.py`: Tägliche Datensammlung von Yahoo Finance
  - `lambda_preprocess_gold.py`: Datenverarbeitung und CSV-Generierung
- **Amazon S3**: Datenspeicherung
  - Raw Data: `raw/YYYY/MM/DD/gold_prices_*.json`
  - Processed Data: `processed/daily/gold_prices_*.csv`
- **Amazon EventBridge**: Automatisiertes Scheduling
  - Tägliche Trigger für Data Collection
  - Sequenzielle Ausführung (Collection → Preprocessing)
- **AWS SageMaker**: ML Model Training & Deployment
  - Training Scripts: `train_arima_sagemaker.py`, `train_xgb_sagemaker.py`
  - Model Artifacts: `.pkl` / `.joblib` Format
  - Endpoint Deployment für Inference

### **Python Libraries & Frameworks**

#### **Data Collection & Processing**
- `yfinance` (v0.2.54): Yahoo Finance API Integration
- `pandas` (≥2.2.0): Datenmanipulation & Feature Engineering
- `numpy`: Numerische Operationen
- `boto3`: AWS SDK für S3, Lambda, SageMaker

#### **Machine Learning**
- `scikit-learn` (≥1.3.0): Preprocessing, Evaluation, Feature Selection
- `statsmodels` (≥0.14.0): ARIMA Time-Series Model
- `xgboost` (≥2.0.0): Gradient Boosting Regressor
- `tensorflow` (≥2.13.0): LSTM Deep Learning Model

#### **Data Storage**
- `pyarrow` (≥16.0.0): Parquet Support (optional)
- `s3fs` (≥2024.1.0): S3 File System Interface

---

## 🏗️ Architektur-Highlights

### **1. Multi-Stage Data Pipeline**

```
Yahoo Finance API
    ↓
AWS Lambda (Collection)
    ↓
S3 Raw Storage (JSON)
    ↓
AWS Lambda (Preprocessing)
    ↓
S3 Processed Storage (CSV)
    ↓
SageMaker Training
    ↓
Model Artifacts (S3)
    ↓
SageMaker Endpoint (Inference)
```

### **2. Feature Engineering Pipeline**

**Optimized Feature Set** (~30 Features):
- **Lag Features**: `lag_1`, `lag_2`, `lag_5`, `lag_10`
- **Rolling Statistics**: 
  - Mean: `roll_mean_5`, `roll_mean_10`, `roll_mean_20`
  - Std: `roll_std_5`, `roll_std_10`, `roll_std_20`
- **Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - EMA (Exponential Moving Average): `ema_12`, `ema_26`
  - Bollinger Bands
- **Volatility Features**: `volatility_5`, `volatility_20`
- **Seasonal Features**: `dayofweek`, `month`, `quarter`
- **Price Features**: `returns`, `high_low_ratio`

**Feature Selection**: Top 12 korrelierte Features pro Target (Next Day / Next Week)

### **3. Multi-Model Approach**

#### **ARIMA (AutoRegressive Integrated Moving Average)**
- **Modell**: ARIMA(3,1,3)
- **Input**: Nur `close` Preis (univariate time series)
- **Training**: Rolling Window (3 Jahre aus 10 Jahren Historie)
- **Performance**: 
  - Next Day RMSE: ~50 USD
  - Next Week RMSE: ~50 USD
- **Status**: ✅ Bestes Modell (höchste Genauigkeit)

#### **XGBoost (Gradient Boosting)**
- **Modell**: XGBRegressor mit optimierten Hyperparametern
- **Features**: Top 12 korrelierte Features (separat für Day/Week)
- **Hyperparameter**:
  - `n_estimators=800`, `max_depth=3`
  - `learning_rate=0.05`, `subsample=0.9`
  - `reg_alpha=0.1`, `reg_lambda=1.0`
- **Preprocessing**: StandardScaler für Feature-Normalisierung
- **Performance**: 
  - Next Day RMSE: ~484 USD
  - Next Week RMSE: ~533 USD
- **Status**: ⚠️ Backup-Modell (akzeptabel, aber deutlich schlechter als ARIMA)

#### **LSTM (Long Short-Term Memory)**
- **Modell**: TensorFlow/Keras Sequential LSTM
- **Architektur**: LSTM Layers + Dense Output
- **Features**: Alle 30 optimierten Features
- **Performance**: 
  - Next Day RMSE: ~3,748 USD
  - Next Week RMSE: ~3,781 USD
- **Status**: ❌ Nicht empfohlen (unbrauchbar für Production)

---

## 🔧 Technische Highlights

### **1. Robust Data Collection**
- **Rate Limit Handling**: 
  - Retry-Logik mit exponential backoff (30s/60s/90s/180s)
  - JSONDecodeError Detection als Rate-Limit-Indikator
  - Multiple Fallback-Strategien (7d → 30d → 90d → 3mo)
- **Error Resilience**: 
  - Graceful Degradation bei API-Fehlern
  - S3-Logging auch bei fehlgeschlagenen Requests
  - Detaillierte CloudWatch Logs

### **2. Optimized Feature Engineering**
- **Reduktion von 96 → 30 Features**: 
  - Eliminierung von Overfitting durch Feature-Reduktion
  - Fokus auf korrelierte Features
- **Data Quality Checks**:
  - NaN/Inf Detection & Handling
  - Constant Feature Removal
  - Correlation-based Feature Selection

### **3. Rolling Window Training**
- **Problem**: 10 Jahre Daten → Regime Changes → schlechte Model-Performance
- **Lösung**: Training nur auf letzten 3 Jahren (Rolling Window)
- **Ergebnis**: Stabile Model-Performance

### **4. SageMaker Integration**
- **Training Scripts**: 
  - `model_fn()`: Model Loading für Inference
  - `predict_fn()`: Prediction Logic
  - Environment Variables: `SM_MODEL_DIR`, `SM_CHANNEL_TRAINING`
- **Model Artifacts**:
  - ARIMA: `arima_model.pkl` (Series + Order)
  - XGBoost: `model_day.joblib`, `model_week.joblib`, `scaler_*.joblib`, `features_*.joblib`

### **5. Production-Ready Error Handling**
- **Lambda Functions**: 
  - Try-Except Blocks mit detailliertem Logging
  - S3-Fallback auch bei Fehlern
  - CloudWatch Integration
- **Data Validation**: 
  - Datum-Konsistenz-Checks
  - Fallback auf neueste verfügbare Daten
  - Deduplication in Preprocessing

---

## 📊 Performance Metrics

### **Evaluation Metrics**
- **RMSE** (Root Mean Squared Error): Hauptmetrik für Modellvergleich
- **MAE** (Mean Absolute Error): Zusätzliche Metrik

### **Model Comparison Summary**

| Modell | Next Day RMSE | Next Week RMSE | Status |
|--------|---------------|----------------|--------|
| **ARIMA** | ~50 USD | ~50 USD | ✅ Best |
| **XGBoost** | ~484 USD | ~533 USD | ⚠️ Backup |
| **LSTM** | ~3,748 USD | ~3,781 USD | ❌ Nicht empfohlen |

---

## 🚀 Deployment & Automation

### **EventBridge Scheduling**
- **Data Collection**: Täglich (z.B. 06:00 UTC)
- **Preprocessing**: Nach erfolgreicher Collection
- **Optionen**: 
  - Separate Rules mit Delay
  - Step Functions für Sequenzierung
  - Lambda-to-Lambda Invocation

### **SageMaker Workflow**
1. **Training Job**: 
   - Input: S3 CSV Data
   - Output: Model Artifacts in S3
2. **Model Registry**: Versionierung & Management
3. **Endpoint Deployment**: Real-time Inference
4. **Monitoring**: CloudWatch Metrics & Logs

---

## 🔐 Security & Best Practices

### **IAM Roles & Permissions**
- **Lambda Roles**: 
  - S3 Read/Write für Bucket
  - CloudWatch Logs
  - Lambda Invoke (für Preprocessing)
- **SageMaker Roles**: 
  - S3 Access für Training Data & Artifacts
  - ECR Access für Custom Containers (optional)

### **Data Privacy**
- Keine persönlichen Daten
- Öffentliche Finanzdaten (Yahoo Finance)
- S3 Bucket Policies für Access Control

---

## 📈 Skalierbarkeit & Erweiterbarkeit

### **Skalierung**
- **Lambda**: Automatisch (Concurrent Executions)
- **SageMaker**: 
  - Multi-Instance Training
  - Auto-Scaling Endpoints
- **S3**: Unbegrenzte Storage

### **Erweiterungsmöglichkeiten**
- **Mehrere Ticker**: `TICKERS=GC=F,XAUUSD=X`
- **Mehrere Intervals**: `INTERVAL=1h,4h,1d`
- **Feature Engineering**: Weitere technische Indikatoren
- **Model Ensembles**: Kombination mehrerer Modelle
- **Real-time Streaming**: Kinesis Data Streams Integration

---

## 🐛 Bekannte Herausforderungen & Lösungen

### **1. Yahoo Finance Rate Limits**
- **Problem**: 429 Too Many Requests
- **Lösung**: 
  - Kleinere Zeitfenster (7d statt 30d)
  - Retry mit Backoff
  - Alternative Datenquellen (AlphaVantage, Stooq)

### **2. Overfitting bei vielen Features**
- **Problem**: 96 Features → schlechte Performance
- **Lösung**: Feature-Reduktion auf 30 optimierte Features

### **3. Regime Changes in langen Zeitreihen**
- **Problem**: 10 Jahre Daten → instabile Modelle
- **Lösung**: Rolling Window (3 Jahre Training)

### **4. LSTM Performance**
- **Problem**: Sehr schlechte Vorhersagen (RMSE > 3,700)
- **Status**: Nicht für Production empfohlen

---

## 📚 Code-Struktur

```
ML_model_gold_price_predictions/
├── pipelines/
│   ├── fetch_gold.py                    # Lokale Datensammlung
│   ├── feature_engineering_optimized.py # Feature Pipeline
│   ├── train_arima_sagemaker.py         # ARIMA Training (SageMaker)
│   ├── train_xgb_sagemaker.py           # XGBoost Training (SageMaker)
│   └── __init__.py
├── lamda_data_collection.py            # AWS Lambda: Data Collection
├── lambda_preprocess_gold.py           # AWS Lambda: Preprocessing
├── requirements.txt                     # Python Dependencies
└── TECHNICAL_OVERVIEW.md               # Diese Datei
```

---

## 🎓 Key Learnings

1. **Feature Engineering > Model Complexity**: Weniger, aber gut gewählte Features performen besser
2. **Time Series Stability**: Rolling Windows helfen bei Regime Changes
3. **Rate Limit Management**: Retry-Logik & kleinere Requests sind essentiell
4. **Model Selection**: Einfache Modelle (ARIMA) können komplexe (LSTM) übertreffen
5. **Production Readiness**: Error Handling & Logging sind kritisch

---

## 🔮 Zukünftige Verbesserungen

- [ ] Alternative Datenquellen (AlphaVantage, Quandl)
- [ ] Model Ensembles (ARIMA + XGBoost)
- [ ] Real-time Inference API
- [ ] Automated Model Retraining Pipeline
- [ ] A/B Testing für Model-Versionen
- [ ] Feature Store (SageMaker Feature Store)
- [ ] Model Monitoring & Drift Detection

---

**Letzte Aktualisierung**: Dezember 2025

