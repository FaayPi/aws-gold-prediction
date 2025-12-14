# Model Summary für AWS SageMaker Deployment

## 📊 Aktueller Bester Stand der Modelle

### Finale Performance (mit 10y-Daten, 3y Rolling Window)

| Modell | Next Day RMSE | Next Day MAE | Next Week RMSE | Next Week MAE | Status |
|--------|---------------|-------------|----------------|---------------|--------|
| **ARIMA** | **50.19 USD** | 36.02 USD | **49.78 USD** | 34.44 USD | ✅ **Produktionsreif** |
| **XGBoost** | 483.74 USD | 396.31 USD | 532.91 USD | 397.74 USD | ⚠️ Akzeptabel (Tuning läuft) |
| **LSTM** | 3,747.80 USD | 3,733.16 USD | 3,781.26 USD | 3,766.63 USD | ❌ Nicht brauchbar |

---

## 🏆 Empfohlene Modelle für SageMaker

### 1. ARIMA (Primäres Modell) ✅

**Performance:**
- RMSE: ~50 USD (Fehler ~2.5% bei ~$2000/oz)
- Sehr konsistent für Next Day und Next Week

**Konfiguration:**
- **Modelltyp**: ARIMA(3,1,3)
- **Daten**: 3-Jahres Rolling Window (letzte 3 Jahre)
- **Input**: Nur Close-Preis (univariate Zeitreihe)
- **Methode**: Rolling Forecast (retrainiert für jeden Testpunkt)

**Vorteile:**
- ✅ Beste Performance
- ✅ Einfach und robust
- ✅ Schnell zu trainieren
- ✅ Keine Feature-Engineering nötig
- ✅ Interpretierbar

**Nachteile:**
- ⚠️ Nutzt nur Preis-Daten (keine externen Features)
- ⚠️ Statische Parameter (p, d, q)

**SageMaker Deployment:**
- **Container**: Python 3.11 mit statsmodels
- **Training**: Minimal (nur ARIMA-Fit)
- **Inference**: Schnell (nur Forecast)

---

### 2. XGBoost (Backup/Alternative) ⚠️

**Performance:**
- RMSE: ~484 USD (Fehler ~24% bei ~$2000/oz)
- Akzeptabel, aber deutlich schlechter als ARIMA

**Konfiguration:**
- **Modelltyp**: XGBoost Regressor
- **Features**: Top 12 Features (aus 30 erstellten Features)
- **Hyperparameter** (aktuell):
  - `n_estimators`: 800
  - `max_depth`: 3
  - `learning_rate`: 0.05
  - `subsample`: 0.9
  - `colsample_bytree`: 0.9
  - `min_child_weight`: 1
  - `reg_alpha`: 0.1
  - `reg_lambda`: 1.0
- **Daten**: 3-Jahres Rolling Window
- **Feature Selection**: Top 12 nach Korrelation

**Top Features (Next Day):**
1. `lag_1` - Preis von gestern
2. `roll_mean_5` - 5-Tage Moving Average
3. `ema_12` - 12-Tage Exponential Moving Average
4. `lag_2` - Preis von vor 2 Tagen
5. `roll_mean_10` - 10-Tage Moving Average
6. `ema_26` - 26-Tage Exponential Moving Average
7. `roll_mean_20` - 20-Tage Moving Average
8. `lag_5` - Preis von vor 5 Tagen
9. `lag_10` - Preis von vor 10 Tagen
10. `roll_std_20` - 20-Tage Standard Deviation
11. `roll_std_10` - 10-Tage Standard Deviation
12. `roll_std_5` - 5-Tage Standard Deviation

**Vorteile:**
- ✅ Kann komplexe Muster lernen
- ✅ Feature-Importance verfügbar
- ✅ Robust gegen Overfitting

**Nachteile:**
- ⚠️ Deutlich schlechter als ARIMA
- ⚠️ Braucht Feature-Engineering
- ⚠️ Hyperparameter-Tuning nötig (läuft noch)

**SageMaker Deployment:**
- **Container**: Python 3.11 mit xgboost, scikit-learn
- **Training**: Feature-Engineering + XGBoost Training
- **Inference**: Feature-Engineering + Prediction

---

### 3. LSTM (Nicht empfohlen) ❌

**Performance:**
- RMSE: ~3,748 USD (Fehler ~187% bei ~$2000/oz)
- **Nicht brauchbar für Produktion**

**Status:** Nicht für SageMaker Deployment empfohlen

---

## 📁 Wichtige Dateien für SageMaker

### Code-Dateien:
1. **`train_arima.py`** - ARIMA Training
2. **`train_xgb.py`** - XGBoost Training
3. **`feature_engineering_optimized.py`** - Feature-Pipeline (für XGBoost)
4. **`fetch_gold.py`** - Daten-Sammlung

### Daten:
- **`gold_GCF_10y_1d.csv`** - 10 Jahre historische Daten
- **Rolling Window**: Nutze nur letzte 3 Jahre für Training

### Konfiguration:
- **Rolling Window**: 3 Jahre (definiert in `ROLLING_YEARS = 3`)
- **Train/Test Split**: 80/20 (zeitbasiert)
- **Feature Selection**: Top 12 Features (für XGBoost)

---

## 🔧 SageMaker Deployment Plan

### Option 1: ARIMA als Primäres Modell (Empfohlen)

**Training Job:**
- **Container**: Python 3.11
- **Dependencies**: `statsmodels`, `pandas`, `numpy`, `scikit-learn`
- **Input**: S3 `s3://gold-ml-data/raw/gold_GCF_10y_1d.csv`
- **Output**: S3 `s3://gold-ml-data/models/arima/`
- **Training Script**: `train_arima.py` (angepasst für SageMaker)

**Endpoint:**
- **Instance Type**: `ml.t2.medium` (klein, günstig)
- **Inference Code**: ARIMA Forecast
- **Input Format**: JSON mit Datum
- **Output Format**: JSON mit Vorhersage

**Kosten**: ~$35/Monat (t2.medium 24/7)

---

### Option 2: XGBoost als Alternative

**Training Job:**
- **Container**: Python 3.11
- **Dependencies**: `xgboost`, `pandas`, `numpy`, `scikit-learn`
- **Input**: S3 `s3://gold-ml-data/raw/gold_GCF_10y_1d.csv`
- **Output**: S3 `s3://gold-ml-data/models/xgboost/`
- **Training Script**: `train_xgb.py` + `feature_engineering_optimized.py`

**Endpoint:**
- **Instance Type**: `ml.t2.medium` oder `ml.m5.large`
- **Inference Code**: Feature-Engineering + XGBoost Prediction
- **Input Format**: JSON mit Datum
- **Output Format**: JSON mit Vorhersage

**Kosten**: ~$35-70/Monat

---

### Option 3: Ensemble (ARIMA + XGBoost)

**Konzept:**
- Kombiniere beide Modelle
- Gewichteter Durchschnitt: 70% ARIMA + 30% XGBoost
- Oder: Nutze ARIMA als Primär, XGBoost als Backup

**Vorteile:**
- ✅ Robustheit (wenn ein Modell fehlschlägt)
- ✅ Potenzielle Verbesserung der Genauigkeit

**Nachteile:**
- ⚠️ Höhere Komplexität
- ⚠️ Höhere Kosten (2 Endpoints)

---

## 📋 SageMaker Setup Checkliste

### 1. Vorbereitung:
- [ ] SageMaker Notebook Instance erstellen
- [ ] S3 Bucket für Daten/Modelle: `gold-ml-data`
- [ ] IAM Role mit SageMaker Permissions

### 2. Daten:
- [ ] `gold_GCF_10y_1d.csv` zu S3 hochladen: `s3://gold-ml-data/raw/`
- [ ] Lambda-Funktion für tägliche Updates (bereits vorhanden)

### 3. Training Scripts anpassen:
- [ ] `train_arima.py` für SageMaker anpassen (S3 Input/Output)
- [ ] `train_xgb.py` für SageMaker anpassen
- [ ] `requirements.txt` für SageMaker Container

### 4. Training:
- [ ] SageMaker Training Job für ARIMA erstellen
- [ ] SageMaker Training Job für XGBoost erstellen (optional)
- [ ] Modelle in S3 speichern

### 5. Deployment:
- [ ] SageMaker Endpoint für ARIMA erstellen
- [ ] Inference Code testen
- [ ] API Gateway Integration (optional)

### 6. Monitoring:
- [ ] CloudWatch Logs für Endpoint
- [ ] Model Performance Tracking
- [ ] Retraining Schedule (täglich/wöchentlich)

---

## 💻 Code-Anpassungen für SageMaker

### ARIMA Training Script (SageMaker-kompatibel):

```python
# train_arima_sagemaker.py
import argparse
import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def model_fn(model_dir):
    """Lädt das trainierte Modell"""
    # ARIMA ist stateless, lade Parameter
    return None

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        return pd.read_json(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Macht Vorhersage"""
    # ARIMA Forecast Logic
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    args = parser.parse_args()
    
    # Training Logic
    # Save model to args.model_dir
```

### XGBoost Training Script (SageMaker-kompatibel):

```python
# train_xgb_sagemaker.py
import argparse
import os
import joblib
import pandas as pd
from xgboost import XGBRegressor
from feature_engineering_optimized import create_optimized_features, get_feature_columns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    args = parser.parse_args()
    
    # Load data from S3
    df = pd.read_csv(f"{args.train}/gold_GCF_10y_1d.csv")
    
    # Feature Engineering
    df = create_optimized_features(df)
    
    # Training
    # ... XGBoost training logic ...
    
    # Save model
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    joblib.dump(scaler, os.path.join(args.model_dir, 'scaler.joblib'))
    joblib.dump(feature_cols, os.path.join(args.model_dir, 'features.joblib'))
```

---

## 📊 Performance-Erwartungen

### ARIMA:
- **RMSE**: ~50 USD
- **MAE**: ~36 USD
- **Fehler**: ~2.5% (bei ~$2000/oz)
- **Status**: ✅ Produktionsreif

### XGBoost:
- **RMSE**: ~484 USD (aktuell)
- **Ziel nach Tuning**: < 400 USD
- **Fehler**: ~24% (aktuell), Ziel: < 20%
- **Status**: ⚠️ Akzeptabel, kann verbessert werden

---

## 🎯 Empfehlung für SageMaker

**Primäres Modell: ARIMA**
- ✅ Beste Performance
- ✅ Einfach zu deployen
- ✅ Geringe Kosten
- ✅ Schnelle Inference

**Backup (optional): XGBoost**
- ⚠️ Nur wenn ARIMA nicht ausreicht
- ⚠️ Höhere Komplexität
- ⚠️ Höhere Kosten

**Nicht verwenden: LSTM**
- ❌ Zu schlechte Performance
- ❌ Zu komplex
- ❌ Nicht produktionsreif

---

## 📝 Nächste Schritte

1. **Warte auf Hyperparameter-Tuning Ergebnisse** (XGBoost)
2. **Erstelle SageMaker-kompatible Training Scripts**
3. **Teste lokal mit SageMaker Local Mode**
4. **Deploye ARIMA zu SageMaker Endpoint**
5. **Teste Endpoint mit realen Daten**
6. **Richte Monitoring ein**

---

## 🔗 Wichtige AWS Services

- **SageMaker**: Model Training & Deployment
- **S3**: Daten & Model Storage
- **Lambda**: Tägliche Daten-Updates (bereits vorhanden)
- **EventBridge**: Scheduling (bereits vorhanden)
- **API Gateway**: REST API (optional)
- **CloudWatch**: Monitoring & Logs

---

**Stand:** Dezember 2024  
**Nächste Aktualisierung:** Nach Hyperparameter-Tuning Abschluss
