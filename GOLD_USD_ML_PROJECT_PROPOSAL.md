# Gold/USD Price Prediction Using MLOps on AWS

## Project Title
**GoldSight: Daily Gold/USD Price Forecasting for Day Traders**

---

## 🎯 Target Audience

### Primary User: Day Traders & Retail Investors

**"Alex the Active Trader"**
- Age: 28-45, tech-savvy
- Trades 2-3 times per week
- **Needs:**
  - Predictions for next hour, next day, next week
  - Clear BUY/HOLD/SELL signals
  - Email alerts for significant price movements
- **Pain Point:** Missing optimal entry/exit points due to manual analysis
- **Value:** Automated predictions save time and improve trading decisions
- **Budget:** Free tier initially, willing to pay $20-50/month for premium

---

## 📊 Data Sources

### Free APIs (No Cost)

**1. Yahoo Finance API**
- **What:** Historical Gold prices (GC=F), S&P 500, VIX
- **Frequency:** Daily + Hourly data
- **Cost:** Free
- **Access:** Python library `yfinance`

**2. Alpha Vantage API**
- **What:** Gold/USD (XAUUSD), USD Index, intraday prices
- **Frequency:** Daily + Intraday (hourly)
- **Cost:** Free (500 API calls/day)
- **Access:** REST API with free API key

### Feature Engineering Data (All Free)

| Feature | Source | Why Important |
|---------|--------|---------------|
| **Gold Price (GC=F)** | Yahoo Finance | Main target variable |
| **USD Index (DXY)** | Alpha Vantage | Currency strength affects gold |
| **10-Year Treasury Yield** | Yahoo Finance | Interest rates impact |
| **S&P 500 (SPY)** | Yahoo Finance | Market sentiment indicator |
| **VIX (Volatility Index)** | Yahoo Finance | Fear gauge - gold is safe haven |
| **Crude Oil (CL=F)** | Yahoo Finance | Commodity correlation |

---

## 🏗️ System Architecture (Simplified)

```
┌─────────────────────────────────────────────────────────────┐
│                    1️⃣ DATA INGESTION                        │
│                    (Runs every hour)                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────────┐        ┌──────────────────┐         │
│   │ Yahoo Finance    │        │ Alpha Vantage    │         │
│   │ • Gold (GC=F)    │        │ • Gold (XAUUSD)  │         │
│   │ • S&P 500        │        │ • USD Index      │         │
│   │ • VIX, Oil       │        │ • Intraday data  │         │
│   └────────┬─────────┘        └────────┬─────────┘         │
│            │                           │                    │
│            └───────────┬───────────────┘                    │
│                        ▼                                     │
│              ┌──────────────────┐                           │
│              │  AWS Lambda      │                           │
│              │  (Python Script) │                           │
│              │  - Fetch data    │                           │
│              │  - Validate      │                           │
│              └────────┬─────────┘                           │
│                       ▼                                      │
│              ┌──────────────────┐                           │
│              │   Amazon S3      │                           │
│              │   raw/YYYY/MM/DD/│                           │
│              │   HH/data.json   │                           │
│              └──────────────────┘                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              2️⃣ DATA PREPROCESSING                          │
│              (Triggered after ingestion)                     │
├─────────────────────────────────────────────────────────────┤
│              ┌──────────────────┐                           │
│              │  AWS Lambda      │                           │
│              │  ETL Job         │                           │
│              │  - Clean data    │                           │
│              │  - Feature eng.  │                           │
│              │    • RSI, MACD   │                           │
│              │    • Lag features│                           │
│              │  - Normalize     │                           │
│              └────────┬─────────┘                           │
│                       ▼                                      │
│              ┌──────────────────┐                           │
│              │   Amazon S3      │                           │
│              │   processed/     │                           │
│              │   features.csv   │                           │
│              └──────────────────┘                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              3️⃣ MODEL TRAINING                              │
│              (Runs daily at midnight)                        │
├─────────────────────────────────────────────────────────────┤
│              ┌──────────────────┐                           │
│              │  SageMaker       │                           │
│              │  Training Job    │                           │
│              │                  │                           │
│              │  Model: LSTM     │                           │
│              │  Input: 7 days   │                           │
│              │  Output: Next    │                           │
│              │    1h, 1d, 7d    │                           │
│              └────────┬─────────┘                           │
│                       ▼                                      │
│              ┌──────────────────┐                           │
│              │  Model Registry  │                           │
│              │  If RMSE < $5/oz │                           │
│              │  → Deploy        │                           │
│              └────────┬─────────┘                           │
└────────────────────────┼────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              4️⃣ PREDICTION API                              │
├─────────────────────────────────────────────────────────────┤
│              ┌──────────────────┐                           │
│              │  SageMaker       │                           │
│              │  Endpoint        │                           │
│              └────────┬─────────┘                           │
│                       ▼                                      │
│              ┌──────────────────┐                           │
│              │  API Gateway     │                           │
│              │  GET /predict    │                           │
│              │  Returns:        │                           │
│              │  {               │                           │
│              │   "1h": $1982,   │                           │
│              │   "1d": $1995,   │                           │
│              │   "7d": $2015    │                           │
│              │  }               │                           │
│              └────────┬─────────┘                           │
└────────────────────────┼────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              5️⃣ USER INTERFACE                              │
├─────────────────────────────────────────────────────────────┤
│              ┌──────────────────┐                           │
│              │   Streamlit      │                           │
│              │   Web Dashboard  │                           │
│              │   + Email Alerts │                           │
│              └──────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 How It Works (Step-by-Step)

### Step 1: Data Collection (Every Hour)
```python
# AWS Lambda triggers hourly
import yfinance as yf
import requests

# Fetch from Yahoo Finance
gold = yf.download("GC=F", period="7d", interval="1h")
sp500 = yf.download("SPY", period="7d", interval="1h")
vix = yf.download("^VIX", period="7d", interval="1h")

# Fetch from Alpha Vantage
url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol=XAU&to_symbol=USD&interval=60min&apikey=YOUR_KEY"
alpha_data = requests.get(url).json()

# Save to S3
save_to_s3(data, "s3://gold-ml/raw/2025/12/08/14/data.json")
```

### Step 2: Feature Engineering
```python
# Create technical indicators
data['RSI'] = calculate_rsi(data['close'])
data['MACD'] = calculate_macd(data['close'])
data['price_1h_ago'] = data['close'].shift(1)
data['price_24h_ago'] = data['close'].shift(24)
data['rolling_mean_7d'] = data['close'].rolling(168).mean()

# Normalize features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
```

### Step 3: Train LSTM Model
```python
# SageMaker training script
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(168, n_features)),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(3)  # Output: [1h, 1d, 7d] predictions
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

### Step 4: Make Predictions
```python
# User visits dashboard
current_data = get_last_7_days()
prediction = model.predict(current_data)

return {
    "next_1h": prediction[0],
    "next_1d": prediction[1],
    "next_7d": prediction[2],
    "signal": "BUY" if prediction[1] > current_price else "SELL"
}
```

---

## 🎨 Web Dashboard (Streamlit)

```
┌──────────────────────────────────────────────────────────┐
│  💰 GoldSight - Gold/USD Predictions                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Current Gold Price: $1,978.50 ↑ +$12.30 (+0.63%)       │
│  Last Updated: 5 minutes ago                             │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │  📈 Price Predictions                            │    │
│  │                                                   │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │    │
│  │  │ Next Hour│  │ Next Day │  │ Next Week│      │    │
│  │  │          │  │          │  │          │      │    │
│  │  │ $1,982   │  │ $1,995   │  │ $2,015   │      │    │
│  │  │ +0.18%   │  │ +0.84%   │  │ +1.85%   │      │    │
│  │  │ 🟢 UP    │  │ 🟢 UP    │  │ 🟢 UP    │      │    │
│  │  └──────────┘  └──────────┘  └──────────┘      │    │
│  │                                                   │    │
│  │  Signal: 🟢 BUY (Confidence: 73%)                │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │  📊 7-Day Price Chart                            │    │
│  │                                                   │    │
│  │  2000 ┼           ╱─╲    ← Predicted             │    │
│  │       │        ╱─╯   ╲                           │    │
│  │  1990 ┼─────╱          ╲                         │    │
│  │       │  ╱                ╲                       │    │
│  │  1980 ┼─╯  ← Actual        ╲                     │    │
│  │       │                       ╲                   │    │
│  │  1970 ┼─────────────────────────                 │    │
│  │       └───────────────────────────               │    │
│  │       -7d  -5d  -3d  -1d  now  +1d  +3d  +7d    │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │  🔔 Email Alerts                                 │    │
│  │                                                   │    │
│  │  Your email: alex@trader.com                     │    │
│  │                                                   │    │
│  │  ☑ Alert if price > $2,000                       │    │
│  │  ☑ Alert if price < $1,950                       │    │
│  │  ☑ Alert if prediction changes to SELL           │    │
│  │                                                   │    │
│  │  [Save Alert Settings]                           │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │  📈 Model Performance                            │    │
│  │                                                   │    │
│  │  RMSE: $3.45/oz        🟢 Good                   │    │
│  │  MAE:  $2.10/oz        🟢 Excellent              │    │
│  │  Accuracy (±$5): 87%   🟢 Good                   │    │
│  │  Last Retrain: 2h ago                            │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │  📰 Market Factors                               │    │
│  │                                                   │    │
│  │  • USD Index: 103.45 → Bearish for gold          │    │
│  │  • VIX: 18.2 → Moderate fear                     │    │
│  │  • S&P 500: +0.8% → Risk-on sentiment            │    │
│  │  • 10Y Yield: 4.23% → Rising rates bearish       │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## 🧪 ML Model Details

### LSTM Architecture (Simplified)

```python
Input: 168 hours (7 days) of data
Features per hour: 10
  - Gold price
  - USD Index
  - VIX
  - S&P 500
  - Oil price
  - RSI, MACD (technical indicators)
  - Hour of day, day of week

Layers:
  - LSTM(128) → captures patterns in 7-day window
  - LSTM(64)  → deeper temporal understanding
  - Dense(32) → combines features
  - Dense(3)  → outputs [1h, 1d, 7d] predictions

Training:
  - Loss: Mean Squared Error (MSE)
  - Optimizer: Adam
  - Epochs: 50
  - Daily retrain with new data
```

### Performance Targets

| Metric | Target | Status Check |
|--------|--------|-------------|
| RMSE | < $5/oz | Deploy only if met |
| MAE | < $3/oz | Acceptable error |
| Directional Accuracy | > 70% | UP/DOWN correct |
| Latency | < 200ms | Fast user experience |

---

## 📅 12-Week Implementation Plan

### Phase 1: MVP (Weeks 1-4)

**Week 1: Data Pipeline**
- Set up AWS account (Free Tier)
- Create Lambda function to fetch Yahoo Finance + Alpha Vantage data
- Store in S3 bucket
- Test hourly scheduling with EventBridge

**Week 2: Preprocessing & Features**
- Lambda for ETL (clean, engineer features)
- Calculate RSI, MACD, lag features
- Save processed data to S3

**Week 3: Train First Model**
- Build LSTM model in SageMaker
- Train on historical data (6 months)
- Evaluate RMSE/MAE
- Save model to S3

**Week 4: Simple Dashboard**
- Build Streamlit web app
- Show current price + predictions
- Deploy on AWS App Runner
- Test end-to-end flow

---

### Phase 2: Improve (Weeks 5-8)

**Week 5-6: Better Model**
- Experiment with hyperparameters
- Add more features (VIX, Oil)
- Ensemble predictions if needed
- Improve accuracy to >85%

**Week 7: Monitoring**
- CloudWatch logs for Lambda
- Track model performance over time
- Alert if RMSE increases (model decay)

**Week 8: Email Alerts**
- Integrate AWS SES (Simple Email Service)
- User can set price alerts
- Daily summary email with predictions

---

### Phase 3: Polish (Weeks 9-12)

**Week 9: API**
- Create REST API (API Gateway + Lambda)
- Endpoint: `GET /predict`
- Return JSON predictions

**Week 10: User Authentication**
- Simple login (username/password)
- Store user preferences (email, alerts)

**Week 11: Testing**
- Backtest predictions vs actual prices
- Calculate profitability if following signals
- Fix bugs

**Week 12: Documentation & Launch**
- Write README
- Create demo video
- Deploy publicly
- Share on Twitter/LinkedIn

---

## 💰 Cost Estimate (AWS Free Tier)

| Service | Usage | Cost |
|---------|-------|------|
| **Lambda** | 720 calls/month (hourly) | FREE (1M free) |
| **S3** | 5 GB storage | FREE (5GB free) |
| **SageMaker Training** | 1 hour/day | FREE (125 hours free) |
| **SageMaker Endpoint** | t2.medium 24/7 | ~$35/month |
| **API Gateway** | 10k requests | FREE (1M free) |
| **App Runner (Streamlit)** | Small instance | ~$5/month |
| **SES (Email)** | 100 emails/day | FREE (62k/month free) |
| **Total** | | **~$40/month** |

**Free for first 12 months with AWS Free Tier!**

---

## 🎯 Success Metrics

**Week 4 (MVP):**
- ✅ Pipeline runs hourly without errors
- ✅ Model RMSE < $8/oz (baseline)
- ✅ Dashboard shows predictions

**Week 8 (Beta):**
- ✅ Model RMSE < $5/oz
- ✅ Email alerts working
- ✅ 5 beta testers using it

**Week 12 (Launch):**
- ✅ Directional accuracy > 70%
- ✅ 50 users signed up
- ✅ 10+ positive feedback

---

## 🚀 Next Steps

1. ✅ **Get API Keys**
   - Yahoo Finance: No key needed (use `yfinance` library)
   - Alpha Vantage: Free key at https://www.alphavantage.co/support/#api-key

2. ✅ **Set up AWS**
   - Create free tier account
   - Set up IAM user with permissions
   - Create S3 bucket: `gold-ml-data`

3. ✅ **Create GitHub Repo**
   - Initialize project structure
   - Add `.gitignore` for AWS credentials
   - Start Week 1!

4. ✅ **Start Coding**
   - Week 1 Day 1: Write Lambda to fetch Yahoo Finance data
   - Test locally first, then deploy to AWS

---

## 📚 Learning Resources

**APIs:**
- Yahoo Finance: https://pypi.org/project/yfinance/
- Alpha Vantage: https://www.alphavantage.co/documentation/

**AWS:**
- Lambda Tutorial: https://aws.amazon.com/lambda/getting-started/
- SageMaker Tutorial: https://aws.amazon.com/sagemaker/getting-started/

**Machine Learning:**
- Time Series with LSTM: https://www.tensorflow.org/tutorials/structured_data/time_series
- Financial Prediction: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

---

**Ready to start? Let's build this! 🚀**
