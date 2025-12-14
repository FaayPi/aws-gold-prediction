"""
Compare ARIMA, XGBoost, and LSTM on Gold Futures (GC=F) daily data.
Assumes the following scripts are runnable in the same env:
 - train_arima.py
 - train_xgb.py
 - train_lstm.py
Each script prints RMSE/MAE; here we import and run their core functions.
"""

from __future__ import annotations

# Import the core routines from the pipelines package
from pipelines import train_arima, train_xgb, train_lstm


def main():
    results = {}
    
    # ARIMA
    print("=" * 60)
    print("Running ARIMA...")
    print("=" * 60)
    try:
        train_arima.main()
    except Exception as e:
        print(f"❌ ARIMA failed: {e}")
    
    # XGBoost
    print("\n" + "=" * 60)
    print("Running XGBoost...")
    print("=" * 60)
    try:
        train_xgb.main()
    except Exception as e:
        print(f"❌ XGBoost failed: {e}")
    
    # LSTM
    print("\n" + "=" * 60)
    print("Running LSTM...")
    print("=" * 60)
    try:
        train_lstm.main()
    except Exception as e:
        print(f"❌ LSTM failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Model comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
