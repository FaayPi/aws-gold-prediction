"""
Optimierte Feature-Engineering-Pipeline für Goldpreis-Prognosen.

Reduzierte Anzahl von Features (20-30 statt 96) um Overfitting zu vermeiden.
Fokus auf wichtigste, bewährte Features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Berechnet RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Berechnet MACD."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def create_optimized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt optimierte Features - reduziert auf wichtigste.
    
    Ziel: ~20-30 Features statt 96
    """
    df = df.copy()
    
    # Sicherstellen, dass Index ein DatetimeIndex ist
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index("Date")
        elif "datetime" in df.columns:
            df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index)
    
    df = df.sort_index()
    close = df["close"]
    
    # ===== 1. Grundlegende Preis-Features (5 Features) =====
    df["return"] = np.log(close).diff()
    df["return_abs"] = np.abs(df["return"])
    df["price_change_pct"] = close.pct_change()
    
    if "high" in df.columns and "low" in df.columns:
        df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]
    
    if "open" in df.columns:
        df["oc_change_pct"] = (df["close"] - df["open"]) / df["open"]
    
    # ===== 2. Wichtigste Lags (6 Features) =====
    for lag in [1, 2, 5, 10]:
        df[f"lag_{lag}"] = close.shift(lag)
    
    # Return Lags
    for lag in [1, 2]:
        df[f"return_lag_{lag}"] = df["return"].shift(lag)
    
    # ===== 3. Wichtigste Rolling Stats (8 Features) =====
    # Nur die wichtigsten Fenster: 5, 10, 20
    for window in [5, 10, 20]:
        df[f"roll_mean_{window}"] = close.rolling(window=window).mean()
        df[f"roll_std_{window}"] = close.rolling(window=window).std()
    
    # Price Position in Range
    for window in [10, 20]:
        roll_min = close.rolling(window=window).min()
        roll_max = close.rolling(window=window).max()
        df[f"price_position_{window}"] = (close - roll_min) / (roll_max - roll_min + 1e-8)
    
    # ===== 4. Technische Indikatoren (5 Features) =====
    # RSI
    df["rsi_14"] = calculate_rsi(close, window=14)
    
    # MACD
    macd_line, signal_line, histogram = calculate_macd(close)
    df["macd"] = macd_line
    df["macd_histogram"] = histogram
    
    # EMA
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    
    # ===== 5. Volatilität (3 Features) =====
    returns = df["return"]
    for window in [5, 10]:
        df[f"volatility_{window}"] = returns.rolling(window=window).std()
    
    if "high" in df.columns and "low" in df.columns:
        df["hl_range_avg_10"] = (df["high"] - df["low"]).rolling(window=10).mean() / df["close"]
    
    # ===== 6. Saisonale Features (3 Features) =====
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_month_end"] = df.index.is_month_end.astype(int)
    
    # ===== 7. Targets =====
    df["y_day"] = df["close"].shift(-1)
    df["y_week"] = df["close"].shift(-5)
    
    # Entferne NaN-Werte
    df = df.dropna()
    
    # Entferne Inf-Werte
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Entferne konstante Features
    for col in df.columns:
        if df[col].nunique() <= 1:
            df = df.drop(columns=[col])
    
    return df


def get_feature_columns(df: pd.DataFrame, exclude_targets: bool = True) -> list[str]:
    """Gibt Liste aller Feature-Spalten zurück."""
    exclude = ["y_day", "y_week"] if exclude_targets else []
    exclude.extend(["open", "high", "low", "close", "volume"])
    
    feature_cols = [col for col in df.columns if col not in exclude]
    return feature_cols


if __name__ == "__main__":
    # Test
    from pathlib import Path
    
    data_path = Path("gold_GCF_3y_1d.csv")
    if not data_path.exists():
        print(f"❌ {data_path} nicht gefunden.")
        exit(1)
    
    print("📊 Lade Daten...")
    df = pd.read_csv(data_path)
    
    if "Date" in df.columns:
        df = df.set_index("Date")
    elif "datetime" in df.columns:
        df = df.set_index("datetime")
    else:
        df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)
    
    print(f"   Original: {len(df)} Zeilen")
    
    print("\n🔧 Erstelle optimierte Features...")
    df_features = create_optimized_features(df)
    
    feature_cols = get_feature_columns(df_features)
    print(f"   Nach Feature-Engineering: {len(df_features)} Zeilen")
    print(f"   Anzahl Features: {len(feature_cols)}")
    print(f"   Features: {', '.join(feature_cols[:10])}...")
    
    # Prüfe auf Probleme
    print("\n✅ Validierung:")
    print(f"   NaN-Werte: {df_features[feature_cols].isna().sum().sum()}")
    print(f"   Inf-Werte: {np.isinf(df_features[feature_cols]).sum().sum()}")
    print(f"   Konstante Features: {sum(df_features[col].nunique() <= 1 for col in feature_cols)}")

