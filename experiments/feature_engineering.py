"""
Feature Engineering Pipeline für Goldpreis-Prognosen.

Erstellt erweiterte Features:
- Technische Indikatoren: RSI, MACD, Bollinger Bands
- Saisonale Features: Wochentag, Monat, Quartal
- Erweiterte Lags und Rolling Statistics
- Volatilitäts-Features
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Berechnet den Relative Strength Index (RSI).
    
    RSI misst die Stärke von Preisbewegungen (0-100).
    - RSI > 70: Überkauft (möglicherweise Verkaufssignal)
    - RSI < 30: Überverkauft (möglicherweise Kaufsignal)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Berechnet MACD (Moving Average Convergence Divergence).
    
    Returns:
        macd_line: MACD-Linie (fast EMA - slow EMA)
        signal_line: Signal-Linie (EMA von MACD)
        histogram: MACD - Signal (Divergenz)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Berechnet Bollinger Bands.
    
    Returns:
        upper_band: Oberes Band (Mittelwert + num_std * StdDev)
        middle_band: Mittleres Band (Moving Average)
        lower_band: Unteres Band (Mittelwert - num_std * StdDev)
    """
    middle_band = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    return upper_band, middle_band, lower_band


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt technische Indikatoren zum DataFrame hinzu."""
    df = df.copy()
    close = df["close"]
    
    # RSI (Relative Strength Index)
    df["rsi_14"] = calculate_rsi(close, window=14)
    df["rsi_7"] = calculate_rsi(close, window=7)  # Kurzfristiger RSI
    
    # MACD
    macd_line, signal_line, histogram = calculate_macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
    df["bb_upper"] = bb_upper
    df["bb_middle"] = bb_middle
    df["bb_lower"] = bb_lower
    df["bb_width"] = (bb_upper - bb_lower) / bb_middle  # Relative Breite
    df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)  # Position im Band (0-1)
    
    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt saisonale Features basierend auf Datum hinzu."""
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Wochentag (0=Montag, 6=Sonntag)
    df["day_of_week"] = df.index.dayofweek
    
    # Monat (1-12)
    df["month"] = df.index.month
    
    # Quartal (1-4)
    df["quarter"] = df.index.quarter
    
    # Tag des Monats (1-31)
    df["day_of_month"] = df.index.day
    
    # Ist Wochenende? (für Trading: Freitag-Effekt)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    
    # Monatsanfang/Ende (oft Volatilität)
    df["is_month_start"] = df.index.is_month_start.astype(int)
    df["is_month_end"] = df.index.is_month_end.astype(int)
    
    return df


def add_lag_features(df: pd.DataFrame, max_lag: int = 20) -> pd.DataFrame:
    """Fügt erweiterte Lag-Features hinzu."""
    df = df.copy()
    close = df["close"]
    
    # Standard Lags (1, 2, 5, 10)
    for lag in [1, 2, 3, 5, 7, 10, 14, 20]:
        if lag <= max_lag:
            df[f"lag_{lag}"] = close.shift(lag)
    
    # Lag Returns (Preisänderungen)
    for lag in [1, 2, 5]:
        df[f"return_lag_{lag}"] = np.log(close).diff(lag)
    
    # Lag Volumes
    if "volume" in df.columns:
        for lag in [1, 2, 5]:
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
    
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt erweiterte Rolling-Statistiken hinzu."""
    df = df.copy()
    close = df["close"]
    
    # Rolling Means (verschiedene Fenster)
    for window in [3, 5, 7, 10, 14, 20, 30, 50]:
        df[f"roll_mean_{window}"] = close.rolling(window=window).mean()
        df[f"roll_std_{window}"] = close.rolling(window=window).std()
        df[f"roll_min_{window}"] = close.rolling(window=window).min()
        df[f"roll_max_{window}"] = close.rolling(window=window).max()
    
    # Rolling Returns (Volatilität)
    returns = np.log(close).diff()
    for window in [5, 10, 20]:
        df[f"roll_return_mean_{window}"] = returns.rolling(window=window).mean()
        df[f"roll_return_std_{window}"] = returns.rolling(window=window).std()
    
    # Price Position in Rolling Range (0-1)
    for window in [10, 20]:
        roll_min = close.rolling(window=window).min()
        roll_max = close.rolling(window=window).max()
        df[f"price_position_{window}"] = (close - roll_min) / (roll_max - roll_min)
    
    # Exponential Moving Averages
    for span in [7, 12, 26, 50]:
        df[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()
    
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt Volatilitäts-Features hinzu."""
    df = df.copy()
    close = df["close"]
    returns = np.log(close).diff()
    
    # Realized Volatility (Standard Deviation der Returns)
    for window in [5, 10, 20, 30]:
        df[f"volatility_{window}"] = returns.rolling(window=window).std()
    
    # High-Low Range (Intraday Volatilität)
    if "high" in df.columns and "low" in df.columns:
        df["hl_range"] = df["high"] - df["low"]
        df["hl_range_pct"] = df["hl_range"] / df["close"]
        for window in [5, 10, 20]:
            df[f"hl_range_avg_{window}"] = df["hl_range"].rolling(window=window).mean()
    
    # True Range (für ATR-ähnliche Features)
    if "high" in df.columns and "low" in df.columns and "close" in df.columns:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close = np.abs(df["low"] - df["close"].shift(1))
        df["true_range"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        for window in [7, 14]:
            df[f"atr_{window}"] = df["true_range"].rolling(window=window).mean()
    
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt grundlegende Preis-Features hinzu."""
    df = df.copy()
    close = df["close"]
    
    # Returns
    df["return"] = np.log(close).diff()
    df["return_abs"] = np.abs(df["return"])
    
    # Price Change
    df["price_change"] = close.diff()
    df["price_change_pct"] = close.pct_change()
    
    # High-Low Features
    if "high" in df.columns and "low" in df.columns:
        df["hl_ratio"] = df["high"] / df["low"]
        df["close_to_high"] = df["close"] / df["high"]
        df["close_to_low"] = df["close"] / df["low"]
    
    # Open-Close Features
    if "open" in df.columns:
        df["oc_change"] = df["close"] - df["open"]
        df["oc_change_pct"] = (df["close"] - df["open"]) / df["open"]
    
    return df


def create_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hauptfunktion: Erstellt alle Features in der richtigen Reihenfolge.
    
    Args:
        df: DataFrame mit Spalten: Date (als Index), open, high, low, close, volume
    
    Returns:
        DataFrame mit allen erstellten Features
    """
    df = df.copy()
    
    # Sicherstellen, dass Index ein DatetimeIndex ist
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index("Date")
        elif "datetime" in df.columns:
            df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index)
    
    # Sortiere nach Datum
    df = df.sort_index()
    
    # 1. Grundlegende Preis-Features
    df = add_price_features(df)
    
    # 2. Lag Features
    df = add_lag_features(df, max_lag=20)
    
    # 3. Rolling Features
    df = add_rolling_features(df)
    
    # 4. Volatilitäts-Features
    df = add_volatility_features(df)
    
    # 5. Technische Indikatoren
    df = add_technical_indicators(df)
    
    # 6. Saisonale Features
    df = add_seasonal_features(df)
    
    # 7. Targets (für Training)
    df["y_day"] = df["close"].shift(-1)
    df["y_week"] = df["close"].shift(-5)
    
    # Entferne NaN-Werte (durch Rolling Windows und Lags entstanden)
    df = df.dropna()
    
    return df


def get_feature_columns(df: pd.DataFrame, exclude_targets: bool = True) -> list[str]:
    """
    Gibt eine Liste aller Feature-Spalten zurück (ohne Targets).
    
    Args:
        df: DataFrame mit Features
        exclude_targets: Wenn True, werden y_day und y_week ausgeschlossen
    
    Returns:
        Liste von Feature-Spalten-Namen
    """
    exclude = ["y_day", "y_week"] if exclude_targets else []
    exclude.extend(["open", "high", "low", "close", "volume"])  # Original-Spalten
    
    feature_cols = [col for col in df.columns if col not in exclude]
    return feature_cols


if __name__ == "__main__":
    # Test der Pipeline
    import sys
    from pathlib import Path
    
    data_path = Path("gold_GCF_3y_1d.csv")
    if not data_path.exists():
        print(f"❌ {data_path} nicht gefunden. Bitte zuerst fetch_gold.py ausführen.")
        sys.exit(1)
    
    print("📊 Lade Daten...")
    df = pd.read_csv(data_path)
    
    if "Date" in df.columns:
        df = df.set_index("Date")
    elif "datetime" in df.columns:
        df = df.set_index("datetime")
    else:
        df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)
    
    print(f"   Original: {len(df)} Zeilen, {len(df.columns)} Spalten")
    
    print("\n🔧 Erstelle Features...")
    df_features = create_feature_pipeline(df)
    
    feature_cols = get_feature_columns(df_features)
    print(f"   Nach Feature-Engineering: {len(df_features)} Zeilen, {len(df_features.columns)} Spalten")
    print(f"   Anzahl Features: {len(feature_cols)}")
    
    print("\n📋 Feature-Kategorien:")
    categories = {
        "Technische Indikatoren": [c for c in feature_cols if any(x in c for x in ["rsi", "macd", "bb_"])],
        "Lags": [c for c in feature_cols if "lag" in c or "return_lag" in c],
        "Rolling Stats": [c for c in feature_cols if "roll_" in c or "ema_" in c],
        "Volatilität": [c for c in feature_cols if "volatility" in c or "atr" in c or "hl_range" in c],
        "Saisonal": [c for c in feature_cols if any(x in c for x in ["day_of", "month", "quarter", "is_"])],
        "Preis-Features": [c for c in feature_cols if any(x in c for x in ["return", "price_", "close_to", "oc_"])],
    }
    
    for cat, cols in categories.items():
        if cols:
            print(f"   {cat}: {len(cols)} Features")
            print(f"      Beispiele: {', '.join(cols[:5])}")
    
    print("\n💾 Speichere Features...")
    output_path = Path("gold_GCF_features.csv")
    df_features.to_csv(output_path, index=True)
    print(f"   ✅ Gespeichert: {output_path.resolve()}")
    print(f"   Shape: {df_features.shape}")
