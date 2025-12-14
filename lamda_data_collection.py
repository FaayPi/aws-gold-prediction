import json
import os
import time
from datetime import datetime, timezone

import boto3
import pandas as pd
import yfinance as yf

s3 = boto3.client("s3")

def fetch_prices(
    tickers: list[str],
    days: int = 7,  # Kleinere Defaults, um Rate-Limits zu vermeiden
    interval: str = "1d",
    fallback_days: int | None = 30,
    max_days: int = 90,
) -> dict:
    """
    Holt OHLCV für die letzten `days` Tage (rollierend) je Ticker.
    Gibt ein Dict ticker -> list of dict rows zurück.
    """
    joined = " ".join(tickers)

    def run_download_period(period_days: int, label: str, max_retries: int = 2):
        """Download mit Retry bei 429 (Rate Limit)"""
        for attempt in range(max_retries + 1):
            try:
                df_local = yf.download(
                    tickers=joined,
                    period=f"{period_days}d",
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    group_by="ticker",
                    threads=False,  # stabiler in Lambda
                )
                print(f"[DEBUG] {label} shape={df_local.shape} period={period_days}d interval={interval}")
                if not df_local.empty:
                    print(f"[DEBUG] {label} columns: {list(df_local.columns)}")
                    idx = df_local.index
                    if len(idx) > 0:
                        print(f"[DEBUG] {label} date range: {idx.min()} -> {idx.max()}")
                    return df_local
                else:
                    print(f"[WARN] {label} returned empty DataFrame")
                    # Wenn leer und nicht letzter Versuch, warte kurz
                    if attempt < max_retries:
                        wait_time = (attempt + 1) * 30  # 30s, 60s
                        print(f"[INFO] {label} retry {attempt + 1}/{max_retries} after {wait_time}s")
                        time.sleep(wait_time)
                    return df_local
            except json.JSONDecodeError as e:
                # JSONDecodeError deutet auf leere/ungültige Antwort hin (Rate Limit)
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 90  # 90s, 180s bei JSON-Fehler
                    print(f"[WARN] {label} JSON decode error (likely rate limit), retry {attempt + 1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] {label} JSON decode error after {max_retries} retries")
                    return pd.DataFrame()
            except Exception as e:
                error_str = str(e)
                # Prüfe auf Rate-Limit-Indikatoren
                is_rate_limit = (
                    "429" in error_str
                    or "Too Many Requests" in error_str
                    or "JSONDecodeError" in error_str
                    or "Expecting value" in error_str
                )
                if is_rate_limit:
                    if attempt < max_retries:
                        wait_time = (attempt + 1) * 90  # 90s, 180s bei Rate Limit
                        print(f"[WARN] {label} rate limited, retry {attempt + 1}/{max_retries} after {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        print(f"[ERROR] {label} rate limited after {max_retries} retries")
                        return pd.DataFrame()
                else:
                    print(f"[WARN] {label} error: {e}")
                    return pd.DataFrame()
        return pd.DataFrame()

    def run_download_range(label: str, start: str, end: str, max_retries: int = 2):
        """Download mit Retry bei 429 (Rate Limit)"""
        for attempt in range(max_retries + 1):
            try:
                df_local = yf.download(
                    tickers=joined,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    group_by="ticker",
                    threads=False,
                )
                print(f"[DEBUG] {label} shape={df_local.shape} start={start} end={end} interval={interval}")
                if not df_local.empty:
                    print(f"[DEBUG] {label} columns: {list(df_local.columns)}")
                    idx = df_local.index
                    if len(idx) > 0:
                        print(f"[DEBUG] {label} date range: {idx.min()} -> {idx.max()}")
                    return df_local
                else:
                    print(f"[WARN] {label} returned empty DataFrame")
                    if attempt < max_retries:
                        wait_time = (attempt + 1) * 30
                        print(f"[INFO] {label} retry {attempt + 1}/{max_retries} after {wait_time}s")
                        time.sleep(wait_time)
                    return df_local
            except json.JSONDecodeError as e:
                # JSONDecodeError deutet auf leere/ungültige Antwort hin (Rate Limit)
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 90  # 90s, 180s bei JSON-Fehler
                    print(f"[WARN] {label} JSON decode error (likely rate limit), retry {attempt + 1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] {label} JSON decode error after {max_retries} retries")
                    return pd.DataFrame()
            except Exception as e:
                error_str = str(e)
                # Prüfe auf Rate-Limit-Indikatoren
                is_rate_limit = (
                    "429" in error_str
                    or "Too Many Requests" in error_str
                    or "JSONDecodeError" in error_str
                    or "Expecting value" in error_str
                )
                if is_rate_limit:
                    if attempt < max_retries:
                        wait_time = (attempt + 1) * 90  # 90s, 180s bei Rate Limit
                        print(f"[WARN] {label} rate limited, retry {attempt + 1}/{max_retries} after {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        print(f"[ERROR] {label} rate limited after {max_retries} retries")
                        return pd.DataFrame()
                else:
                    print(f"[WARN] {label} error: {e}")
                    return pd.DataFrame()
        return pd.DataFrame()

    df = run_download_period(days, "primary")

    # Fallback 1: fallback_days (env)
    if df.empty and fallback_days and fallback_days > days:
        df = run_download_period(fallback_days, "fallback_env")

    # Fallback 2: max_days (größerer Zeitraum, z.B. 1825d)
    if df.empty and max_days > max(days, fallback_days or 0):
        df = run_download_period(max_days, "fallback_max_days")

    # Fallback 3: expliziter Datumsbereich (3 Monate rückwärts)
    if df.empty:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start_str = (datetime.now(timezone.utc) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        df = run_download_range("fallback_range_3m", start=start_str, end=today_str)

    # Fallback 4: Ticker.history() pro Symbol (period=3mo) mit Retry
    if df.empty:
        merged = []
        for t in tickers:
            for attempt in range(3):  # max 3 Versuche
                try:
                    hist = yf.Ticker(t).history(
                        period="3mo",  # 3 Monate statt 1 Jahr
                        interval=interval,
                        auto_adjust=False,
                    )
                    print(f"[DEBUG] history {t} shape={hist.shape} period=3mo interval={interval}")
                    if not hist.empty:
                        hist = hist.reset_index().rename(columns={"Date": "datetime"})
                        hist["datetime"] = hist["datetime"].astype(str)
                        hist["ticker"] = t
                        merged.append(hist)
                        break  # Erfolg, keine weiteren Versuche
                    else:
                        print(f"[WARN] history {t} empty")
                        if attempt < 2:
                            time.sleep((attempt + 1) * 30)
                except json.JSONDecodeError as e:
                    # JSONDecodeError deutet auf Rate Limit
                    if attempt < 2:
                        wait_time = (attempt + 1) * 90
                        print(f"[WARN] history {t} JSON decode error (likely rate limit), retry {attempt + 1}/2 after {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        print(f"[ERROR] history {t} JSON decode error after retries")
                except Exception as e:
                    error_str = str(e)
                    is_rate_limit = (
                        "429" in error_str
                        or "Too Many Requests" in error_str
                        or "JSONDecodeError" in error_str
                        or "Expecting value" in error_str
                    )
                    if is_rate_limit:
                        if attempt < 2:
                            wait_time = (attempt + 1) * 90
                            print(f"[WARN] history {t} rate limited, retry {attempt + 1}/2 after {wait_time}s")
                            time.sleep(wait_time)
                        else:
                            print(f"[ERROR] history {t} rate limited after retries")
                    else:
                        print(f"[WARN] history {t} error: {e}")
                        break  # Anderer Fehler, keine Retries
        if merged:
            df = pd.concat(merged, ignore_index=False)
        else:
            print("[WARN] history() returned empty for all tickers")

    out: dict[str, list[dict]] = {}
    # yfinance liefert MultiIndex, wir vereinfachen auf jeden Ticker
    for ticker in tickers:
        if isinstance(df.columns, pd.MultiIndex):
            sub = df.xs(ticker, axis=1, level=1)
        else:
            sub = df
        sub = sub.reset_index().rename(columns={"Date": "datetime"})
        sub["datetime"] = sub["datetime"].astype(str)
        out[ticker] = sub.to_dict(orient="records")
    return out

def lambda_handler(event, context):
    # Env Variablen
    bucket_name = os.environ["BUCKET_NAME"]
    tickers_env = os.environ.get("TICKERS", "GC=F")  # mehrere mit Komma möglich
    days = int(os.environ.get("LOOKBACK_DAYS", "7"))  # Kleinere Defaults
    fallback_days = int(os.environ.get("FALLBACK_LOOKBACK_DAYS", "30"))
    max_days = int(os.environ.get("MAX_LOOKBACK_DAYS", "90"))
    interval = os.environ.get("INTERVAL", "1d")      # z.B. 1d, 1h, 4h
    prefix = os.environ.get("S3_PREFIX", "raw")

    tickers = [t.strip() for t in tickers_env.split(",") if t.strip()]

    now = datetime.now(timezone.utc)
    ts_str = now.strftime("%Y%m%dT%H%M%SZ")
    date_prefix = now.strftime("%Y/%m/%d")
    s3_key = f"{prefix}/{date_prefix}/gold_prices_{ts_str}.json"

    try:
        data = fetch_prices(
            tickers,
            days=days,
            interval=interval,
            fallback_days=fallback_days,
            max_days=max_days,
        )
    except Exception as e:
        # Im Fehlerfall trotzdem S3 schreiben, damit Logs nicht verloren gehen
        data = {"error": str(e), "tickers": tickers}

    body = {
        "timestamp_utc": ts_str,
        "interval": interval,
        "lookback_days": days,
        "tickers": tickers,
        "data": data,
    }

    s3.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=json.dumps(body),
        ContentType="application/json",
    )

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "OK", "s3_key": s3_key}),
    }
    