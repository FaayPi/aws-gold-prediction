"""
Download Gold prices from Yahoo Finance and save to CSV.
Symbols:
  GC=F      - Gold Futures (USD/oz)
  XAUUSD=X  - Gold Spot USD
  XAUEUR=X  - Gold Spot EUR

Usage:
    python fetch_gold.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
import argparse

import pandas as pd
import yfinance as yf


def fetch_gold_gc(
    years: int = 10,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download historical data for GC=F (Gold Futures USD/oz).

    years:    wie viele Jahre Historie (Standard: 10)
    interval: z.B. 1d, 1h, 4h (Yahoo Finance Interval)
    """
    sym = "GC=F"
    print(f"\nDownloading {sym} …")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    df = yf.download(
        sym,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
        threads=False,
    )
    print(f"Rows: {len(df)}")
    if df.empty:
        print("DataFrame empty (evtl. Yahoo-Rate-Limit oder Netzwerkblock).")
        return df

    print("Date range:", df.index[0].date(), "to", df.index[-1].date())
    print("Columns:", list(df.columns))

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([c for c in cols if c]) for cols in df.columns]

    # Keep only needed columns and rename (open/high/low/close/volume)
    keep_cols = [
        c
        for c in df.columns
        if c.lower().startswith("open")
        or c.lower().startswith("high")
        or c.lower().startswith("low")
        or c.lower().startswith("close")
        or c.lower().startswith("volume")
    ]
    df = df[keep_cols].copy()
    # Standardize names
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if "open" in cl:
            rename_map[c] = "open"
        elif "high" in cl:
            rename_map[c] = "high"
        elif "low" in cl:
            rename_map[c] = "low"
        elif "close" in cl:
            rename_map[c] = "close"
        elif "volume" in cl:
            rename_map[c] = "volume"
    df = df.rename(columns=rename_map)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=10, help="Anzahl Jahre Historie (Standard: 10)")
    parser.add_argument("--interval", type=str, default="1d", help="Interval z.B. 1d, 1h, 4h")
    parser.add_argument("--output", type=Path, default=None, help="Ausgabedatei (optional)")
    args = parser.parse_args()

    df = fetch_gold_gc(years=args.years, interval=args.interval)
    if df.empty:
        return
    default_name = f"gold_GCF_{args.years}y_{args.interval}.csv"
    out_path = args.output if args.output else Path(default_name)
    df.to_csv(out_path, index=True)
    print(f"✅ Gespeichert: {out_path.resolve()} ({len(df)} Zeilen)")


if __name__ == "__main__":
    main()

