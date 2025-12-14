"""
Preprocess raw gold price JSON files (from yfinance) into a CSV in S3.

- Reads raw files under raw/YYYY/MM/DD/ (prefix configurable)
- Extracts rows per ticker: datetime,ticker,open,high,low,close,volume
- Filters to the given process date (event["process_date"] or today UTC)
- Deduplicates by (ticker, datetime)
- Writes CSV to processed/daily/gold_prices_YYYYMMDD.csv (prefix configurable)
"""

import csv
import json
import os
from datetime import datetime, timezone
from io import StringIO
from typing import Dict, List, Tuple

import boto3

s3 = boto3.client("s3")


def parse_process_date(event) -> Tuple[datetime, bool]:
    """
    Returns the process date (UTC).
    - If event contains {"process_date": "YYYY-MM-DD"} use that.
    - Otherwise, use today (UTC).
    """
    if event and isinstance(event, dict) and "process_date" in event:
        process_date_str = event["process_date"]
        try:
            return datetime.strptime(process_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc), True
        except ValueError as exc:
            raise ValueError(f"Invalid process_date format, expected YYYY-MM-DD, got: {process_date_str}") from exc
    return datetime.now(timezone.utc), False


def lambda_handler(event, context):
    bucket_name = os.environ["BUCKET_NAME"]
    raw_prefix = os.environ.get("RAW_PREFIX", "raw/")
    processed_prefix = os.environ.get("PROCESSED_PREFIX", "processed/daily/")

    process_date, process_date_from_event = parse_process_date(event)
    year = process_date.strftime("%Y")
    month = process_date.strftime("%m")
    day = process_date.strftime("%d")
    date_str = process_date.strftime("%Y-%m-%d")
    date_compact = process_date.strftime("%Y%m%d")

    # raw/YYYY/MM/DD/
    day_prefix = f"{raw_prefix}{year}/{month}/{day}/"
    print(f"[INFO] Processing raw files from prefix: s3://{bucket_name}/{day_prefix}")

    # Collect all raw JSON keys for the day
    raw_keys: List[str] = []
    continuation_token = None
    while True:
        list_kwargs = {"Bucket": bucket_name, "Prefix": day_prefix}
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token
        resp = s3.list_objects_v2(**list_kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"):
                raw_keys.append(key)
        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break

    if not raw_keys:
        msg = f"No raw files found for {date_str} under prefix {day_prefix}"
        print(f"[WARN] {msg}")
        return {"statusCode": 200, "body": json.dumps({"message": msg})}

    # Dedup by (ticker, datetime) after we know the target date
    candidate_rows: List[Dict] = []

    for key in raw_keys:
        print(f"[INFO] Reading raw file: {key}")
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        body = obj["Body"].read().decode("utf-8")
        payload = json.loads(body)

        data = payload.get("data", {})
        # data is expected: { "GC=F": [ { "datetime": "...", "Open": ..., "High": ..., "Low": ..., "Close": ..., "Volume": ... }, ... ] }
        for ticker, rows in data.items():
            if not isinstance(rows, list):
                continue
            ticker_up = str(ticker).upper()
            for row in rows:
                dt_str = row.get("datetime")
                if not dt_str:
                    continue
                candidate_rows.append(
                    {
                        "datetime": dt_str,
                        "ticker": ticker_up,
                        "open": row.get("Open"),
                        "high": row.get("High"),
                        "low": row.get("Low"),
                        "close": row.get("Close"),
                        "volume": row.get("Volume"),
                    }
                )

    # Decide which date to process: preferred process_date; if none found and not forced by event, fall back to latest available
    rows_by_key: Dict[Tuple[str, str], Dict] = {}
    available_dates = {r["datetime"][:10] for r in candidate_rows}
    selected_date = date_str
    if not process_date_from_event and selected_date not in available_dates and available_dates:
        selected_date = max(available_dates)
        print(f"[INFO] No rows for {date_str}, falling back to latest available date: {selected_date}")
        date_compact = selected_date.replace("-", "")

    for row in candidate_rows:
        if row["datetime"][:10] != selected_date:
            continue
        key_tuple = (row["ticker"], row["datetime"])
        rows_by_key[key_tuple] = row

    if not rows_by_key:
        msg = f"No matching rows for {selected_date} from raw files."
        print(f"[WARN] {msg}")
        return {"statusCode": 200, "body": json.dumps({"message": msg})}

    # Build CSV in-memory
    output = StringIO()
    fieldnames = ["datetime", "ticker", "open", "high", "low", "close", "volume"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for (_, _), row in sorted(rows_by_key.items(), key=lambda kv: kv[0]):
        writer.writerow(row)
    csv_data = output.getvalue()
    output.close()

    out_key = f"{processed_prefix}gold_prices_{date_compact}.csv"
    s3.put_object(
        Bucket=bucket_name,
        Key=out_key,
        Body=csv_data.encode("utf-8"),
        ContentType="text/csv",
    )

    msg = f"Processed {len(rows_by_key)} rows for {date_str} into {out_key}"
    print(f"[INFO] {msg}")
    return {"statusCode": 200, "body": json.dumps({"message": msg, "output_key": out_key})}
