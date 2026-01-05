#!/usr/bin/env python3
"""
Lokales Test-Skript für die Lambda-Funktion
Simuliert die Lambda-Umgebung für lokales Testen
"""

import os
import json
from datetime import datetime, timezone

# Setze Umgebungsvariablen (wie in Lambda)
os.environ["BUCKET_NAME"] = "test-bucket"  # Wird nicht wirklich verwendet
os.environ["TICKERS"] = "GC=F"
os.environ["LOOKBACK_DAYS"] = "7"
os.environ["FALLBACK_LOOKBACK_DAYS"] = "30"
os.environ["MAX_LOOKBACK_DAYS"] = "90"
os.environ["INTERVAL"] = "1d"
os.environ["S3_PREFIX"] = "raw"

# Mock Lambda Context
class MockContext:
    def __init__(self):
        self.function_name = "test-function"
        self.function_version = "$LATEST"
        self.invoked_function_arn = "arn:aws:lambda:eu-central-1:123456789012:function:test"
        self.memory_limit_in_mb = 512
        self.aws_request_id = "test-request-id"
        self.log_group_name = "/aws/lambda/test"
        self.log_stream_name = "test-stream"
        self._start_time = datetime.now(timezone.utc)
    
    def get_remaining_time_in_millis(self):
        elapsed = (datetime.now(timezone.utc) - self._start_time).total_seconds() * 1000
        return max(300000 - elapsed, 0)  # 5 Minuten

# Import Lambda-Funktion
try:
    from lamda_data_collection import lambda_handler, fetch_prices, test_connectivity
except ImportError:
    print("ERROR: Kann lamda_data_collection nicht importieren")
    print("Stelle sicher, dass lamda_data_collection.py im gleichen Verzeichnis ist")
    exit(1)

def test_connectivity_check():
    """Testet die Connectivity-Funktion"""
    print("\n" + "="*60)
    print("TEST 1: Connectivity Check")
    print("="*60)
    result = test_connectivity()
    if result:
        print("✅ Connectivity test erfolgreich")
    else:
        print("❌ Connectivity test fehlgeschlagen")
        print("   Prüfen Sie Ihre Internet-Verbindung")
    return result

def test_fetch_prices():
    """Testet die fetch_prices Funktion"""
    print("\n" + "="*60)
    print("TEST 2: Fetch Prices (ohne S3 Upload)")
    print("="*60)
    try:
        tickers = ["GC=F"]
        data = fetch_prices(
            tickers=tickers,
            days=7,
            interval="1d",
            fallback_days=30,
            max_days=90,
        )
        
        if data and any(v for v in data.values() if isinstance(v, list) and len(v) > 0):
            print("✅ Daten erfolgreich abgerufen")
            for ticker, records in data.items():
                print(f"   Ticker: {ticker}, Anzahl Records: {len(records)}")
                if records:
                    print(f"   Erster Record: {records[0]}")
                    print(f"   Letzter Record: {records[-1]}")
            return True
        else:
            print("❌ Keine Daten abgerufen")
            print(f"   Data: {data}")
            return False
    except Exception as e:
        print(f"❌ Fehler beim Abrufen der Daten: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_lambda_handler():
    """Testet die Lambda Handler Funktion (mit Mock S3)"""
    print("\n" + "="*60)
    print("TEST 3: Lambda Handler (mit Mock S3)")
    print("="*60)
    
    # Mock S3 Client
    class MockS3:
        def __init__(self):
            self.objects = {}
        
        def put_object(self, Bucket, Key, Body, ContentType):
            self.objects[Key] = {
                "body": Body,
                "content_type": ContentType,
            }
            print(f"   [MOCK S3] Uploaded: s3://{Bucket}/{Key}")
            print(f"   [MOCK S3] Size: {len(Body)} bytes")
    
    # Ersetze S3 Client temporär
    import lamda_data_collection
    original_s3 = lamda_data_collection.s3
    mock_s3 = MockS3()
    lamda_data_collection.s3 = mock_s3
    
    try:
        event = {}
        context = MockContext()
        
        result = lambda_handler(event, context)
        
        print(f"   Status Code: {result['statusCode']}")
        body = json.loads(result['body'])
        print(f"   Message: {body.get('message', 'N/A')}")
        print(f"   S3 Key: {body.get('s3_key', 'N/A')}")
        print(f"   Has Data: {body.get('has_data', 'N/A')}")
        
        # Prüfe ob Daten in Mock S3 gespeichert wurden
        if mock_s3.objects:
            key = list(mock_s3.objects.keys())[0]
            s3_data = json.loads(mock_s3.objects[key]['body'])
            print(f"\n   S3 Data Preview:")
            print(f"   - Timestamp: {s3_data.get('timestamp_utc')}")
            print(f"   - Tickers: {s3_data.get('tickers')}")
            if 'data' in s3_data:
                data = s3_data['data']
                if isinstance(data, dict):
                    for ticker, records in data.items():
                        if isinstance(records, list):
                            print(f"   - {ticker}: {len(records)} records")
                        else:
                            print(f"   - {ticker}: {records}")
        
        if result['statusCode'] == 200:
            print("✅ Lambda Handler erfolgreich")
            return True
        else:
            print("⚠️  Lambda Handler mit Warnung (Status Code != 200)")
            return False
    except Exception as e:
        print(f"❌ Fehler im Lambda Handler: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    finally:
        # Stelle originalen S3 Client wieder her
        lamda_data_collection.s3 = original_s3

def main():
    """Führt alle Tests aus"""
    print("\n" + "="*60)
    print("LAMBDA FUNCTION LOCAL TEST")
    print("="*60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    
    results = []
    
    # Test 1: Connectivity
    results.append(("Connectivity", test_connectivity_check()))
    
    # Test 2: Fetch Prices
    results.append(("Fetch Prices", test_fetch_prices()))
    
    # Test 3: Lambda Handler
    results.append(("Lambda Handler", test_lambda_handler()))
    
    # Zusammenfassung
    print("\n" + "="*60)
    print("TEST ZUSAMMENFASSUNG")
    print("="*60)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALLE TESTS BESTANDEN")
        print("Die Lambda-Funktion sollte in AWS funktionieren.")
    else:
        print("❌ EINIGE TESTS FEHLGESCHLAGEN")
        print("Bitte beheben Sie die Fehler vor dem Deployment.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())

