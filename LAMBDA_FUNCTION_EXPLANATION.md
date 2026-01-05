# Schritt-für-Schritt Erklärung: Lambda Data Collection Function

## Übersicht

Diese Lambda-Funktion holt täglich Gold-Preisdaten von Yahoo Finance und speichert sie in einem S3 Bucket. Die Funktion ist robust aufgebaut mit mehreren Fallback-Mechanismen und Retry-Logik.

---

## 📋 Gesamt-Ablauf

```
1. Lambda wird getriggert (EventBridge/Timer)
   ↓
2. lambda_handler() startet
   ↓
3. Konfiguration aus Umgebungsvariablen lesen
   ↓
4. fetch_prices() aufrufen
   ↓
5. Daten in S3 speichern
   ↓
6. Status zurückgeben
```

---

## 🔍 Detaillierte Schritt-für-Schritt Erklärung

### **PHASE 1: Initialisierung (Zeilen 1-19)**

```python
import json, os, time, datetime, traceback
import boto3, pandas, yfinance, requests
```

**Was passiert:**
- Alle benötigten Bibliotheken werden importiert
- S3 Client wird initialisiert (`boto3.client("s3")`)
- User-Agent wird definiert (wichtig, damit Yahoo Finance die Requests nicht blockiert)

**Warum wichtig:**
- `yfinance`: Holt Daten von Yahoo Finance
- `boto3`: Kommuniziert mit AWS S3
- `pandas`: Verarbeitet die Daten als DataFrames
- `requests`: Testet Internet-Verbindung

---

### **PHASE 2: Connectivity Test (Zeilen 21-29)**

```python
def test_connectivity() -> bool:
    response = requests.get("https://www.google.com", timeout=5)
    return response.status_code == 200
```

**Was passiert:**
1. Versucht eine Verbindung zu Google herzustellen
2. Prüft ob HTTP Status 200 zurückkommt
3. Gibt `True` zurück wenn erfolgreich, sonst `False`

**Warum wichtig:**
- Lambda-Funktionen in VPCs ohne NAT Gateway haben keinen Internet-Zugang
- Dieser Test zeigt sofort, ob das Problem die Netzwerk-Verbindung ist

---

### **PHASE 3: Hauptfunktion `fetch_prices()` (Zeilen 31-283)**

#### **Schritt 3.1: Vorbereitung (Zeilen 42-48)**

```python
if not test_connectivity():
    return {}  # Abbruch wenn kein Internet

joined = " ".join(tickers)  # z.B. "GC=F" oder "GC=F SPY"
```

**Was passiert:**
1. Connectivity-Test wird ausgeführt
2. Wenn kein Internet: Funktion bricht ab und gibt leeres Dict zurück
3. Ticker-Liste wird zu einem String zusammengefügt (für yfinance)

**Beispiel:**
- Input: `["GC=F", "SPY"]`
- Output: `"GC=F SPY"`

---

#### **Schritt 3.2: Hilfsfunktion `run_download_period()` (Zeilen 50-124)**

Diese Funktion versucht Daten für einen bestimmten Zeitraum zu holen.

**Ablauf:**

1. **Versuch 1 (Zeilen 53-67):**
   ```python
   # Setze User-Agent (wichtig!)
   yf_utils.get_user_agent = lambda: USER_AGENT
   
   # Lade Daten
   df = yf.download(
       tickers="GC=F",
       period="7d",      # Letzte 7 Tage
       interval="1d",    # Tägliche Daten
       threads=False     # Keine Parallelisierung (stabiler)
   )
   ```

2. **Erfolg? (Zeilen 69-74):**
   - Wenn DataFrame nicht leer: ✅ Daten zurückgeben
   - Loggt: Shape, Spalten, Datumsbereich

3. **Leer? (Zeilen 76-82):**
   - Wenn leer UND nicht letzter Versuch:
     - Warte 30s (Versuch 1) oder 60s (Versuch 2)
     - Versuche es erneut

4. **JSONDecodeError? (Zeilen 83-93):**
   - **Das ist Ihr Fehler!** Yahoo Finance gibt leere Antwort zurück
   - Mögliche Ursachen:
     - Rate Limiting (zu viele Requests)
     - Netzwerk-Problem
     - Yahoo Finance blockiert den Request
   - **Retry-Logik:**
     - Versuch 1: Warte 15s → Retry
     - Versuch 2: Warte 30s → Retry
     - Versuch 3: ❌ Fehler, gebe leeres DataFrame zurück

5. **Andere Fehler? (Zeilen 94-123):**
   - Prüft ob es ein Rate-Limit-Fehler ist (429, "Too Many Requests")
   - Wenn Rate-Limit: Retry mit Wartezeit
   - Wenn Timeout: Retry mit kürzerer Wartezeit
   - Sonst: ❌ Fehler, gebe leeres DataFrame zurück

**Warum diese Struktur:**
- **Retry-Mechanismus**: Yahoo Finance kann temporär nicht erreichbar sein
- **Kurze Wartezeiten**: Lambda hat nur 3 Minuten Timeout
- **Detailliertes Logging**: Jeder Schritt wird geloggt für Debugging

---

#### **Schritt 3.3: Hilfsfunktion `run_download_range()` (Zeilen 126-194)**

Ähnlich wie `run_download_period()`, aber mit explizitem Datumsbereich.

**Unterschied:**
- `run_download_period()`: `period="7d"` (relativ zu heute)
- `run_download_range()`: `start="2024-01-01", end="2024-01-08"` (absolut)

**Wann verwendet:**
- Als Fallback, wenn period-basierte Downloads fehlschlagen

---

#### **Schritt 3.4: Fallback-Strategie (Zeilen 196-271)**

Die Funktion versucht mehrere Methoden, um Daten zu bekommen:

**Fallback 1: Primary Download (Zeile 196)**
```python
df = run_download_period(days, "primary")
# z.B. 7 Tage
```

**Fallback 2: Erweiterter Zeitraum (Zeilen 199-200)**
```python
if df.empty and fallback_days > days:
    df = run_download_period(fallback_days, "fallback_env")
# z.B. 30 Tage statt 7
```

**Fallback 3: Maximaler Zeitraum (Zeilen 203-204)**
```python
if df.empty and max_days > fallback_days:
    df = run_download_period(max_days, "fallback_max_days")
# z.B. 90 Tage
```

**Fallback 4: Expliziter Datumsbereich (Zeilen 207-210)**
```python
if df.empty:
    start = heute - 90 Tage
    end = heute
    df = run_download_range("fallback_range_3m", start, end)
```

**Fallback 5: Ticker.history() Methode (Zeilen 213-271)**
```python
if df.empty:
    for ticker in tickers:
        ticker_obj = yf.Ticker("GC=F")
        hist = ticker_obj.history(period="3mo")
        # Alternative API-Methode
```

**Warum so viele Fallbacks?**
- Yahoo Finance API kann unzuverlässig sein
- Verschiedene Methoden haben unterschiedliche Erfolgsraten
- Maximiert die Chance, Daten zu bekommen

---

#### **Schritt 3.5: Daten-Transformation (Zeilen 273-283)**

```python
out = {}
for ticker in tickers:
    # Extrahiere Daten für diesen Ticker
    if MultiIndex-Spalten:
        sub = df.xs(ticker, axis=1, level=1)  # MultiIndex auflösen
    else:
        sub = df
    
    # Formatiere für JSON
    sub = sub.reset_index()
    sub["datetime"] = sub["datetime"].astype(str)
    out[ticker] = sub.to_dict(orient="records")
```

**Was passiert:**
1. Für jeden Ticker werden die Daten extrahiert
2. DataFrame wird in ein Dictionary konvertiert
3. Datum wird zu String konvertiert (JSON-kompatibel)

**Beispiel Output:**
```json
{
  "GC=F": [
    {
      "datetime": "2024-01-01",
      "Open": 2050.0,
      "High": 2060.0,
      "Low": 2045.0,
      "Close": 2055.0,
      "Volume": 1000000
    },
    ...
  ]
}
```

---

### **PHASE 4: Lambda Handler `lambda_handler()` (Zeilen 285-373)**

Dies ist die Hauptfunktion, die von AWS Lambda aufgerufen wird.

#### **Schritt 4.1: Initialisierung (Zeilen 287-288)**

```python
print(f"[INFO] Lambda started. Event: {json.dumps(event)}")
print(f"[INFO] Remaining time: {context.get_remaining_time_in_millis() / 1000:.1f}s")
```

**Was passiert:**
- Loggt den Event (Trigger-Informationen)
- Zeigt verbleibende Zeit (Lambda hat Timeout-Limit)

---

#### **Schritt 4.2: Konfiguration lesen (Zeilen 290-310)**

```python
bucket_name = os.environ["BUCKET_NAME"]  # Zwingend erforderlich
tickers_env = os.environ.get("TICKERS", "GC=F")  # Default: "GC=F"
days = int(os.environ.get("LOOKBACK_DAYS", "7"))
interval = os.environ.get("INTERVAL", "1d")
prefix = os.environ.get("S3_PREFIX", "raw")
```

**Was passiert:**
1. Liest Umgebungsvariablen aus Lambda-Konfiguration
2. Setzt Defaults falls nicht gesetzt
3. Konvertiert Strings zu Integers wo nötig

**Beispiel Konfiguration:**
- `BUCKET_NAME`: `gold-price-data-bucket`
- `TICKERS`: `GC=F,SPY` (mehrere mit Komma)
- `LOOKBACK_DAYS`: `7`
- `INTERVAL`: `1d` (täglich)

---

#### **Schritt 4.3: S3 Key generieren (Zeilen 312-315)**

```python
now = datetime.now(timezone.utc)
ts_str = now.strftime("%Y%m%dT%H%M%SZ")  # z.B. "20240103T113000Z"
date_prefix = now.strftime("%Y/%m/%d")   # z.B. "2024/01/03"
s3_key = f"raw/2024/01/03/gold_prices_20240103T113000Z.json"
```

**Was passiert:**
- Erstellt einen eindeutigen Dateinamen mit Timestamp
- Organisiert Dateien nach Datum im S3 Bucket

**S3 Struktur:**
```
s3://bucket/
  raw/
    2024/
      01/
        03/
          gold_prices_20240103T113000Z.json
          gold_prices_20240103T120000Z.json
```

---

#### **Schritt 4.4: Daten abrufen (Zeilen 317-335)**

```python
try:
    data = fetch_prices(tickers, days=days, interval=interval, ...)
    
    if not data or all(not v for v in data.values()):
        data = {"error": "No data retrieved", "tickers": tickers}
except Exception as e:
    data = {"error": str(e), "traceback": traceback.format_exc()}
```

**Was passiert:**
1. Ruft `fetch_prices()` auf (siehe Phase 3)
2. Prüft ob Daten vorhanden sind
3. Bei Fehler: Erstellt Error-Dict mit Traceback

**Warum try/except:**
- Auch bei Fehlern soll die Funktion nicht abstürzen
- Fehler-Informationen werden in S3 gespeichert für späteres Debugging

---

#### **Schritt 4.5: Daten in S3 speichern (Zeilen 337-365)**

```python
body = {
    "timestamp_utc": ts_str,
    "interval": interval,
    "lookback_days": days,
    "tickers": tickers,
    "data": data  # Die eigentlichen Preisdaten
}

s3.put_object(
    Bucket=bucket_name,
    Key=s3_key,
    Body=json.dumps(body, default=str),
    ContentType="application/json"
)
```

**Was passiert:**
1. Erstellt JSON-Struktur mit Metadaten + Daten
2. Konvertiert zu JSON-String
3. Upload zu S3

**JSON-Struktur:**
```json
{
  "timestamp_utc": "20240103T113000Z",
  "interval": "1d",
  "lookback_days": 7,
  "tickers": ["GC=F"],
  "data": {
    "GC=F": [
      {"datetime": "2024-01-01", "Open": 2050.0, ...},
      ...
    ]
  }
}
```

---

#### **Schritt 4.6: Status zurückgeben (Zeilen 355-365)**

```python
has_data = data and any(v for v in data.values() if isinstance(v, list) and len(v) > 0)
status_code = 200 if has_data else 207  # 207 = Partial Success

return {
    "statusCode": status_code,
    "body": json.dumps({
        "message": "OK" if has_data else "Partial success - check data field",
        "s3_key": s3_key,
        "has_data": has_data
    })
}
```

**Was passiert:**
1. Prüft ob echte Daten vorhanden sind (nicht nur Error-Message)
2. Status Code:
   - `200`: ✅ Erfolg mit Daten
   - `207`: ⚠️ Teilweise erfolgreich (keine Daten, aber Datei erstellt)
   - `500`: ❌ Fehler (z.B. S3 Upload fehlgeschlagen)

---

## 🔄 Vollständiger Ablauf-Beispiel

### **Erfolgreicher Durchlauf:**

```
1. EventBridge triggert Lambda um 9:00 UTC
   ↓
2. lambda_handler() startet
   ↓
3. Liest BUCKET_NAME="gold-data", TICKERS="GC=F"
   ↓
4. Ruft fetch_prices(["GC=F"], days=7) auf
   ↓
5. test_connectivity() → ✅ Erfolg
   ↓
6. run_download_period(7, "primary")
   - Versuch 1: yf.download() → ✅ Erfolg, 7 Datensätze
   ↓
7. Daten transformieren zu Dict
   ↓
8. Zurück zu lambda_handler()
   ↓
9. S3 Upload: s3://gold-data/raw/2024/01/03/gold_prices_20240103T090000Z.json
   ↓
10. Return: statusCode=200, has_data=true
```

### **Fehlgeschlagener Durchlauf (Ihr aktuelles Problem):**

```
1. EventBridge triggert Lambda
   ↓
2. lambda_handler() startet
   ↓
3. Ruft fetch_prices() auf
   ↓
4. test_connectivity() → ✅ Erfolg (Internet funktioniert)
   ↓
5. run_download_period(7, "primary")
   - Versuch 1: yf.download() → ❌ JSONDecodeError
     "Expecting value: line 1 column 1 (char 0)"
     → Leere Antwort von Yahoo Finance
   - Warte 15s
   - Versuch 2: yf.download() → ❌ JSONDecodeError
   - Warte 30s
   - Versuch 3: yf.download() → ❌ JSONDecodeError
   → Leeres DataFrame zurückgegeben
   ↓
6. Fallback 1: run_download_period(30, "fallback_env")
   - Gleiche Fehler → Leeres DataFrame
   ↓
7. Fallback 2-4: Alle versuchen → Alle fehlgeschlagen
   ↓
8. Zurück zu lambda_handler() mit leerem Dict
   ↓
9. S3 Upload trotzdem (mit Error-Message)
   ↓
10. Return: statusCode=207, has_data=false
```

---

## 🐛 Ihr aktuelles Problem analysiert

**Fehler:** `JSONDecodeError('Expecting value: line 1 column 1 (char 0)')`

**Bedeutung:**
- Yahoo Finance gibt eine **leere Antwort** zurück
- Der JSON-Parser erwartet JSON, bekommt aber nichts

**Mögliche Ursachen:**

1. **VPC ohne NAT Gateway** (wahrscheinlichstes Problem)
   - Lambda hat keinen Internet-Zugang
   - Lösung: VPC-Konfiguration entfernen

2. **Yahoo Finance blockiert Requests**
   - User-Agent fehlt (wurde jetzt hinzugefügt)
   - Rate Limiting

3. **Yahoo Finance API ist temporär down**
   - Retry-Logik sollte helfen

**Was die verbesserte Version macht:**
- ✅ User-Agent wird gesetzt
- ✅ Connectivity-Test zeigt sofort Netzwerk-Probleme
- ✅ Kürzere Retry-Zeiten (15s statt 90s)
- ✅ Detailliertes Logging für Debugging

---

## 📊 Datenfluss-Diagramm

```
┌─────────────────┐
│ EventBridge     │
│ (Täglicher      │
│  Trigger)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ lambda_handler()│
│ - Liest Config  │
│ - Generiert S3  │
│   Key           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ fetch_prices()  │
│ - Connectivity  │
│   Test          │
│ - Primary       │
│   Download      │
│ - Fallbacks     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ run_download_   │
│ period()        │
│ - Retry-Logik   │
│ - Error-        │
│   Handling      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ yfinance API    │
│ (Yahoo Finance) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DataFrame       │
│ Transformation  │
│ → Dict          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ S3 Upload       │
│ (JSON-Datei)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Return Status   │
│ (200/207/500)   │
└─────────────────┘
```

---

## 💡 Wichtige Erkenntnisse

1. **Robustheit**: Mehrere Fallback-Strategien maximieren Erfolgschance
2. **Fehlerbehandlung**: Auch bei Fehlern wird eine Datei in S3 erstellt (für Debugging)
3. **Logging**: Jeder Schritt wird geloggt → CloudWatch Logs zeigen genau was passiert
4. **Retry-Logik**: Automatische Retries bei temporären Fehlern
5. **Timeout-Management**: Kurze Wartezeiten verhindern Lambda Timeouts

---

## 🔧 Nächste Schritte für Sie

1. **VPC prüfen**: Lambda muss Internet-Zugang haben
2. **Lokal testen**: `python test_lambda_local.py`
3. **Deployen**: Siehe `AWS_LAMBDA_DEPLOYMENT_GUIDE.md`
4. **Logs prüfen**: CloudWatch Logs nach Deployment

