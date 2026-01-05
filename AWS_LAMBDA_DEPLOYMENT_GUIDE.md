# AWS Lambda Deployment Guide - Gold Price Data Collection

## Übersicht

Diese Anleitung führt Sie Schritt für Schritt durch die Einrichtung einer Lambda-Funktion, die täglich Gold-Preisdaten von Yahoo Finance abruft und in S3 speichert.

## Schritt 1: S3 Bucket erstellen

1. Gehen Sie zur AWS S3 Console
2. Klicken Sie auf "Create bucket"
3. Bucket-Name: z.B. `gold-price-data-<your-name>` (muss global eindeutig sein)
4. Region: Wählen Sie die gleiche Region wie für Lambda (z.B. `eu-central-1`)
5. Block Public Access: Aktiviert lassen (Standard)
6. Klicken Sie auf "Create bucket"

## Schritt 2: Lambda-Funktion erstellen

### 2.1 Lambda-Funktion anlegen

1. Gehen Sie zur AWS Lambda Console
2. Klicken Sie auf "Create function"
3. Wählen Sie "Author from scratch"
4. Funktion-Name: `gold-price-data-collection`
5. Runtime: `Python 3.11` oder `Python 3.12`
6. Architecture: `x86_64`
7. Klicken Sie auf "Create function"

### 2.2 Code hochladen

**Option A: ZIP-Datei erstellen (empfohlen für Dependencies)**

1. Erstellen Sie ein Verzeichnis für das Lambda-Package:
```bash
mkdir lambda_package
cd lambda_package
```

2. Kopieren Sie die Lambda-Funktion:
```bash
cp ../lamda_data_collection.py lambda_function.py
```

3. Installieren Sie Dependencies in das Verzeichnis:
```bash
pip install -r ../requirements.txt -t .
```

4. Erstellen Sie eine ZIP-Datei:
```bash
zip -r lambda_function.zip .
```

5. In Lambda Console:
   - Gehen Sie zu "Code" Tab
   - Klicken Sie auf "Upload from" → ".zip file"
   - Wählen Sie die `lambda_function.zip` Datei
   - Klicken Sie auf "Save"

**Option B: Lambda Layer für Dependencies (besser für größere Pakete)**

1. Erstellen Sie ein Layer-Verzeichnis:
```bash
mkdir lambda_layer
cd lambda_layer
mkdir python
cd python
```

2. Installieren Sie Dependencies:
```bash
pip install yfinance pandas pyarrow requests -t .
```

3. Erstellen Sie ZIP:
```bash
cd ..
zip -r lambda_layer.zip python/
```

4. In Lambda Console:
   - Gehen Sie zu "Layers" (links im Menü)
   - Klicken Sie auf "Create layer"
   - Name: `gold-price-dependencies`
   - Upload die ZIP-Datei
   - Wählen Sie kompatible Runtimes (Python 3.11, 3.12)
   - Klicken Sie auf "Create"

5. Layer zur Funktion hinzufügen:
   - Gehen Sie zurück zur Lambda-Funktion
   - Scrollen Sie zu "Layers"
   - Klicken Sie auf "Add a layer"
   - Wählen Sie "Custom layers" → `gold-price-dependencies`
   - Klicken Sie auf "Add"

6. Code hochladen:
   - Kopieren Sie nur `lamda_data_collection.py` als `lambda_function.py`
   - In Lambda Console: Code direkt einfügen oder als ZIP hochladen

### 2.3 Handler konfigurieren

1. In Lambda Console → "Code" Tab
2. Scrollen Sie zu "Runtime settings"
3. Klicken Sie auf "Edit"
4. Handler: `lambda_function.lambda_handler` (wenn Datei `lambda_function.py` heißt)
   ODER: `lamda_data_collection.lambda_handler` (wenn Datei `lamda_data_collection.py` heißt)
5. Klicken Sie auf "Save"

## Schritt 3: Umgebungsvariablen setzen

1. In Lambda Console → "Configuration" Tab
2. Klicken Sie auf "Environment variables"
3. Klicken Sie auf "Edit"
4. Fügen Sie folgende Variablen hinzu:

| Key | Value | Beschreibung |
|-----|-------|--------------|
| `BUCKET_NAME` | `gold-price-data-<your-name>` | Name Ihres S3 Buckets |
| `TICKERS` | `GC=F` | Ticker-Symbol (mehrere mit Komma: `GC=F,SPY`) |
| `LOOKBACK_DAYS` | `7` | Anzahl Tage zurück |
| `FALLBACK_LOOKBACK_DAYS` | `30` | Fallback-Zeitraum |
| `MAX_LOOKBACK_DAYS` | `90` | Maximaler Zeitraum |
| `INTERVAL` | `1d` | Intervall (1d, 1h, 4h) |
| `S3_PREFIX` | `raw` | S3 Prefix für Dateien |

5. Klicken Sie auf "Save"

## Schritt 4: IAM-Berechtigungen konfigurieren

1. In Lambda Console → "Configuration" Tab
2. Klicken Sie auf "Permissions"
3. Klicken Sie auf die Role (z.B. `gold-price-data-collection-role-xxx`)
4. In IAM Console öffnet sich die Role
5. Klicken Sie auf "Add permissions" → "Create inline policy"
6. Wählen Sie "JSON" Tab
7. Fügen Sie folgende Policy ein:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::gold-price-data-<your-name>/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": "arn:aws:s3:::gold-price-data-<your-name>"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}
```

8. Ersetzen Sie `<your-name>` mit Ihrem Bucket-Namen
9. Klicken Sie auf "Review policy"
10. Name: `LambdaS3Access`
11. Klicken Sie auf "Create policy"

## Schritt 5: VPC-Konfiguration prüfen

**WICHTIG: Lambda muss Internet-Zugang haben!**

1. In Lambda Console → "Configuration" Tab
2. Klicken Sie auf "VPC"
3. **Wenn "No VPC" angezeigt wird**: ✅ Gut! Lambda hat Internet-Zugang
4. **Wenn ein VPC konfiguriert ist**: 
   - Lambda braucht entweder:
     - **NAT Gateway** im VPC (kostet ~$32/Monat)
     - ODER: Entfernen Sie die VPC-Konfiguration (wenn nicht nötig)

5. Um VPC zu entfernen:
   - Klicken Sie auf "Edit"
   - Wählen Sie "No VPC"
   - Klicken Sie auf "Save"

## Schritt 6: Timeout und Memory konfigurieren

1. In Lambda Console → "Configuration" Tab
2. Klicken Sie auf "General configuration"
3. Klicken Sie auf "Edit"
4. Timeout: `3 minutes` (180 Sekunden) - wichtig für Retries
5. Memory: `512 MB` (oder mehr wenn nötig)
6. Klicken Sie auf "Save"

## Schritt 7: EventBridge-Trigger für tägliche Ausführung

1. In Lambda Console → "Configuration" Tab
2. Klicken Sie auf "Triggers"
3. Klicken Sie auf "Add trigger"
4. Wählen Sie "EventBridge (CloudWatch Events)"
5. Regel: "Create a new rule"
6. Regel-Name: `daily-gold-price-collection`
7. Regel-Beschreibung: `Trigger Lambda daily at 9:00 AM UTC`
8. Regel-Typ: "Schedule expression"
9. Schedule expression: `cron(0 9 * * ? *)` (täglich um 9:00 UTC)
   - Oder: `rate(1 day)` (täglich um Mitternacht UTC)
10. Klicken Sie auf "Add"

## Schritt 8: Testen

### 8.1 Manueller Test in Lambda Console

1. In Lambda Console → "Test" Tab
2. Klicken Sie auf "Create new test event"
3. Event-Name: `test-event`
4. Event JSON:
```json
{}
```
5. Klicken Sie auf "Save"
6. Klicken Sie auf "Test"
7. Prüfen Sie die Logs in CloudWatch

### 8.2 CloudWatch Logs prüfen

1. In Lambda Console → "Monitor" Tab
2. Klicken Sie auf "View CloudWatch logs"
3. Prüfen Sie die Logs auf Fehler

### 8.3 S3 prüfen

1. Gehen Sie zur S3 Console
2. Öffnen Sie Ihren Bucket
3. Prüfen Sie ob Dateien im `raw/YYYY/MM/DD/` Verzeichnis erstellt wurden

## Schritt 9: Troubleshooting

### Problem: JSONDecodeError "Expecting value: line 1 column 1 (char 0)"

**Ursachen:**
1. **Kein Internet-Zugang**: Lambda ist in VPC ohne NAT Gateway
   - Lösung: VPC entfernen oder NAT Gateway hinzufügen

2. **Yahoo Finance blockiert Requests**: User-Agent fehlt
   - Lösung: Code wurde aktualisiert mit User-Agent Header

3. **Rate Limiting**: Zu viele Requests
   - Lösung: Retry-Logik wurde verbessert mit kürzeren Wartezeiten

### Problem: Timeout

**Ursachen:**
1. Zu lange Retry-Wartezeiten
   - Lösung: Wartezeiten wurden reduziert (15s statt 90s)

2. Zu kurzer Lambda Timeout
   - Lösung: Timeout auf 3 Minuten erhöhen

### Problem: ModuleNotFoundError

**Ursachen:**
1. Dependencies nicht in Lambda Package
   - Lösung: Lambda Layer verwenden oder Dependencies in ZIP einbinden

### Problem: Access Denied zu S3

**Ursachen:**
1. IAM-Role hat keine S3-Berechtigungen
   - Lösung: IAM-Policy wie in Schritt 4 hinzufügen

## Schritt 10: Monitoring einrichten

1. In Lambda Console → "Monitor" Tab
2. Prüfen Sie:
   - Invocations: Anzahl Aufrufe
   - Duration: Ausführungszeit
   - Errors: Fehleranzahl
   - Throttles: Rate-Limiting

3. CloudWatch Alarms erstellen:
   - Gehen Sie zu CloudWatch → Alarms
   - Erstellen Sie Alarm für Lambda Errors
   - Erstellen Sie Alarm für Lambda Timeouts

## Nächste Schritte

- ✅ Lambda-Funktion läuft täglich
- ✅ Daten werden in S3 gespeichert
- ✅ Logs sind in CloudWatch verfügbar
- 🔄 Optional: Preprocessing-Lambda für Datenverarbeitung
- 🔄 Optional: SageMaker für Modell-Training

## Kosten-Schätzung

| Service | Usage | Kosten |
|---------|-------|--------|
| Lambda | 30 Invocations/Monat | **FREE** (1M free) |
| S3 Storage | ~100 MB/Monat | **FREE** (5GB free) |
| CloudWatch Logs | ~10 MB/Monat | **FREE** (5GB free) |
| EventBridge | 30 Rules/Monat | **FREE** (1M free) |
| **Total** | | **$0/Monat** (Free Tier) |

## Support

Bei Problemen:
1. Prüfen Sie CloudWatch Logs
2. Prüfen Sie Lambda Metrics
3. Testen Sie lokal mit `test_lambda_local.py`
4. Prüfen Sie S3-Berechtigungen

