# Real-Time Logging Architecture

This document explains how real-time logs are collected and processed throughout the WAF system.

## 📊 Complete Logging Flow

```
REQUEST → ANALYSIS → LOGGING → MONITORING → STORAGE
   ↓        ↓          ↓           ↓           ↓
Browser  WAF API   FastAPI      Dashboard   Nginx Access
         main.py   logger        + Files      Log File
```

---

## 🔵 Part 1: Log Generation (Collection)

### 1.1 **WAF Service Logs** - `waf/app/main.py`

**Location**: FastAPI service (port 8000)  
**When**: Every HTTP request passed through the WAF  
**Format**: Python logging with structured messages

#### The Analysis Endpoint - `/analyze`
```python
# waf/app/main.py lines 150-236
@app.api_route("/analyze", methods=[...])
async def analyze_request(request: Request):
    # Layer 1: Rule-Based Detection
    if is_keyword_suspicious and has_injection:
        logger.warning(f"BLOCKING via Rule-Based Detection: IP={original_ip} URI={original_uri}")
        return Response(status_code=403)
    
    # Layer 2: AI-Based Detection
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs, dim=1)
        malicious_prob = probs[0][1].item()
    
    # Layer 3: Uncertainty Detection
    if UNCERTAINTY_THRESHOLD <= malicious_prob < AI_CONFIDENCE_THRESHOLD:
        logger.warning(f"UNCERTAIN PREDICTION: IP={original_ip} URI={original_uri} MaliciousProb={malicious_prob:.4f}")
    
    # Layer 4: Final Decision
    if should_block:
        logger.warning(f"BLOCKING: {log_msg} Reason={block_reason}")
        return Response(status_code=403)
    else:
        logger.info(f"ALLOWED: {log_msg}")
        return Response(status_code=200)
```

**Log Messages Generated**:
1. **Rule-Based Block**: `"BLOCKING via Rule-Based Detection: IP=... URI=... Keywords=[...]"`
2. **Encoding Attack**: `"BLOCKING Encoding Attack: IP=... URI=..."`
3. **Uncertain Prediction**: `"UNCERTAIN PREDICTION (Potential Zero-Day): IP=... URI=... MaliciousProb=0.87..."`
4. **AI Detection Block**: `"BLOCKING: IP=... BenignProb=0.05 MaliciousProb=0.95 Reason=..."`
5. **Allowed Request**: `"ALLOWED: IP=... BenignProb=0.98 MaliciousProb=0.02"`

**Logging Configuration**:
```python
# waf/app/main.py lines 14-16
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("waf-service")
```

### 1.2 **Nginx Access & Error Logs**

**Location**: `nginx/logs/access.log` and `nginx/logs/error.log`  
**When**: Every request hits Nginx (all HTTP traffic)  
**Format**: Nginx combined log format with custom request body field

#### Nginx Log Configuration - `nginx/nginx.conf`
```
log_format custom_log_format '$remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent "$http_referer" "$http_user_agent" "$request_body"';
access_log /var/log/nginx/access.log custom_log_format;
error_log /var/log/nginx/error.log;
```

**Example Log Entry**:
```
172.18.0.1 - - [17/Feb/2026:10:00:00 +0000] "GET /rest/products/search?q=apple HTTP/1.1" 200 1234 "-" "curl/7.64.1" "-"
172.18.0.1 - - [17/Feb/2026:10:00:01 +0000] "GET /search?q=' OR 1=1 HTTP/1.1" 403 567 "-" "curl/7.64.1" "-"
```

---

## 🔄 Part 2: Log Processing & Parsing

### 2.1 **Log Parser** - `waf/utils/log_parser.py`

**Purpose**: Convert raw Nginx logs into structured Python objects

**Location**: `waf/utils/log_parser.py` lines 1-70

```python
class NginxLogParser:
    def __init__(self):
        self.log_pattern = re.compile(
            r'(?P<ip>[\d\.]+)\s-\s(?P<user>\S+)\s\[(?P<time>[^\]]+)\]\s"(?P<request>[^"]+)"\s(?P<status>\d+)\s(?P<bytes>\d+)\s"(?P<referer>[^"]*)"\s"(?P<ua>[^"]*)"\s"(?P<body>.*)"'
        )
    
    def parse_line(self, line: str) -> dict:
        """Parses a single line of Nginx log into a structured dictionary."""
        match = self.log_pattern.match(line)
        # Returns: {
        #   "ip": "172.18.0.1",
        #   "timestamp": "17/Feb/2026:10:00:00 +0000",
        #   "method": "GET",
        #   "uri": "/rest/products/search?q=apple",
        #   "status": 200,
        #   "user_agent": "curl/7.64.1",
        #   "body": "",
        #   "referer": "-"
        # }
    
    def parse_file(self, file_path: str):
        """Generator that yields parsed log entries from a file."""
        with open(file_path, 'r') as f:
            for line in f:
                parsed = self.parse_line(line.strip())
                if parsed:
                    yield parsed
```

**Input**: Raw Nginx log file  
**Output**: Python dict with parsed fields

---

## 📡 Part 3: Real-Time Monitoring & Collection

### 3.1 **Dashboard** - `waf_dashboard.py`

**Purpose**: Display real-time detection analysis with all 4 layers  
**How it works**: Fetches Docker logs in real-time

```python
# waf_dashboard.py lines 17-27
def get_waf_logs(last_n=20):
    """Fetch recent WAF logs"""
    try:
        result = subprocess.run(
            ["docker-compose", "logs", "waf-service", f"--tail={last_n}"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT  # Changed from hardcoded path
        )
        return result.stdout  # Returns last N log lines from WAF container
    except:
        return ""
```

#### How Dashboard Processes Logs - `waf_dashboard.py` lines 50-92

```python
def analyze_request(desc, method, path, expected_status):
    # 1. Make the HTTP request
    response = requests.request(method, f"{BASE_URL}{path}")
    
    # 2. Fetch WAF logs (last 30 lines after the request)
    logs = get_waf_logs(30)
    
    # 3. Parse logs to extract detection information
    detections = {
        'rule_based': [],
        'ai_detection': [],
        'uncertainty': [],
        'decision': None
    }
    
    for line in logs.split('\n'):
        if 'Rule-Based' in line or 'Keywords' in line:
            detections['rule_based'].append(line.strip()[-70:])
        elif 'AI Detection' in line and 'prob=' in line:
            detections['ai_detection'].append(line.strip()[-70:])
        elif 'UNCERTAINTY' in line:
            detections['uncertainty'].append(line.strip()[-70:])
        elif 'BLOCKING' in line:
            detections['decision'] = 'BLOCKED (403)'
        elif 'ALLOWING' in line:
            detections['decision'] = 'ALLOWED (200)'
    
    # 4. Display detection breakdown
    print("Layer 1 - Rule-Based Detection:", detections['rule_based'])
    print("Layer 2 - AI Detection (BERT):", detections['ai_detection'])
    print("Layer 3 - Uncertainty Detection:", detections['uncertainty'])
    print("Layer 4 - Final Decision:", detections['decision'])
```

**Output Example**:
```
🔍 REQUEST ANALYSIS: SQL Injection Attack
════════════════════════════════════════════════════════════════════════════════
   Timestamp: 10:00:45
   Method: GET
   Path: /search?q=' OR 1=1
   Expected: 403
   Actual: 403 ✓ PASS
   Latency: 42.5ms

   📋 DETECTION LAYERS:
   Layer 1 - Rule-Based Detection:
      ✓ BLOCKING via Rule-Based Detection: URI=/search Keywords=['or 1=1', 'select']
   Layer 2 - AI Detection (BERT):
      ✓ AI Detection: MaliciousProb=0.9650
   Layer 3 - Uncertainty Detection:
      (no uncertainty - high confidence)
   Layer 4 - Final Decision:
      → BLOCKED (403)
```

### 3.2 **Real-Time Test Suite** - `test_realtime.py`

Similar approach to dashboard, but focused on testing and simpler output.

```python
# test_realtime.py lines 28-35
def get_waf_logs():
    result = subprocess.run(
        ["docker-compose", "logs", "waf-service", "--tail=5"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT  # Dynamic path
    )
    return result.stdout
```

---

## 💾 Part 4: Log Storage & Analysis

### 4.1 **Persistent Storage**

#### Nginx Access Logs
- **File**: `nginx/logs/access.log`
- **Purpose**: Historical record of all requests and responses
- **Format**: Nginx combined format
- **Retention**: Persistent (survives container restart)

#### Docker Container Logs
- **Source**: WAF service standard output
- **Storage**: Docker daemon manages (usually in `/var/lib/docker/containers/`)
- **Retrieval**: `docker-compose logs waf-service`
- **Retention**: By default kept unless containers are removed

### 4.2 **Online Learning** - `waf/train/online_learning.py`

**Purpose**: Retrain model on recent logs (for false positive correction)

```python
# waf/train/online_learning.py lines 38-75
def online_learning(log_file="../../nginx/logs/access.log", epochs=1):
    # 1. Load existing model
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # 2. Parse Nginx logs to extract new training data
    parser = NginxLogParser()
    new_data = []
    
    for entry in parser.parse_file(log_file):
        if entry['status'] < 400:  # Get successful (benign) requests
            norm = normalizer.normalize(entry)
            new_data.append(norm)
    
    # 3. Fine-tune model on new data
    dataset = WAFDataset(new_data, labels=[0]*len(new_data), tokenizer)
    dataloader = DataLoader(dataset, batch_size=8)
    
    # Train loop: backward pass, optimizer update
    optimizer.step()
    
    # 4. Save updated model
    torch.save(model.state_dict(), MODEL_PATH)
```

**Workflow**:
1. Read Nginx access logs
2. Extract successful requests (status < 400)
3. Normalize them using `RequestNormalizer`
4. Fine-tune existing model on new data
5. Save updated model weights back to disk

### 4.3 **Dataset Building** - `waf/data/build_dataset.py`

**Purpose**: Build training datasets from log files

```python
# waf/data/build_dataset.py lines 14-40
def build_dataset():
    parser = NginxLogParser()
    
    # Read Nginx logs
    for entry in parser.parse_file(LOG_FILE):
        if "/_waf_check" not in entry['uri']:  # Filter WAF checks
            normalized = normalizer.normalize(entry)
            dataset_file.write(normalized + "\n")
    
    # Train tokenizer on dataset
    tokenizer = HttpTokenizer(vocab_size=1000)
    tokenizer.train([DATASET_FILE])
    tokenizer.save(TOKENIZER_FILE)
```

---

## 🔍 Part 5: Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      REQUEST LIFECYCLE                             │
└─────────────────────────────────────────────────────────────────────┘

HTTP Request
    ↓
[Nginx Reverse Proxy]
    ├→ Extracts: Method, URI, IP, Headers, Status
    ├→ Logs to: nginx/logs/access.log
    └→ Auth Subrequest to WAF
           ↓
    [WAF Service /analyze endpoint]
           ├→ Layer 1: Rule-Based Detection
           ├→ Layer 2: BERT Model Inference
           ├→ Layer 3: Uncertainty Detection
           ├→ Layer 4: Combined Decision
           └→ logger.warning() / logger.info()
                      ↓
    [Docker Container Stdout]
           └→ Captured by Docker daemon
                      ↓
          (Used by Dashboard/Testing)
    ┌──────────────────┬──────────────────┬──────────────────┐
    ↓                  ↓                  ↓                  ↓
Real-Time        Persistent        Online Learning    Dataset Building
Monitoring       Storage            (Retraining)      (Training)
(Dashboard)   (Nginx Logs)      (online_learning.py) (build_dataset.py)
    │              │                    │                   │
    ├→ Parse Logs  ├→ Store in file    ├→ Parse logs      ├→ Parse logs
    ├→ Extract     ├→ Historical       ├→ Filter benign   ├→ Normalize
    │  Detection   │  Record           ├→ Fine-tune       ├→ Tokenize
    │  Info        └─ Searchable       ├→ Save model      └→ Train
    └→ Display                         └─ Reload WAF       Tokenizer


┌────────────────────────────────────────────────────────────────────┐
│                         LOG SOURCES                                │
└────────────────────────────────────────────────────────────────────┘

1. FASTAPI LOGS (In-Memory)
   Source: waf/app/main.py → logger
   Method: Python logging module
   Format: TEXT with structured info
   Access: docker-compose logs waf-service
   Duration: Container lifetime

2. NGINX LOGS (File-Based)
   Source: nginx/logs/access.log
   Method: Nginx access_log directive
   Format: Custom combined + request_body
   Access: Direct file read
   Duration: Persistent until deleted

3. DOCKER LOGS (Container Logs)
   Source: All STDOUT/STDERR from services
   Method: Docker daemon captures outputs
   Format: TEXT
   Access: docker logs / docker-compose logs
   Duration: Until container removed
```

---

## 🔧 How to Use the Logging System

### View Real-Time Logs

**Dashboard (Recommended - Shows all 4 layers)**:
```bash
python3 waf_dashboard.py
```

**Docker Logs (Raw)**:
```bash
docker-compose logs -f waf-service
```

**Nginx Access Logs (All Requests)**:
```bash
tail -f nginx/logs/access.log
```

**Filter for Blocked Requests**:
```bash
tail -f nginx/logs/access.log | grep " 403 "
```

### Extract Data for Analysis

**Parse All Logs Programmatically**:
```python
from waf.utils.log_parser import NginxLogParser

parser = NginxLogParser()
for entry in parser.parse_file("nginx/logs/access.log"):
    print(f"{entry['ip']} {entry['method']} {entry['uri']} → {entry['status']}")
```

### Retrain Model on Recent Logs

```bash
# Fine-tune model on new traffic
python3 waf/train/online_learning.py

# Reload model in WAF
docker-compose restart waf-service
```

---

## 📊 Summary Table

| Component | Location | Triggers | Format | Storage | Duration |
|-----------|----------|----------|--------|---------|----------|
| **FastAPI Logger** | waf/app/main.py | Every /analyze call | TEXT (structured) | Docker stdout | Container lifetime |
| **Nginx Access** | nginx/logs/access.log | Every HTTP request | Nginx combined | File | Persistent |
| **Nginx Error** | nginx/logs/error.log | Errors/warnings | Nginx format | File | Persistent |
| **Docker Logs** | Docker daemon | Container stdout | TEXT | Docker storage | Until image removed |

---

## 🎯 Key Takeaways

1. **Real-time logs come from FastAPI logger** in `waf/app/main.py`
2. **Dashboard fetches logs via `docker-compose logs`** command
3. **Nginx logs provide persistent historical records** for analysis
4. **Log parser (`NginxLogParser`)** converts raw logs to structured data
5. **Online learning uses Nginx logs** to retrain on recent traffic
6. **All paths are now dynamic** - works for any user/installation

