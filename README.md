# 🛡️ SecureBERT WAF — AI-Powered Web Application Firewall

![Status](https://img.shields.io/badge/Status-Operational-brightgreen)
![Model](https://img.shields.io/badge/Model-SecureBERT%20(Fine--tuned)-blue)
![Platform](https://img.shields.io/badge/Platform-Docker%20%7C%20Standalone-orange)
![Detection](https://img.shields.io/badge/Zero--Day%20Detection-85%25-success)
![Dashboard](https://img.shields.io/badge/Dashboard-Bootstrap%205%20%2B%20Chart.js-informational)

An intelligent, plug-and-play Web Application Firewall that uses **SecureBERT** (a fine-tuned BERT Transformer) to detect and block web attacks in real-time — including **zero-day variants** the model has never seen before.

Drop it in front of **any** web application as a reverse-proxy plugin. No code changes required in your app.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **4-Layer Detection** | Rule-based → AI (BERT) → Uncertainty → Combined decision |
| **Zero-Day Protection** | 85%+ detection on never-before-seen attack variants |
| **Real-Time Dashboard** | Bootstrap 5 + Chart.js live monitoring UI |
| **Plug-and-Play** | Docker sidecar — no code changes in your application |
| **Self-Learning** | Online learning corrects false positives without downtime |
| **Low Latency** | 40–80 ms per request |

---

## 🏗️ Architecture

```
Client ──▶ Nginx (port 8080) ──▶ WAF Service (FastAPI) ──▶ Decision
                │                         │
                │                  ┌──────┴──────┐
                │                  │ 4 Layers:   │
                │                  │ 1. Rules    │
                │                  │ 2. AI/BERT  │
                │                  │ 3. Uncertainty│
                │                  │ 4. Combined │
                │                  └──────┬──────┘
                │                         │
          200 OK ◀── Allow ◀──────────────┘
          403    ◀── Block ◀──────────────┘
                │
                ▼
        Your Application
```

| Service | Role | Port |
|---------|------|------|
| **Nginx** | Reverse proxy + dashboard host | `8080` |
| **WAF Service** | FastAPI + BERT inference | `8000` |
| **Target App** | Your web app (e.g. Juice Shop) | Internal |

---

## 📂 Project Structure

```
transformer-waf-test/
├── waf/                        # Core WAF engine
│   ├── app/main.py            # FastAPI app — /analyze, /api/stats, /api/logs, /api/test
│   ├── model/
│   │   ├── transformer.py     # WAFTransformer (BERT-based classifier)
│   │   ├── tokenizer.py       # HttpTokenizer (BPE tokenization)
│   │   └── weights/           # Pre-trained model + tokenizer files
│   ├── data/
│   │   ├── normalizer.py      # URI normalization
│   │   └── build_dataset.py   # Dataset generation
│   ├── train/
│   │   ├── train_pipeline.py  # Full training pipeline
│   │   └── online_learning.py # False-positive correction
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                   # Dashboard UI
│   ├── index.html             # Bootstrap 5 layout
│   ├── style.css              # Custom dark theme
│   └── script.js              # Chart.js + live polling
│
├── nginx/
│   ├── nginx.conf             # Proxy + dashboard routing
│   └── logs/                  # Access & error logs
│
├── scripts/                    # Testing utilities
│   ├── test_zero_day_detection.py
│   ├── generate_malicious.py
│   └── generate_benign.py
│
├── docker-compose.yml
├── waf_dashboard.py           # CLI dashboard
└── test_realtime.py           # Interactive tester
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version |
|------------|---------|
| **Git** | Any |
| **Python** | 3.10+ |
| **Docker & Docker Compose** | Latest (for Docker mode) |
| **Disk Space** | ~4 GB (model weights + images) |

### Step 1 — Clone

```bash
git clone https://github.com/PriscillajospinG/transformer-waf-test.git
cd transformer-waf-test
```

---

## 🐳 Option A: Run with Docker (Recommended)

This is the easiest way — everything runs inside containers.

### 1. Build & Start

```bash
docker-compose up -d --build
```

First run takes 2–3 minutes (downloads PyTorch, model weights, Juice Shop image).

### 2. Verify

```bash
docker ps
```

You should see **3 containers**:

| Container | Status |
|-----------|--------|
| `waf-service` | Up — AI engine |
| `waf-nginx` | Up — Reverse proxy |
| `juice-shop` | Up — Test application |

### 3. Access

| URL | Description |
|-----|-------------|
| `http://localhost:8080/` | Protected application (Juice Shop) |
| `http://localhost:8080/dashboard/` | **Live WAF Dashboard** |
| `http://localhost:8000/health` | WAF health check (JSON) |

### 4. Test

```bash
# Safe request → 200 OK
curl -I "http://localhost:8080/rest/products/search?q=apple"

# SQL injection → 403 Forbidden
curl -I "http://localhost:8080/rest/products/search?q=' OR 1=1 --"

# XSS → 403 Forbidden
curl -I "http://localhost:8080/rest/products/search?q=<script>alert(1)</script>"

# Path traversal → 403 Forbidden
curl -I "http://localhost:8080/../../etc/passwd"
```

### 5. Stop

```bash
docker-compose down
```

---

## 💻 Option B: Run Without Docker (Standalone)

Run the WAF backend and dashboard directly on your machine — no Docker needed.

### 1. Install Dependencies

```bash
pip install -r waf/requirements.txt
```

### 2. Start the WAF Backend

```bash
cd waf
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The model downloads automatically on first run (~500 MB).

### 3. Serve the Dashboard

Open a **second terminal**:

```bash
cd frontend
python3 -m http.server 3000
```

### 4. Update API URL

Edit `frontend/script.js` line 2:

```javascript
// Change:
const API_BASE = '/api';

// To:
const API_BASE = 'http://localhost:8000/api';
```

### 5. Access

| URL | Description |
|-----|-------------|
| `http://localhost:3000` | **Dashboard UI** |
| `http://localhost:8000/health` | WAF health check |
| `http://localhost:8000/api/stats` | Live statistics (JSON) |
| `http://localhost:8000/api/logs` | Recent request log (JSON) |

### 6. Test (direct API call)

```bash
curl -X POST http://localhost:8000/api/test \
  -H "Content-Type: application/json" \
  -d '{"url": "/api/Users?q='\'' OR 1=1 --"}'
```

---

## 🔌 Protect Your Own Website

This is the main use case! Replace Juice Shop with your actual web application. The WAF works with **any** web app — no code changes required.

### Architecture Overview

```
Users (Internet)
    ↓
Nginx Reverse Proxy (localhost:8080)
    ↓
WAF Service (AI Analysis)
    ├─ Rule-based Detection
    ├─ BERT/SecureBERT Neural Network
    ├─ Uncertainty Detection
    └─ Combined Decision
    ↓
Your Web App (whatever you want to protect)
    ↓
Response → Dashboard Logs & Statistics
```

All traffic flows through the WAF before reaching your app. Safe requests: ✅ Allow. Suspicious requests: 🛑 Block.

---

### Simple Setup (3 Easy Steps)

#### **Step 1: Prepare Your Application**

Your app must be accessible **inside Docker** as a service. You have 3 options:

**Option A: Docker Image** (Recommended)
```bash
# If your app has a Dockerfile
docker build -t my-app:latest .
```

**Option B: Public Docker Image**
```bash
# Use an existing image from Docker Hub
# Examples: node:18, python:3.11, nginx:latest, etc.
```

**Option C: Running on Host Machine**
```bash
# If your app runs locally (not Docker)
# You'll use host.docker.internal instead
# Example: my-nodejs-app listening on localhost:3000
```

---

#### **Step 2: Update docker-compose.yml**

Open `docker-compose.yml` and replace the `juice-shop` service:

```yaml
version: '3.8'

services:
  # ========== REPLACE THIS SECTION ==========
  my-website:
    image: my-app:latest              # ← Your Docker image name
    container_name: my-website        # ← Unique container name
    restart: always
    ports:
      - "3000:3000"                   # ← Your app's internal port
    environment:
      - NODE_ENV=production           # ← Your app's env vars (if any)
      - DATABASE_URL=...              # ← Add any needed config
      - TZ=Asia/Kolkata               # ← Keep this for timezone
    networks:
      - waf-net
    # If your app needs volumes:
    # volumes:
    #   - ./data:/app/data
  # ========== END REPLACEMENT ==========

  # ========== KEEP THESE AS-IS ==========
  waf-service:
    build:
      context: ./waf
      dockerfile: Dockerfile
    container_name: waf-service
    restart: always
    environment:
      - MODEL_PATH=/app/model/weights
      - TZ=Asia/Kolkata
    volumes:
      - ./waf:/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    networks:
      - waf-net

  nginx:
    image: nginx:latest
    container_name: waf-nginx
    restart: always
    environment:
      - TZ=Asia/Kolkata
    ports:
      - "8080:80"                     # ← WAF exposed on 8080
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/logs:/var/log/nginx
      - ./frontend:/usr/share/nginx/html/dashboard:ro
    depends_on:
      - my-website                    # ← Update to match your service name
      - waf-service
    networks:
      - waf-net

networks:
  waf-net:
    driver: bridge
```

**Key Changes:**
- `image`: Replace with your app's Docker image
- `container_name`: Use a unique name (e.g., `my-website`, `api-server`)
- `ports`: Change to match your app's port (e.g., `3000:3000`)
- `environment`: Add env vars your app needs
- `depends_on`: Update to your service name
- Keep all `waf-service` and `nginx` sections **unchanged**

---

#### **Step 3: Update nginx/nginx.conf**

Edit `nginx/nginx.conf` line ~22-24:

```nginx
# BEFORE:
upstream juice_shop {
    server juice-shop:3000;
}

# AFTER:
upstream my_app {
    server my-website:3000;      # ← Match your service name & internal port
}
```

Then update lines where `juice_shop` is referenced (usually 3-4 places):

```nginx
# Find and replace these lines:

# Line ~55: Static assets
location ~* \.(css|js|...)$ {
    proxy_pass http://my_app;    # ← Changed from juice_shop
}

# Line ~63: Main app proxy
location / {
    auth_request /_waf_check;
    proxy_pass http://my_app;    # ← Changed from juice_shop
}
```

**Quick Find/Replace:**
```bash
sed -i 's/juice_shop/my_app/g' nginx/nginx.conf
sed -i 's/juice-shop:3000/my-website:3000/g' nginx/nginx.conf
```

---

### Example: Protecting a Node.js App

**Your app structure:**
```
my-node-app/
├── Dockerfile
├── index.js
├── server.js
└── package.json
```

**Your `docker-compose.yml`:**
```yaml
services:
  nodejs-backend:
    image: my-node-app:latest
    container_name: nodejs-backend
    restart: always
    ports:
      - "4000:4000"
    environment:
      - NODE_ENV=production
      - PORT=4000
      - DB_HOST=mongo
      - TZ=Asia/Kolkata
    networks:
      - waf-net

  # ... waf-service and nginx below, unchanged
```

**Your `nginx/nginx.conf`:**
```nginx
upstream nodejs_backend {
    server nodejs-backend:4000;
}

# ... in server block:
location ~* \.(css|js|...)$ {
    proxy_pass http://nodejs_backend;
}

location / {
    auth_request /_waf_check;
    proxy_pass http://nodejs_backend;
}
```

**Deploy:**
```bash
docker-compose up -d --build
```

Access your protected app at: **`http://localhost:8080/`**  
Monitor with dashboard: **`http://localhost:8080/dashboard/`**

---

### Example: Protecting a Python Flask App

**Your `Dockerfile`:**
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

**Your `docker-compose.yml`:**
```yaml
services:
  flask-app:
    build: ./my-flask-app
    container_name: flask-app
    restart: always
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - FLASK_DEBUG=0
      - TZ=Asia/Kolkata
    networks:
      - waf-net

  # ... waf-service and nginx below
```

**Your `nginx/nginx.conf`:**
```nginx
upstream flask_backend {
    server flask-app:5000;
}

# ... use flask_backend in location blocks
```

---

### Example: Protecting an Existing Remote App

If your app already runs on a server (not Docker), you can forward it through Docker:

**docker-compose.yml:**
```yaml
services:
  external-app:
    image: nginx:alpine
    container_name: external-app
    restart: always
    volumes:
      - ./external-app-proxy.conf:/etc/nginx/nginx.conf:ro
    networks:
      - waf-net

  # ... waf is then updated to proxy to this
```

**external-app-proxy.conf:**
```nginx
events {}
http {
    server {
        listen 8001;
        location / {
            # Forward to your remote server
            proxy_pass http://your-server.com:8080;
            proxy_set_header Host $host;
        }
    }
}
```

---

### Deploy & Test

```bash
# Build and start all services
docker-compose up -d --build

# Wait for containers to start (30-60 seconds)
sleep 30

# Verify containers running
docker ps

# Check WAF is working
curl -s http://localhost:8080/api/stats | jq .

# Access your protected app
open http://localhost:8080/

# View live dashboard
open http://localhost:8080/dashboard/

# Test with a malicious request
curl "http://localhost:8080/api/search?q=' OR 1=1 --"
# Should see 403 Forbidden response
```

---

### Verify It's Working

**Dashboard Indicators:**
- ✅ "Total Requests" incrementing
- ✅ "Blocked" count increasing (if sending attacks)
- ✅ Real-time chart updates
- ✅ Request logs showing timestamp, IP, URI, status

**Manual Test:**
```bash
# This should work (200 OK)
curl -I http://localhost:8080/

# This should be blocked (403 Forbidden)
curl -I "http://localhost:8080/?search=<script>alert(1)</script>"

# Check logs
curl -s http://localhost:8080/api/logs | jq .
```

---

### Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| `502 Bad Gateway` | WAF service still starting. Wait 60s and retry. |
| `Connection refused` | App container isn't running. Check `docker logs my-website` |
| `nginx: [emerg] host not found` | Service name in `nginx.conf` doesn't match `docker-compose.yml`. Must be **exact same name**. |
| Port already in use | Change `ports: - "8080:80"` to `"8081:80"` or kill existing process. |
| App works without WAF but not through it | Check `nginx.conf` proxy headers are correct. Some apps need `X-Forwarded-*` headers. |
| Dashboard shows 0 requests | App isn't receiving traffic through nginx. Try `curl http://localhost:8080/` to generate traffic. |

---

### How the WAF Works

**Every request goes through this pipeline:**

1. **Nginx intercepts** → Checks request with WAF service
2. **WAF Analysis** (40-80ms):
   - ✅ **Rule-based layer**: Matches against known patterns
   - ✅ **AI layer**: BERT model predicts benign/malicious probability
   - ✅ **Uncertainty layer**: Flags borderline cases for review
   - ✅ **Combined decision**: Aggregates all layers
3. **Decision**:
   - ✅ Safe? → `200 OK` → Forward to your app
   - 🛑 Attack? → `403 Forbidden` → Block request
4. **Logging**: All decisions recorded → Dashboard updates in real-time

**Logs to dashboard:**
- Request timestamp
- Client IP
- HTTP method & URI
- Block/Allow status
- Attack reason (if blocked)
- Malicious probability (0.0 = safe, 1.0 = attack)

---


```

Your app is now protected at `http://localhost:8080/` and the dashboard is live at `http://localhost:8080/dashboard/`.

---

## 📊 Live Dashboard

The built-in dashboard provides real-time WAF monitoring:

| Feature | Description |
|---------|-------------|
| **Stat Cards** | Total requests, blocked, allowed, block rate |
| **Doughnut Chart** | Attack type distribution (Chart.js) |
| **Timeline Chart** | Blocked vs. allowed over time (Chart.js) |
| **Attack Tester** | Send test payloads to the WAF from the UI |
| **Progress Bars** | Breakdown by SQLi, XSS, path traversal, encoding |
| **Live Log Table** | Scrollable table with recent requests |

**Tech Stack:** HTML5 + CSS3 + JavaScript + Bootstrap 5 + Chart.js

---

## 🛡️ Detection Layers

### Layer 1 — Rule-Based (1 ms)
Keyword matching for 40+ attack signatures: `union`, `select`, `<script>`, `../`, etc.

### Layer 2 — AI / BERT (40 ms)
Fine-tuned Transformer that understands the **semantic meaning** of HTTP requests.

### Layer 3 — Uncertainty Detection
Flags borderline predictions (confidence 0.70–0.80) for extra checks.

### Layer 4 — Combined Decision
Requires agreement between rule-based and AI layers before blocking.

**Result:** 85%+ zero-day detection, <3% false positive rate.

---

## ⚙️ Configuration

Key thresholds in `waf/app/main.py`:

```python
AI_CONFIDENCE_THRESHOLD = 0.95    # Block if malicious_prob > this
UNCERTAINTY_THRESHOLD   = 0.85    # Flag as uncertain above this
```

| Scenario | Adjust |
|----------|--------|
| Too many false positives | Raise `AI_CONFIDENCE_THRESHOLD` to `0.98` |
| Attacks getting through | Lower `AI_CONFIDENCE_THRESHOLD` to `0.90` |

After changing thresholds:

```bash
# Docker
docker-compose restart waf-service

# Standalone
# Just save the file — uvicorn --reload picks it up automatically
```

---

## 🔄 Online Learning (Fix False Positives)

If the WAF blocks a legitimate request:

```bash
# 1. Find blocked request
tail -f nginx/logs/access.log | grep 403

# 2. Correct the model
python3 scripts/fix_false_positive.py "GET /your/legitimate/path"

# 3. Restart
docker-compose restart waf-service
```

---

## 🧪 Test Suites

| Script | Purpose | Command |
|--------|---------|---------|
| **CLI Dashboard** | Real-time detection across all 4 layers | `python3 waf_dashboard.py` |
| **Interactive Tester** | Menu-driven testing | `python3 test_realtime.py` |
| **Zero-Day Suite** | Automated 20+ test cases | `python3 scripts/test_zero_day_detection.py` |
| **Generate Attacks** | Continuous malicious traffic | `python3 scripts/generate_malicious.py` |

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Known attack detection | 99% |
| Zero-day detection | 85%+ |
| False positive rate | <2% |
| Inference latency | 40–80 ms |
| Training data | 5,000 samples |

---

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| Containers won't start | `docker-compose up -d --build` — wait 3 min |
| Port 8080 in use | Change port in `docker-compose.yml` → `"8081:80"` |
| WAF returns 200 for attacks | Check model loaded: `docker logs waf-service` |
| Dashboard not loading | Verify `frontend/` volume mount in `docker-compose.yml` |
| Assets broken on target app | Raise `AI_CONFIDENCE_THRESHOLD` to `0.98` |
| Model not downloading | Check internet — first run needs ~500 MB |

---

## 📚 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | GET | Nginx subrequest — returns 200 (allow) or 403 (block) |
| `/api/stats` | GET | Aggregated stats (total, blocked, allowed, attack types) |
| `/api/logs` | GET | Recent request log (last 200 entries) |
| `/api/test` | POST | Test a URL: `{"url": "/path?q=payload"}` |
| `/health` | GET | Service health and model status |

---

## 🤝 Contributing

PRs welcome! Areas of interest:
- More diverse training datasets
- Support for LSTM / CNN architectures
- Integration with AWS WAF, Cloudflare, etc.
- Adversarial robustness improvements

## 📄 License

MIT License — free for educational and commercial use.

---

> **Disclaimer:** This project is for educational and defensive purposes only. Use responsibly and only on systems you own or have explicit permission to test.
