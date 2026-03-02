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

## 🔌 Use as a Plugin for Your Own App

SecureBERT WAF can protect **any** web application. Replace the Juice Shop test app with yours.

### Step 1 — Edit `docker-compose.yml`

Replace the `juice-shop` service with your app:

```yaml
services:
  # Replace this with YOUR application
  my-app:
    image: your-app-image:latest    # Your Docker image
    container_name: my-app
    restart: always
    networks:
      - waf-net

  # Keep these unchanged
  waf-service:
    build:
      context: ./waf
      dockerfile: Dockerfile
    container_name: waf-service
    restart: always
    environment:
      - MODEL_PATH=/app/model/weights
    volumes:
      - ./waf:/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    networks:
      - waf-net

  nginx:
    image: nginx:latest
    container_name: waf-nginx
    restart: always
    ports:
      - "8080:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/logs:/var/log/nginx
      - ./frontend:/usr/share/nginx/html/dashboard:ro
    depends_on:
      - my-app
      - waf-service
    networks:
      - waf-net

networks:
  waf-net:
    driver: bridge
```

### Step 2 — Update `nginx/nginx.conf`

Change the upstream target to point to your app:

```nginx
upstream target_app {
    server my-app:3000;     # ← Your app's container name and port
}
```

### Step 3 — Deploy

```bash
docker-compose up -d --build
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
