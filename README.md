# Transformer-based Web Application Firewall (WAF)

## 🚀 Overview
This project implements a production-grade, **Transformer-based Web Application Firewall (WAF)** designed to protect web applications from zero-day attacks without relying on static rules. It uses a deep learning model to analyze HTTP requests in real-time and block malicious intent.

The system is designed as a **Plug-and-Play** solution using Nginx as a reverse proxy, making it easy to drop in front of any existing application. The target vulnerable application for this demo is **OWASP Juice Shop**.

## ✨ Features
- **🤖 Deep Learning Core**: Uses a custom PyTorch Transformer Encoder trained on HTTP traffic.
- **🛡️ Real-Time Protection**: Intercepts requests via Nginx `auth_request` and provides millisecond-latency decisions.
- **🔄 Continuous Learning**: Includes a feedback loop mechanism (`online_learning.py`) to fix false positives on-the-fly without full retraining.
- **🕵️ Pattern Recognition**: Detects SQL Injection, XSS, Path Traversal, and more based on semantic structure, not just keywords.
- **📦 Dockerized**: simple `docker-compose up` deployment.

## 🏗️ Architecture
```mermaid
graph LR
    User[User / Attacker] -->|HTTP Request| Nginx[Nginx Reverse Proxy]
    Nginx -->|Check Request| WAF[WAF Service (FastAPI)]
    WAF -- Tokenize & Predict --> Model[Transformer Model]
    Model -->|Score| WAF
    WAF -->|Allow (200) / Block (403)| Nginx
    
    subgraph "Decision"
    Nginx -->|If Allowed| App[Protected App (Juice Shop)]
    Nginx -->|If Blocked| Block[403 Forbidden Page]
    end
```

## 🛠️ Installation & Setup

### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (for running local scripts)

### 1. Start the System
```bash
docker-compose up -d --build
```
This starts:
- **Juice Shop** (Target App) on port `3000` (internal)
- **WAF Service** (ML Inference) on port `8000` (internal)
- **Nginx** (Gateway) on port `8080` (public)

### 2. Access the Application
Visit **[http://localhost:8080](http://localhost:8080)** to use the protected application.

## 🧪 Verification & Usage
We provide scripts to verify the WAF's effectiveness.

### Automatic Verification
Run the verification suite to test benign and malicious payloads:
```bash
python3 scripts/verify_waf.py
```
*Expected Output: All tests passed.*

### Manual Testing
**Benign Request (Allowed):**
```bash
curl -I "http://localhost:8080/rest/products/search?q=apple"
# HTTP/1.1 200 OK
```

**Malicious Attack (Blocked):**
```bash
curl -I "http://localhost:8080/rest/products/search?q=' OR 1=1 --"
# HTTP/1.1 403 Forbidden
```

## 🧠 Continuous Learning
If the model creates a **False Positive** (blocks a valid request), you can "teach" it to accept that pattern instantly.

1. **Identify the blocker**: e.g., searching for "apple" is blocked.
2. **Run the Fix Script**:
   ```bash
   python3 scripts/fix_false_positive.py
   ```
   *This script fine-tunes the model on the benign sample while replaying malicious samples to maintain boundaries, then restarts the WAF service.*

## 📂 Project Structure
- `nginx/`: Nginx configuration and logs.
- `waf/`: Source code for the WAF service.
  - `app/`: FastAPI application.
  - `model/`: PyTorch model definition.
  - `train/`: Training pipelines (Offline & Online).
  - `data/`: Tokenizers and normalizers.
- `scripts/`: Utility scripts for traffic generation and verification.

## 🔧 Tech Stack
- **Languages**: Python, Lua (Nginx integration)
- **ML Framework**: PyTorch, HuggingFace Tokenizers
- **API**: FastAPI, Uvicorn
- **Infrastructure**: Docker, Nginx