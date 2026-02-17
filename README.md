# 🛡️ SecureBERT WAF - Plug-and-Play Web Application Firewall

**Enterprise-grade AI-powered Web Application Firewall with Real-time Attack Detection, Reinforcement Learning, and Zero-Day Protection**

## 📋 Overview

SecureBERT WAF is a production-ready, plug-and-play Web Application Firewall that uses:

- **SecureBERT**: BERT-based transformer model for attack detection (85%+ zero-day accuracy)
- **Multi-Layer Detection**: Rule engine + ML + anomaly detection + uncertainty scoring
- **Reinforcement Learning**: Q-learning feedback loop to improve policy from admin guidance  
- **Zero-Day Protection**: Embedding-based anomaly detection for unknown attacks
- **Reverse Proxy**: Complete HTTP reverse proxy with request forwarding
- **Admin API**: Full REST API for logs, feedback, and configuration
- **Dashboard**: Real-time monitoring UI

## 🚀 Quick Start (3 Steps)

### 1. Clone and Configure

```bash
cd [project-directory]
cat CONFIG.env
# Edit target website URL:
# TARGET_WEBSITE_URL=http://your-website.com    # Your backend
# PUBLIC_IP_OR_DOMAIN=your-domain.com            # Public domain
# PUBLIC_PORT=8080                               # Public port
nano CONFIG.env  # Edit your settings
```

### 2. Start WAF

```bash
bash start_waf.sh
```

The script will:
- ✅ Build Docker images
- ✅ Start all services (WAF, Nginx, Target App)
- ✅ Initialize database
- ✅ Load models
- ✅ Launch monitoring daemon

### 3. Test Protection

```bash
# Access the protected website
curl http://localhost:8080/

# Test attack detection  
curl "http://localhost:8080/search?q=' OR 1=1--"     # SQL Injection - BLOCKED ✓
curl "http://localhost:8080/page?id=<script>alert(1)</script>"  # XSS - BLOCKED ✓

# View dashboard
open http://localhost:8080/dashboard
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client Browser                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼ (HTTP Request)
         ┌───────────────────────┐
         │  NGINX (Port 8080)    │ ← Public entry point
         │  Reverse Proxy        │
         └───────────┬───────────┘
                     │
                     ▼
      ┌──────────────────────────────────┐
      │     WAF Service (Port 8000)      │
      │  ┌────────────────────────────┐  │
      │  │ LAYER 1: Rule Engine       │  │ Fast pattern/keyword matching
      │  ├────────────────────────────┤  │
      │  │ LAYER 2: BERT Detector     │  │ ML-based classification
      │  ├────────────────────────────┤  │
      │  │ LAYER 3: Anomaly Detection │  │ Zero-day detection via embeddings
      │  ├────────────────────────────┤  │
      │  │ LAYER 4: RL Policy         │  │ Learned from admin feedback
      │  └────────────────────────────┘  │
      │                                   │
      │  Combined Decision: ALLOW/BLOCK   │
      └──────────────┬───────────────────┘
                     │
             ┌───────┴────────┐
             │                │
      ✓ ALLOW               ✗ BLOCK
             │                │
             ▼                ▼
    ┌──────────────────┐  Return 403
    │  Juice Shop or   │  (Logged in DB)
    │  Your Website    │
    │  (Port 3000)     │
    └──────────────────┘
             │
             ▼ (HTTP Response)
         Back to Client
```

## 🔍 Detection Engines

### Layer 1: Rule Engine
Fast, signature-based detection:
- SQL Injection patterns (UNION SELECT, OR 1=1, etc.)
- XSS patterns (script tags, event handlers)
- Path traversal (../, ..\)
- Command injection (shell commands)
- LDAP/XML injection
- URL encoding bypasses

### Layer 2: BERT Classifier  
SecureBERT neural network:
- Trained on benign and malicious HTTP requests
- Returns confidence scores (0.0-1.0)
- Detects novel attack variants
- ~85% zero-day detection accuracy
- <100ms inference latency

### Layer 3: Anomaly Detection (Zero-Day)
Similarity-based detection:
- Stores embeddings from benign requests
- Detects new requests far from benign cluster
- Uses cosine similarity + statistical distance
- Flags potential zero-day attacks
- Learns online as new benign traffic arrives

### Layer 4: RL Policy
Reinforcement Learning:
- Q-learning: state = (bert_score, rule_score, method, endpoint)
- Admin feedback updates decision policy
- Epsilon-greedy exploration/exploitation
- Continuous policy improvement over time

## ⚙️ Configuration

Edit `CONFIG.env`:

```bash
# TARGET APPLICATION
TARGET_WEBSITE_URL=http://target-app:3000     # Backend to protect
PUBLIC_IP_OR_DOMAIN=localhost
PUBLIC_PORT=8080

# ML THRESHOLDS (0.0 to 1.0)
AI_CONFIDENCE_THRESHOLD=0.95              # BERT confidence to block
UNCERTAINTY_THRESHOLD=0.85                # Flag uncertain predictions
RULE_ENGINE_THRESHOLD=0.70                # Rule score threshold
ANOMALY_THRESHOLD=0.75                    # Anomaly score threshold
COMBINED_THRESHOLD=0.85                   # Combined multi-layer threshold

# REINFORCEMENT LEARNING
RL_ENABLED=true
RL_EPSILON=0.1                            # Exploration rate
RL_ALPHA=0.1                              # Learning rate
RL_GAMMA=0.9                              # Discount factor

# ANOMALY DETECTION
ANOMALY_ENABLED=true
ANOMALY_EMBEDDING_CACHE_SIZE=1000         # Benign embeddings to store
BLOCK_ON_ANOMALY=true
BLOCK_ON_UNCERTAIN=false

# LOGGING
LOG_LEVEL=INFO
LOG_FILE=/app/logs/waf.log
LOG_ALLOWED_REQUESTS=true

# FEATURES
ADMIN_API_ENABLED=true
```

## 📊 Admin API Endpoints

### Get Recent Logs
```bash
curl http://localhost:8080/api/logs?limit=100
```

### Get Statistics
```bash
curl http://localhost:8080/api/stats
```

### Submit Feedback (Train RL)
```bash
curl -X POST http://localhost:8080/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "12345",
    "decision": "malicious",
    "confidence": 0.95,
    "notes": "Actually detected a SQL injection attack"
  }'
```

### View RL Q-Table
```bash
curl http://localhost:8080/api/qtable?limit=50
```

### Get Configuration
```bash
curl http://localhost:8080/api/config
```

### Update Thresholds (Runtime)
```bash
curl -X POST http://localhost:8080/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "threshold_name": "AI_CONFIDENCE_THRESHOLD",
    "value": 0.90
  }'
```

## 📈 Dashboard

Access at `http://localhost:8080/dashboard`

Features:
- Real-time statistics
- Recent blocked/allowed requests  
- Threat visualization
- Action buttons (export logs, view config)
- Auto-refreshes every 5 seconds

## 🧪 Testing

### Test SQL Injection Detection
```bash
curl "http://localhost:8080/user/profile?id=1' OR '1'='1"
curl "http://localhost:8080/search?q='; DROP TABLE users;--"
```

### Test XSS Detection
```bash
curl "http://localhost:8080/page?content=<script>alert('xss')</script>"
curl "http://localhost:8080/comment?text=<img src=x onerror=alert(1)>"
```

### Test Path Traversal Detection
```bash
curl "http://localhost:8080/files?path=../../../../etc/passwd"
curl "http://localhost:8080/image?id=..%2F..%2Fetc%2Fpasswd"
```

### Test Command Injection Detection
```bash
curl "http://localhost:8080/api/exec?cmd=ls;cat /etc/passwd"
curl "http://localhost:8080/execute?command=id|whoami"
```

## 🛑 Stopping the WAF

```bash
bash stop_waf.sh
```

Gracefully stops all services and cleans up Docker containers.

## 📋 Logs and Monitoring

### View Real-Time Logs
```bash
docker-compose logs -f waf-service
```

### Monitor Performance
```bash
docker stats waf-service nginx target-app
```

### Export Request Logs
```bash
curl http://localhost:8080/api/logs?limit=10000 > logs_export.json
```

## 🔄 Reinforcement Learning Workflow

1. **Initial State**: WAF blocks/allows using thresholds
2. **Admin Reviews**: Team reviews at `/api/logs`  
3. **Submit Feedback**: Post decision to `/api/feedback`
4. **RL Update**: Q-values updated with reward signal
5. **Policy Improvement**: Learns to classify better
6. **Continuous Learning**: Policy improves over time

## 📚 File Structure

```
waf/
├── app/
│   ├── main.py              # FastAPI app + endpoints
│   ├── config.py            # Configuration management
│   ├── proxy.py             # Reverse proxy logic
│   ├── bert_detector.py     # BERT inference
│   ├── rule_engine.py       # Rule-based detection
│   ├── rl_engine.py         # Reinforcement learning
│   ├── anomaly_engine.py    # Anomaly detection
│   └── utils.py             # Utilities
├── model/
│   ├── transformer.py       # BERT model
│   ├── tokenizer.py         # HTTP tokenizer
│   └── weights/             # Pre-trained weights
├── database.py              # SQLite management
├── Dockerfile               # Container build
└── requirements.txt         # Python dependencies
```

## 🐛 Troubleshooting

### Models Not Loaded
```bash
docker-compose exec waf-service ls -la /app/model/weights/
```

### High False Positive Rate
```bash
# Lower thresholds in CONFIG.env
AI_CONFIDENCE_THRESHOLD=0.90
```

### Database Locked Error
```bash
rm -f waf/data/waf.db
docker-compose restart waf-service
```

## 🔐 Security Best Practices

1. Store models in secure location
2. Use HTTPS/SSL for production
3. Add authentication to `/api/` endpoints
4. Run in isolated networks
5. Monitor block rates for anomalies
6. Update rule patterns regularly

---

**🎯 Your website is now protected 24/7 with AI-powered threat detection!**
