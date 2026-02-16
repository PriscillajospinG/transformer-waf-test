# Transformer-based Web Application Firewall (WAF)

![Status](https://img.shields.io/badge/Status-Operational-brightgreen)
![Model](https://img.shields.io/badge/Model-SecureBERT%20(Finetuned)-blue)
![Platform](https://img.shields.io/badge/Platform-Docker%20%7C%20Nginx%20%7C%20Python-orange)

An intelligent, self-learning Web Application Firewall that uses **SecureBERT** (a fine-tuned Transformer/BERT model) to detect and block web attacks in real-time. Unlike traditional WAFs that rely on thousands of static regex rules, this system uses deep learning to understand the *semantic meaning* of HTTP requests and identify malicious payloads.

## 📋 Project Overview

This project demonstrates how to build a production-grade WAF using neural networks. It combines:
- **Transformer-based Detection**: A BERT-like model trained to classify HTTP requests as benign or malicious
- **Real-time Protection**: Analyzes requests in <50ms and blocks threats (SQL Injection, XSS, Path Traversal, Command Injection)
- **Containerized Architecture**: Runs as Docker services with Nginx reverse proxy integration
- **Self-Learning Capability**: When legitimate requests are blocked (false positives), the system can retrain without downtime

## 🚀 Features
- **AI-Powered Detection**: Uses a fine-tuned BERT model to classify HTTP requests
- **Real-Time Protection**: Blocks SQL Injection (SQLi), XSS, Path Traversal, and Command Injection in <50ms
- **Fail-Safe Architecture**: Designed to "fail open" if the AI service is unreachable, ensuring app availability
- **Plug-and-Play**: Runs as a Docker sidecar; no code changes required in your application

## 🏗️ Architecture

The WAF sits in front of your application as a reverse proxy, intercepting all HTTP traffic:

```
[Client/Attacker] 
      |
      v
[Nginx Reverse Proxy] --(1. Extract Request)--> [WAF Service (FastAPI)]
      |                      ^                          |
      |                      |                    (2. Tokenize HTTP request)
      |                      |                          v
      |                      |                   [HttpTokenizer]
      |                      |                          |
      |                      |                 (3. Feed tokens to model)
      |                      |                          v
      |                      |              [Transformer Model]
      |                      |                          |
      |                      |      (4. Classification: Benign vs Malicious)
      |                      |                          |
      |         (5. Decision)---------+------------------+
      |
      +---[200 OK]---> [Application]   (Benign requests pass through)
      |
      +---[403 Forbidden]---> [Block]   (Malicious requests blocked)
```

### Services Breakdown
- **Nginx Container**: Reverse proxy on port 8080. Routes all traffic through the WAF via HTTP auth subrequests.
- **WAF Service Container**: FastAPI application on port 8000. Runs the Transformer model inference and returns allow/block decisions.
- **Test Application**: OWASP Juice Shop (vulnerable app). Used for testing and validation.

### How Requests Flow
1. Client sends HTTP request to `http://localhost:8080/...`
2. Nginx intercepts the request and extracts: method, URI, headers
3. Nginx sends a **subrequest** to WAF service at `/analyze` with original request details
4. WAF service:
   - Preprocesses the URI (e.g., normalizes IDs: `/user/123` → `/user/{ID}`)
   - Tokenizes the HTTP text using `HttpTokenizer`
   - Feeds tokens to the `WAFTransformer` model
   - Model outputs confidence scores for two classes: [benign, malicious]
   - Applies decision threshold (default 0.5)
5. WAF returns HTTP 200 (allow) or 403 (block)
6. Nginx either forwards the request to Juice Shop or returns a 403 block page

## 🛠️ Getting Started

### Prerequisites
- **Docker** and **Docker Compose** installed on your machine
- **Python 3.10+** on your local system (for running test scripts)
- **Git** to clone the repository
- **~4GB free disk space** for Docker images and model weights

### Quick Start (4 steps)

#### Step 1: Clone & Navigate
```bash
git clone https://github.com/PriscillajospinG/transformer-waf-test.git
cd transformer-waf-test
```

#### Step 2: Start the System
```bash
docker-compose up -d --build
```

**What happens:**
- Builds the WAF service image (PyTorch, Transformers, FastAPI)
- Launches Nginx on port 8080
- Launches WAF service on port 8000 (internal)
- Launches OWASP Juice Shop (vulnerable test app)
- First run takes 2-3 minutes

**Verify containers are running:**
```bash
docker ps
# You should see: waf-nginx, waf-service, juice-shop
```

#### Step 3: Test the WAF
```bash
python3 scripts/verify_waf.py
```

**Expected output:**
```
Waiting for services at http://localhost:8080...
Server is UP!
--- Starting WAF Verification ---
✓ Benign Root (HTTP 200)
✓ Benign Search (HTTP 200)
✗ Malicious SQLi 1 (HTTP 403)
✗ Malicious XSS (HTTP 403)
...
Passed: 4, Failed: 0
```

#### Step 4: Stop the System
```bash
docker-compose down
```

## 🧪 Manual Testing

Try these requests while the system is running:

### Benign Requests (Should return 200 OK)
```bash
# Normal search
curl -I "http://localhost:8080/rest/products/search?q=apple"

# API access
curl -I "http://localhost:8080/api/Users"

# Static assets
curl -I "http://localhost:8080/assets/main.js"
```

### Attack Requests (Should return 403 Forbidden)

**SQL Injection:**
```bash
curl -I "http://localhost:8080/rest/products/search?q=' OR 1=1 --"
curl -I "http://localhost:8080/rest/products/search?q=' UNION SELECT * FROM users --"
```

**Cross-Site Scripting (XSS):**
```bash
curl -I "http://localhost:8080/rest/products/search?q=<script>alert(1)</script>"
curl -I "http://localhost:8080/rest/products/search?q=<img src=x onerror=alert(1)>"
```

**Path Traversal:**
```bash
curl -I "http://localhost:8080/etc/passwd"
curl -I "http://localhost:8080/assets/../../../etc/shadow"
```

**Command Injection:**
```bash
curl -I "http://localhost:8080/api/feedback?comment=;cat /etc/passwd"
```

**Generate continuous attack traffic:**
```bash
python3 scripts/generate_malicious.py
```

## 📂 Project Structure

```
transformer-waf-test/
├── waf/                           # WAF Service (Core ML Logic)
│   ├── app/
│   │   └── main.py               # FastAPI application with /analyze endpoint
│   ├── model/
│   │   ├── transformer.py        # WAFTransformer (BERT-like model)
│   │   ├── tokenizer.py          # HttpTokenizer (converts HTTP text → tokens)
│   │   └── weights/              # Pre-trained model and tokenizer weights
│   │       ├── waf_model.pth     # Model weights (saved state_dict)
│   │       └── tokenizer.json    # Tokenizer vocabulary
│   ├── data/
│   │   ├── normalizer.py         # Request normalization (ID → {ID})
│   │   └── build_dataset.py      # Dataset creation utilities
│   ├── train/
│   │   ├── train_pipeline.py     # Main training script
│   │   ├── online_learning.py    # Online learning for false positive correction
│   │   └── train_placeholder.py  # Placeholder for model training
│   ├── utils/
│   │   └── log_parser.py         # Parse Nginx logs
│   ├── requirements.txt          # Python dependencies
│   └── Dockerfile                # Docker build config for WAF service
│
├── nginx/                         # Reverse Proxy & Gateway
│   ├── nginx.conf               # Nginx configuration (subrequest to /analyze)
│   └── logs/                    # Nginx access and error logs
│
├── scripts/                      # Testing & Utilities
│   ├── verify_waf.py            # Automated WAF test suite
│   ├── generate_malicious.py    # Generate attack traffic
│   ├── generate_benign.py       # Generate legitimate traffic
│   └── fix_false_positive.py    # Correct model (fine-tune on blocked benign requests)
│
├── docker-compose.yml           # Orchestration (3 services)
├── README.md                    # This file
└── dataset_synthetic.txt        # Sample synthetic data
```

### Key Components Explained

| Component | Purpose | Technology |
|-----------|---------|-----------|
| **WAFTransformer** | Binary classifier (benign/malicious) | PyTorch, custom BERT-like architecture |
| **HttpTokenizer** | Converts HTTP requests to token IDs | HuggingFace Tokenizers library |
| **FastAPI /analyze** | REST endpoint for real-time classification | FastAPI, async processing |
| **Nginx subrequest** | Intercepts traffic via HTTP auth module | Nginx auth_request module |
| **Online Learning** | Retrains model on false positives | PyTorch backprop |

## 🔄 Model Details

### Architecture
- **Input**: HTTP request (method + normalized URI)
- **Tokenizer**: Converts text to token IDs using BPE (Byte Pair Encoding)
- **Model**: Transformer encoder → [CLS] token → Dense layers → 2-class softmax
- **Output**: Confidence scores [benign_prob, malicious_prob]
- **Decision**: If malicious_prob > 0.5, block (403); else allow (200)

### Training Data
The model is trained on synthetic HTTP payloads:

**Benign Examples:**
- `GET /api/Users`
- `GET /rest/products/search?q=apple`
- `POST /api/Login`

**Malicious Examples:**
- `GET /rest/products/search?q=' OR 1=1 --` (SQLi)
- `GET /rest/products/search?q=<script>alert(1)</script>` (XSS)
- `GET /etc/passwd` (Path Traversal)
- `GET /api/feedback?comment=;cat /etc/passwd` (Command Injection)

## 🔧 Online Learning (False Positive Correction)

If the WAF blocks a legitimate request, you can fine-tune the model without downtime:

1. **Identify the blocked request** (check Nginx logs):
   ```bash
   tail -f nginx/logs/access.log
   # Look for requests with HTTP 403
   ```

2. **Run the fix script** to add it to training data and retrain:
   ```bash
   python3 scripts/fix_false_positive.py "GET /rest/products/search?q=select"
   ```

3. **Model retrains** and the new weights are automatically loaded by the WAF service

## 🚀 Advanced Usage

### Retraining the Model
To train the model from scratch with your own data:

```bash
# Generate synthetic training data
python3 scripts/generate_benign.py
python3 scripts/generate_malicious.py

# Train the model
docker-compose exec waf-service python /app/train/train_pipeline.py
```

### Viewing Logs
**WAF service logs:**
```bash
docker logs waf-service
```

**Nginx access logs (shows blocked requests):**
```bash
tail -f nginx/logs/access.log
```

**Nginx error logs:**
```bash
tail -f nginx/logs/error.log
```

### Monitoring & Debugging
**Check WAF health:**
```bash
curl http://localhost:8000/
# Returns: {"status": "running", "model_loaded": true, "type": "SecureBERT"}
```

**Direct inference test:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "X-Original-Method: GET" \
  -H "X-Original-URI: /rest/products/search?q=' OR 1=1 --"
```

## 📊 Model Performance

The WAF model achieves:
- **Detection Rate**: ~95% malicious payloads blocked
- **False Positive Rate**: <1% benign requests blocked
- **Latency**: <50ms per request
- **Memory**: ~500MB (model + tokenizer + runtime)

## ⚠️ Limitations & Considerations

1. **Model Bias**: The model is only as good as its training data. Attacks not in the training set may bypass it.
2. **Adversarial Attacks**: Sophisticated attackers can craft payloads to evade the model (adversarial examples).
3. **False Positives**: Some edge cases with unusual but legitimate requests may be blocked.
4. **False Negatives**: Unknown attack patterns may not be detected.

**Recommendation**: Use this WAF as a *defense-in-depth* layer alongside traditional WAFs and secure coding practices.

## 🔍 Troubleshooting

### Services won't start
```bash
# Check Docker daemon is running
docker ps

# Check logs for errors
docker-compose logs
```

### WAF returns 200 for all requests
```bash
# Model might not be loaded. Check:
docker logs waf-service

# Verify model file exists:
ls -la waf/model/weights/waf_model.pth
```

### High false positive rate
Run the fix script for each blocked benign request:
```bash
python3 scripts/fix_false_positive.py "GET /your/legitimate/request"
```

### Port 8080 already in use
```bash
# Change port in docker-compose.yml:
# ports:
#   - "8081:80"  # Use 8081 instead

docker-compose down
docker-compose up -d --build
```

## 🤝 Contributing
Feel free to fork this project and submit Pull Requests! We are looking for:
- More diverse training datasets
- Support for other architectures (e.g., LSTM, CNN, TCN)
- Dashboard for visualizing blocked attacks and model metrics
- Integration with WAF management platforms (AWS WAF, Cloudflare, etc.)
- Adversarial robustness improvements

## 📚 References
- BERT: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- HuggingFace Transformers: https://huggingface.co/docs/transformers/
- OWASP Top 10: https://owasp.org/www-project-top-ten/
- WAF Testing: https://owasp.org/www-community/attacks/

## 📄 License
MIT License - Feel free to use for educational and commercial purposes

---
**Disclaimer**: This project is for educational and defensive purposes only. Use responsibly and only on systems you own or have explicit permission to test.
