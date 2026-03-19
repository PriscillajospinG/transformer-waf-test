# SecureBERT WAF Project Guide (Student-Friendly)

This document explains your project end-to-end in simple language so you can:
- run it confidently
- debug it quickly
- explain it clearly in viva

---

## 1. How To Run The Project

### Option A: Run with Docker (recommended)

### Step 1: Go to project folder
```bash
cd "/Users/priscillajosping/Desktop/Mini Project/transformer-waf-test"
```
What this does:
- Moves your terminal into the project root where docker-compose.yml exists.

### Step 2: Start all services
```bash
docker-compose up -d --build
```
What this does:
- Builds the WAF image from waf/Dockerfile.
- Starts these containers:
  - backend-app (default: OWASP Juice Shop)
  - waf-service (FastAPI + SecureBERT)
  - waf-nginx (reverse proxy + WAF gatekeeper)

### Step 3: Check container status
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```
What to expect:
- waf-service should become healthy.
- waf-nginx should be up on port 8080.
- backend-app should be up.

### Step 4: Open the app and dashboard
- Protected app: http://localhost:8080/
- Dashboard: http://localhost:8080/dashboard/

### Step 5: Verify API (dashboard backend)
```bash
curl -H "Authorization: Bearer secure-api-token-change-me" http://localhost:8080/api/stats
```
Expected:
- HTTP 200 and JSON stats.

---

### Option B: Local run (without Docker)

This is useful for learning/debugging, but Docker is easier for demos.

You need 3 terminal sessions:

1) Start backend app (example: Juice Shop) on port 3000.

2) Start FastAPI WAF:
```bash
cd waf
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

3) Start nginx using your template rendered into nginx.conf and point it to localhost backend/waf.

Note:
- Local mode is mainly for debugging code.
- Docker mode is best for full architecture testing.

---

## 2. What Is Happening Internally

When a client sends a request:

1. Request first hits nginx container (waf-nginx).
2. nginx does not directly forward to backend immediately.
3. nginx sends an internal subrequest to FastAPI endpoint: /analyze.
4. FastAPI reads request metadata (URI, query, method, client IP).
5. FastAPI runs rule checks + SecureBERT model checks.
6. FastAPI returns:
   - 200 -> allow request
   - 403 -> block request
7. nginx:
   - if allow: forwards request to backend app
   - if block: returns block response page/message

How logs are generated:
- nginx writes JSON access logs to nginx/logs/access.log
- FastAPI writes structured JSON events to container logs
- Dashboard consumes /api/stats and /api/logs

---

## 3. How The WAF Works

## 3.1 Request interception
- nginx location / applies auth_request /_waf_check for all app traffic.
- This guarantees no direct bypass for normal app routes.

## 3.2 /analyze endpoint behavior
- Endpoint: FastAPI /analyze
- Input comes through headers set by nginx:
  - X-Original-URI
  - X-Original-Query
  - X-Original-Method
  - X-Original-IP

## 3.3 Decision flow (layers)
Layer 1: direct signatures
- immediate block for obvious patterns like:
  - OR 1=1
  - UNION SELECT
  - <script>
  - ../

Layer 2: rule/anomaly checks
- suspicious keywords
- encoding abuse patterns
- injection regex patterns
- special character anomaly

Layer 3: SecureBERT model scoring
- URI + method are tokenized
- model outputs malicious probability

Layer 4: final decision logic
- block if strong malicious evidence
- allow otherwise
- fail-open on internal model errors (availability-first)

Output contract:
- only 200 (allow) or 403 (block) for /analyze

---

## 4. Model Explanation (SecureBERT)

What is SecureBERT here:
- A Transformer-based classifier (PyTorch) used to score malicious likelihood.

Input to model:
- Normalized text built from request metadata (mainly method + URI/query).

Tokenizer behavior:
- Uses bert-base-uncased tokenizer.
- Max length set to 256 tokens.
- Handles null bytes and bad input safely.

Model output:
- Two-class probability distribution (benign vs malicious).
- WAF uses malicious probability for decisions.

Threshold idea:
- If malicious probability is very high and other malicious signals exist, block.
- If uncertain or low confidence, use rule/anomaly context.

---

## 5. Docker Architecture (Plug-and-Play)

Why separate containers:
- backend-app: your actual protected application.
- waf-service: AI detection engine (FastAPI + model).
- waf-nginx: traffic gatekeeper and router.

How networking works:
- All containers are on internal bridge network waf-net.
- nginx calls waf-service and backend-app by container name.

Why plug-and-play:
- You can replace backend image/container without touching app code.
- Main user config is environment variables:
  - APP_HOST
  - APP_PORT
  - optional APP_IMAGE

Example:
```bash
APP_HOST=my-django-container APP_PORT=8000 docker-compose up -d --build
```

---

## 6. Real-World Connection

How this is similar to Cloudflare/AWS WAF:
- reverse-proxy enforcement in front of app
- request inspection before origin
- allow/block based on policy and detection
- logging and observability

What to improve for true production:
1. mTLS between components
2. distributed rate limiting (Redis)
3. model versioning + canary rollout
4. audit-grade SIEM integration
5. centralized metrics (Prometheus/Grafana)
6. async queue for heavy model inference
7. stronger block response UX and incident IDs
8. policy management UI and API

---

## 7. Debugging Guide

## 7.1 If nginx issue suspected
Check logs:
```bash
docker logs waf-nginx --tail 200
```
Check rendered config:
```bash
docker exec waf-nginx cat /etc/nginx/nginx.conf
```
Validate nginx config:
```bash
docker exec waf-nginx nginx -t
```

Common nginx problems:
- upstream host not found -> APP_HOST wrong or backend not on same network
- 500 on root -> auth_request metadata mismatch or waf-service unavailable
- 403 everywhere -> overly strict detection or bad header mapping

## 7.2 If FastAPI issue suspected
Check logs:
```bash
docker logs waf-service --tail 200
```
Health check:
```bash
curl http://localhost:8000/
```

Common FastAPI problems:
- model not loaded -> wrong model path or weight file issue
- 403 api/stats -> invalid API token
- startup slow -> model warmup time expected

## 7.3 If model issue suspected
Symptoms:
- many false positives
- too many fail-open logs

Actions:
- inspect request_decision log events
- tune thresholds via env
- test with known benign and malicious samples

---

## 8. Simple Summary (5-6 lines for viva)

1. My project is an AI-powered WAF using SecureBERT in front of a web app.
2. Every request first hits nginx, which asks FastAPI /analyze whether to allow or block.
3. FastAPI applies rule checks plus Transformer model scoring to detect attacks.
4. If request is safe, nginx forwards it to the backend app; otherwise it returns 403.
5. The system is plug-and-play because backend host/port are configurable by environment.
6. I validated it with normal, SQL injection, and XSS requests.

---

# 2-Minute Viva Script

Problem:
- Traditional rule-only WAFs miss new attack variants and produce false positives.

Solution:
- I built SecureBERT WAF: nginx + FastAPI + Transformer model in a containerized architecture.

Architecture:
- Client -> nginx -> FastAPI /analyze -> SecureBERT -> decision -> backend app.
- Dashboard APIs show stats and decisions.

How it works:
- nginx intercepts every request and sends metadata to /analyze.
- FastAPI runs layered detection: signatures, anomaly checks, and SecureBERT probability.
- It returns 200 to allow or 403 to block.

Why unique:
- It combines deterministic security rules with Transformer intelligence.
- It is plug-and-play: user can protect almost any containerized app by setting APP_HOST and running docker-compose.

---

## Quick Demo Checklist

1. Start stack:
```bash
docker-compose up -d --build
```

2. Normal request (should allow):
```bash
curl -i "http://localhost:8080/"
```

3. SQLi test (should block):
```bash
curl -i --get "http://localhost:8080/" --data-urlencode "q=' OR 1=1 --"
```

4. XSS test (should block):
```bash
curl -i --get "http://localhost:8080/" --data-urlencode "q=<script>alert(1)</script>"
```

5. Dashboard stats (authenticated):
```bash
curl -H "Authorization: Bearer secure-api-token-change-me" "http://localhost:8080/api/stats"
```












Perfect bro 🔥 — I’ll teach you this like a mentor, not like a document.
By the end, you should be able to explain + debug + demo confidently.

⸻

🧠 1. BIG PICTURE (Understand First)

👉 Your project is basically:

A smart security gate (WAF) placed in front of a website

Real meaning:
	•	User → comes to website
	•	Your WAF → checks request
	•	If safe → allow
	•	If attack → block

⸻

⚡ 2. WHAT EXACTLY YOU BUILT

You built 3 parts working together:

🧱 Components
	1.	Nginx (Gatekeeper)

	•	First point of contact
	•	Stops or forwards traffic

	2.	FastAPI (Brain controller)

	•	Decides allow/block
	•	Talks to AI model

	3.	SecureBERT (AI model 🧠)

	•	Detects if request is malicious

	4.	Backend App (Juice Shop)

	•	Actual website

⸻

🔄 3. COMPLETE FLOW (MOST IMPORTANT)

When user sends request:

User → Nginx → FastAPI → SecureBERT → Decision → Backend

Step-by-step:

✅ Step 1: Request hits Nginx

Example:

http://localhost:8080/?q=hello

👉 Nginx does NOT send it directly to backend

⸻

✅ Step 2: Nginx asks FastAPI

It sends internal request:

/analyze

With headers:
	•	URI
	•	Method
	•	IP

⸻

✅ Step 3: FastAPI processes

FastAPI does:

🔹 Layer 1: Rules
	•	Check:
	•	OR 1=1
	•	<script>
	•	../

👉 Fast detection

⸻

🔹 Layer 2: Patterns
	•	Regex
	•	suspicious keywords

⸻

🔹 Layer 3: AI (SecureBERT)
Example input:

GET /search?q=' OR 1=1 --

Model says:

Malicious probability = 0.97


⸻

🔹 Layer 4: Final Decision

IF strong attack → BLOCK
ELSE → ALLOW


⸻

✅ Step 4: Response to Nginx

FastAPI returns:
	•	200 → allow
	•	403 → block

⸻

✅ Step 5: Nginx acts
	•	If 200 → send to backend
	•	If 403 → stop request

⸻

🧠 4. SIMPLE UNDERSTANDING (SUPER IMPORTANT)

👉 Think like this:

Component	Role
Nginx	Security guard
FastAPI	Manager
SecureBERT	Intelligence expert
Backend	Actual office


⸻

🐳 5. DOCKER (WHY 2–3 CONTAINERS?)

👉 You separated things:

Container 1:
	•	WAF (Nginx + FastAPI + AI)

Container 2:
	•	User app

👉 Why?

✔ Easy to plug
✔ Works with any app
✔ Clean architecture

⸻

🔌 6. WHY THIS IS “PLUG AND PLAY”

User just does:

docker-compose up

OR:

APP_HOST=my-app docker-compose up

👉 That’s it

No code change needed

⸻

⚠️ 7. MOST COMMON ERRORS (IMPORTANT FOR VIVA)

❌ Problem: Everything blocked

👉 Cause:
	•	rule too strict
	•	model threshold too low

⸻

❌ Problem: Nothing blocked

👉 Cause:
	•	nginx not calling /analyze
	•	headers not passed

⸻

❌ Problem: 502 error

👉 Cause:
	•	waf-service not running

⸻

❌ Problem: model not working

👉 Cause:
	•	weights not loaded

⸻

🧪 8. HOW YOU TEST (VERY IMPORTANT)

✅ Normal request:

curl http://localhost:8080/

✔ Should work

⸻

❌ SQL Injection:

?q=' OR 1=1 --

✔ Should block

⸻

❌ XSS:

?q=<script>alert(1)</script>

✔ Should block

⸻

🎯 9. HOW TO EXPLAIN IN VIVA (MEMORIZE THIS)

👉 Say this:

“My project is an AI-powered Web Application Firewall using a Transformer model.
Every request is intercepted by Nginx and sent to a FastAPI service.
The system uses rule-based detection and a SecureBERT model to classify requests.
Based on the decision, the request is either allowed or blocked.
The system is plug-and-play and can protect any web application using Docker.”

💥 This alone can get you high marks

⸻

🧠 10. WHAT MAKES YOUR PROJECT STRONG

✔ AI + Security combined
✔ Real-time detection
✔ Plug-and-play architecture
✔ Works with any app
✔ Uses modern Transformer model

⸻

🔥 FINAL SIMPLE SUMMARY

👉 Your system is:

A smart AI firewall that sits in front of a website, checks every request using rules + AI, and blocks attacks before they reach the app.

⸻

🚀 If you want next level (HIGH MARKS)

I can help you with:
	•	🎤 Exact viva questions + answers
	•	🧪 Live demo script (impress examiner)
	•	🧠 Diagram explanation (for PPT)
	•	⚡ What to say if they ask “why transformer?”

Just tell me 👍