# Transformer-based Web Application Firewall (WAF)

![Status](https://img.shields.io/badge/Status-Operational-brightgreen)
![Model](https://img.shields.io/badge/Model-SecureBERT%20(Finetuned)-blue)
![Platform](https://img.shields.io/badge/Platform-Docker%20%7C%20Nginx%20%7C%20Python-orange)

An intelligent, self-learning Web Application Firewall that uses **SecureBERT** (a Transformer model) to detect and block zero-day web attacks in real-time. Unlike traditional WAFs that rely on thousands of static regex rules, this system learns the *semantic meaning* of malicious payloads.

## 🚀 Features
- **AI-Powered Detection**: Uses a fine-tuned BERT model to classify HTTP requests.
- **Real-Time Protection**: Blocks SQL Injection (SQLi), XSS, Path Traversal, and Command Injection in <50ms.
- **Fail-Safe Architecture**: Designed to "fail open" if the AI service is unreachable, ensuring app availability.
- **Plug-and-Play**: Runs as a Docker sidecar; no code changes required in your application.

## 🏗️ Architecture
The system sits in front of your application as a reverse proxy.

```
[Attacker/User] 
      |
      v
[Nginx Gateway] --(1. Check Request)--> [WAF Service (SecureBERT)]
      |                                         |
      |                                  (2. Analyze Token Sequence)
      |                                         v
      |                                  [Model Inference]
      |                                         |
      |                                  (3. Allow/Block Decision)
      |                                         |
      | <----(4. Return 200 or 403)-------------+
      |
      +---[200 OK]---> [Your Application] (e.g., Juice Shop)
      |
      +---[403 Forbidden]---> [Block Page]
```

## 🛠️ Getting Started

### Prerequisites
- **Docker** and **Docker Compose** installed on your machine.
- **Git** to clone the repository.
- (Optional) **Python 3.10+** for running local scripts.

### 1. Installation
Clone the repository and navigate to the project folder:
```bash
git clone https://github.com/PriscillajospinG/transformer-waf-test.git
cd transformer-waf-test
```

### 2. Start the System
Launch the entire stack (WAF + Nginx + Vulnerable App) with one command:
```bash
docker-compose up -d --build
```
*Note: The first run may take a few minutes to download the BERT model and build the containers.*

### 3. Verification
Once the containers are running (`docker ps`), verifying the protection is simple:

**Run the Automated Test Suite:**
```bash
python3 scripts/verify_waf.py
```
*You should see `Passed: 6, Failed: 0` indicating that benign requests were allowed and attacks were blocked.*

## 🧪 Manual Testing
You can try attacking the system yourself!

**1. Normal Access (Should work):**
```bash
curl -I "http://localhost:8080/rest/products/search?q=apple"
# Returns: HTTP/1.1 200 OK
```

**2. SQL Injection Attack (Should be blocked):**
```bash
curl -I "http://localhost:8080/rest/products/search?q=' OR 1=1 --"
# Returns: HTTP/1.1 403 Forbidden
```

**3. XSS Attack (Should be blocked):**
```bash
curl -I "http://localhost:8080/rest/products/search?q=<script>alert(1)</script>"
# Returns: HTTP/1.1 403 Forbidden
```

## 📂 Project Structure
| Directory | Description |
|-----------|-------------|
| `waf/` | **Core Logic**. Contains the Python code for the AI Service, Model (`securebert`), and API. |
| `nginx/` | **Gateway**. Nginx configuration that routes traffic and integrates the WAF. |
| `scripts/` | **Tools**. Python scripts to generate traffic, train the model, and verify functionality. |
| `docker-compose.yml` | **Orchestration**. Defines how the containers (App, WAF, Nginx) talk to each other. |

## 🔧 continuous Learning
If the model blocks a legitimate request (False Positive), you can "teach" it to correct itself without restarting the system.

1. Identify the blocked request (e.g., `q=select`).
2. Run the fix script:
   ```bash
   python3 scripts/fix_false_positive.py
   ```
   *This script fine-tunes the model on the new example and restarts the service automatically.*

## 🤝 Contributing
Feel free to fork this project and submit Pull Requests! We are looking for:
- More diverse training datasets.
- Support for other architectures (e.g., LSTM, CNN).
- Dashboard for visualizing blocked attacks.

---
**Disclaimer**: This project is for educational and defensive purposes only. Do not use the attack tools on servers you do not own.