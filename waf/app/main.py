from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import torch
import os
import sys
import re
import time
from collections import deque
from datetime import datetime

# Ensure we can import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import WAFTransformer
from model.tokenizer import HttpTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("waf-service")

app = FastAPI()

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model & Tokenizer
model = None
tokenizer = None
device = torch.device("cpu")

# ===== DASHBOARD DATA =====
request_log = deque(maxlen=200)
stats = {
    "total_requests": 0,
    "blocked": 0,
    "allowed": 0,
    "attack_types": {
        "sqli": 0,
        "xss": 0,
        "path_traversal": 0,
        "encoding": 0,
        "other": 0
    }
}

def classify_attack(uri, keywords):
    """Classify the attack type based on keywords/patterns"""
    uri_lower = uri.lower()
    if any(k in uri_lower for k in ['select', 'union', 'drop', 'insert', 'or 1=1', 'and 1=1', "'"]):
        return "sqli"
    if any(k in uri_lower for k in ['script', 'alert', 'onerror', 'onclick', 'onload', 'eval']):
        return "xss"
    if any(k in uri_lower for k in ['../', '..\\', 'etc/', 'passwd', 'shadow']):
        return "path_traversal"
    if any(k in uri_lower for k in ['%00', '%2e']):
        return "encoding"
    return "other"

def log_request(ip, uri, method, status, reason, prob=0.0):
    """Log a request for the dashboard"""
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "ip": ip,
        "method": method,
        "uri": uri[:120],
        "status": status,
        "reason": reason,
        "probability": round(prob, 4)
    }
    request_log.appendleft(entry)
    stats["total_requests"] += 1
    if status == "blocked":
        stats["blocked"] += 1
        attack_type = classify_attack(uri, [])
        stats["attack_types"][attack_type] += 1
    else:
        stats["allowed"] += 1

class TestRequest(BaseModel):
    url: str

# ===== ZERO-DAY DETECTION CONFIGURATION =====
# Very high threshold to avoid false positives while catching clear attacks
AI_CONFIDENCE_THRESHOLD = 0.95  # Very high: only block if extremely confident it's an attack
UNCERTAINTY_THRESHOLD = 0.85    # If 0.85 < prob < 0.95, it's uncertain - flag it
MAX_ALLOWED_SPECIAL_CHARS = 10
SUSPICIOUS_KEYWORDS = [
    'union', 'select', 'drop', 'insert', 'delete', 'update', 'exec', 'script',
    'eval', 'alert', 'onclick', 'onerror', 'onload', 'passwd', 'shadow',
    'cat', 'ls', 'wget', 'curl', 'bash', 'cmd', 'command', 'shell',
    'or 1=1', 'and 1=1', '../', '..\\', 'etc/', 'var/', 'opt/', 'root/',
    '%00', '/*', '*/', '--', ';', '|', '&', '`', '${'
]

def load_artifacts():
    global model, tokenizer
    try:
        # Paths are absolute in Docker
        TOKENIZER_PATH = "/app/model/weights/tokenizer.json"
        MODEL_PATH = "/app/model/weights/waf_model.pth"
        
        logger.info(f"Loading SecureBERT tokenizer...")
        tokenizer = HttpTokenizer()
        # We try to load local if exists, else it pulls from hub inside the class
        if os.path.exists(os.path.dirname(TOKENIZER_PATH)):
             tokenizer.load(TOKENIZER_PATH)
        
        logger.info(f"Loading SecureBERT model from {MODEL_PATH}...")
        model = WAFTransformer(num_classes=2)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        else:
            logger.warning("Model weights not found! Running with random weights (please train first).")
            
        model.to(device)
        model.eval()
        logger.info("Artifacts loaded successfully.")
        logger.info(f"Zero-day detection enabled with threshold: {AI_CONFIDENCE_THRESHOLD}")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        tokenizer = None
        model = None

@app.on_event("startup")
async def startup_event():
    load_artifacts()

def preprocess_request(method, uri, body=""):
    """Normalize request for model input"""
    uri_norm = re.sub(r'\d+', '{ID}', uri)
    text = f"{method} {uri_norm}"
    return text

def detect_suspicious_keywords(uri):
    """
    Rule-based detection: Check for known malicious keywords
    Returns: (is_suspicious, matched_keywords)
    """
    uri_lower = uri.lower()
    matched = []
    
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in uri_lower:
            matched.append(keyword)
    
    return len(matched) > 0, matched

def detect_special_char_anomaly(uri):
    """
    Anomaly detection: Too many special characters may indicate attacks
    Returns: (is_anomaly, special_char_count)
    """
    special_chars = sum(1 for c in uri if c in '!@#$%^&*()[]{};<>|\\"\'-,./?=')
    is_anomaly = special_chars > MAX_ALLOWED_SPECIAL_CHARS
    return is_anomaly, special_chars

def detect_encoding_attacks(uri):
    """
    Detect URL encoding tricks used in zero-day bypass attempts
    """
    suspicious_patterns = [
        r'%[0-9a-f]{2}',      # URL encoding: %00, %2F, etc.
        r'\\x[0-9a-f]{2}',    # Hex encoding
        r'%2e%2e',            # Encoded dot-dot
        r'\.\./|\.\\\.',      # Path traversal
        r'&#[0-9]+;',         # HTML entity encoding
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, uri, re.IGNORECASE):
            return True
    return False

def detect_injection_patterns(uri):
    """
    Detect common injection patterns including novel variations
    """
    injection_patterns = [
        # SQL Injection patterns
        r"('\s*(AND|OR)\s*'?[^']*'?\s*=|\bOR\b.*=|AND\s*\d+=\d+)",
        r"(UNION\s+SELECT|UNION\s+ALL|UNION\s*/\*)",
        r"(DROP|DELETE|INSERT|UPDATE|EXEC|SCRIPT)\s+",
        r"(';\s*(DROP|DELETE|TRUNCATE|INSERT))",
        
        # Command Injection
        r"([`;&|]|\|\||&&|\n|\r)\s*(cat|ls|wget|curl|bash|sh|cmd)",
        
        # XSS patterns
        r"(<\s*script[^>]*>|javascript:|onerror\s*=|onclick\s*=|onload\s*=|eval\()",
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, uri, re.IGNORECASE):
            return True
    return False

@app.get("/")
def health_check():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "type": "SecureBERT",
        "zero_day_protection": True,
        "confidence_threshold": AI_CONFIDENCE_THRESHOLD
    }

# ===== DASHBOARD API =====
@app.get("/api/stats")
def get_stats():
    return stats

@app.get("/api/logs")
def get_logs():
    return list(request_log)

@app.post("/api/test")
async def test_url(req: TestRequest):
    """Test a URL against the WAF from the dashboard"""
    uri = req.url
    ip = "dashboard-test"
    method = "GET"
    
    if model is None or tokenizer is None:
        return {"status": "allowed", "reason": "Model not loaded", "probability": 0}
    
    try:
        # Layer 1: Rule-based
        is_keyword_suspicious, keywords = detect_suspicious_keywords(uri)
        has_encoding_attack = detect_encoding_attacks(uri)
        has_injection = detect_injection_patterns(uri)
        is_char_anomaly, char_count = detect_special_char_anomaly(uri)
        
        if is_keyword_suspicious and has_injection:
            log_request(ip, uri, method, "blocked", "Rule-Based", 1.0)
            return {"status": "blocked", "reason": "Rule-Based Detection", "probability": 1.0}
        
        if has_encoding_attack:
            log_request(ip, uri, method, "blocked", "Encoding Attack", 1.0)
            return {"status": "blocked", "reason": "Encoding Attack", "probability": 1.0}
        
        # Layer 2: AI
        text = preprocess_request(method, uri)
        encoding = tokenizer.encode(text)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=1)
            malicious_prob = probs[0][1].item()
        
        if malicious_prob > AI_CONFIDENCE_THRESHOLD:
            log_request(ip, uri, method, "blocked", f"AI Detection", malicious_prob)
            return {"status": "blocked", "reason": f"AI Detection (prob={malicious_prob:.4f})", "probability": malicious_prob}
        
        is_uncertain = UNCERTAINTY_THRESHOLD <= malicious_prob < AI_CONFIDENCE_THRESHOLD
        if is_uncertain and (is_keyword_suspicious or is_char_anomaly):
            log_request(ip, uri, method, "blocked", "Uncertain + Anomaly", malicious_prob)
            return {"status": "blocked", "reason": f"Uncertain AI + Anomaly (prob={malicious_prob:.4f})", "probability": malicious_prob}
        
        log_request(ip, uri, method, "allowed", "Safe", malicious_prob)
        return {"status": "allowed", "reason": "Safe", "probability": malicious_prob}
    except Exception as e:
        return {"status": "error", "reason": str(e), "probability": 0}

@app.api_route("/analyze", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def analyze_request(request: Request):
    """
    Enhanced analysis with multiple detection layers for zero-day protection
    """
    global model, tokenizer
    
    original_uri = request.headers.get("X-Original-URI", "/")
    original_method = request.headers.get("X-Original-Method", "GET")
    original_ip = request.headers.get("X-Original-IP", "0.0.0.0")
    
    if model is None or tokenizer is None:
        return Response(status_code=200)

    try:
        # ===== LAYER 1: Rule-Based Detection (Fast) =====
        is_keyword_suspicious, keywords = detect_suspicious_keywords(original_uri)
        has_encoding_attack = detect_encoding_attacks(original_uri)
        has_injection = detect_injection_patterns(original_uri)
        is_char_anomaly, char_count = detect_special_char_anomaly(original_uri)
        
        # Quick block if rule-based detection triggers
        if is_keyword_suspicious and has_injection:
            logger.warning(f"BLOCKING via Rule-Based Detection: IP={original_ip} URI={original_uri} Keywords={keywords}")
            log_request(original_ip, original_uri, original_method, "blocked", "Rule-Based", 1.0)
            return Response(content="Blocked by SecureBERT WAF (Rule-Based)", status_code=403)
        
        if has_encoding_attack:
            logger.warning(f"BLOCKING Encoding Attack: IP={original_ip} URI={original_uri}")
            log_request(original_ip, original_uri, original_method, "blocked", "Encoding", 1.0)
            return Response(content="Blocked by SecureBERT WAF (Encoding Attack)", status_code=403)
        
        # ===== LAYER 2: AI-Based Detection (SecureBERT) =====
        text = preprocess_request(original_method, original_uri)
        
        # Tokenize
        encoding = tokenizer.encode(text)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Inference with confidence scores
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=1)
            benign_prob = probs[0][0].item()
            malicious_prob = probs[0][1].item()
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = max(benign_prob, malicious_prob)
        
        # ===== LAYER 3: Uncertainty Detection (flag zero-day candidates) =====
        is_uncertain = UNCERTAINTY_THRESHOLD <= malicious_prob < AI_CONFIDENCE_THRESHOLD
        
        if is_uncertain:
            logger.warning(f"UNCERTAIN PREDICTION (Potential Zero-Day): IP={original_ip} URI={original_uri} MaliciousProb={malicious_prob:.4f} Confidence={confidence:.4f}")
        
        # ===== LAYER 4: Combined Decision Logic =====
        # Block if:
        # 1. AI says malicious (prob > threshold)
        # 2. Uncertain + has anomalies
        # 3. Character anomaly + suspicious keywords
        
        should_block = False
        block_reason = ""
        
        if malicious_prob > AI_CONFIDENCE_THRESHOLD:
            should_block = True
            block_reason = f"AI Detection (prob={malicious_prob:.4f})"
        elif is_uncertain and (is_keyword_suspicious or is_char_anomaly):
            should_block = True
            block_reason = f"Uncertain AI + Anomaly Detection (prob={malicious_prob:.4f}, fields={[k for k in ['keywords', 'charset'] if (k=='keywords' and is_keyword_suspicious) or (k=='charset' and is_char_anomaly)]})"
        elif is_char_anomaly and is_keyword_suspicious:
            should_block = True
            block_reason = f"Character Anomaly + Keywords (special_chars={char_count})"
        
        # Logging
        log_msg = f"IP={original_ip} URI={original_uri} BenignProb={benign_prob:.4f} MaliciousProb={malicious_prob:.4f} Confidence={confidence:.4f}"
        
        if should_block:
            logger.warning(f"BLOCKING: {log_msg} Reason={block_reason}")
            log_request(original_ip, original_uri, original_method, "blocked", block_reason, malicious_prob)
            return Response(content=f"Blocked by SecureBERT WAF ({block_reason})", status_code=403)
        else:
            logger.info(f"ALLOWED: {log_msg}")
            log_request(original_ip, original_uri, original_method, "allowed", "Safe", malicious_prob)
            return Response(status_code=200)
        
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        # Fail open - allow request if error
        return Response(status_code=200)
