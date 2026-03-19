import json
import logging
import os
import re
import sys
from collections import deque
from datetime import datetime, timezone
from threading import Lock
from urllib.parse import unquote_plus

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Ensure sibling imports work in container and local runs.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tokenizer import HttpTokenizer
from model.transformer import WAFTransformer


# ---------- Logging ----------
logger = logging.getLogger("waf-service")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.handlers = [handler]


def log_event(level: str, event: str, **fields):
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "event": event,
    }
    payload.update(fields)
    message = json.dumps(payload, ensure_ascii=True)
    if level == "debug":
        logger.debug(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "critical":
        logger.critical(message)
    else:
        logger.info(message)


# ---------- App ----------
app = FastAPI(title="SecureBERT WAF", version="2.0.0")

allowed_origins = [
    x.strip()
    for x in os.getenv("ALLOWED_ORIGINS", "http://localhost:8080").split(",")
    if x.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=True,
)


# ---------- Globals ----------
device = torch.device("cpu")
model = None
tokenizer = None

stats_lock = Lock()
all_request_log = deque(maxlen=1200)  # ONLY source of truth - stores all requests
stats = {
    "total_requests": 0,
    "blocked": 0,
    "allowed": 0,
    "attack_types": {
        "sqli": 0,
        "xss": 0,
        "path_traversal": 0,
        "encoding": 0,
        "other": 0,
    },
}

API_TOKEN = os.getenv("API_TOKEN", "secure-api-token-change-me")
MAX_URI_LEN = 4096
MAX_TEST_URL_LEN = 2048
AI_CONFIDENCE_THRESHOLD = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.95"))
UNCERTAINTY_THRESHOLD = float(os.getenv("UNCERTAINTY_THRESHOLD", "0.85"))
MAX_ALLOWED_SPECIAL_CHARS = int(os.getenv("MAX_ALLOWED_SPECIAL_CHARS", "10"))

SUSPICIOUS_KEYWORDS = [
    "union",
    "select",
    "drop",
    "insert",
    "delete",
    "update",
    "exec",
    "script",
    "eval",
    "alert",
    "onclick",
    "onerror",
    "onload",
    "passwd",
    "shadow",
    "or 1=1",
    "and 1=1",
    "../",
    "..\\",
    "%00",
    "--",
    ";",
    "|",
]

EXCLUDED_URI_SUBSTRINGS = (
    "/socket.io/",
    "/favicon.ico",
    "/assets/",
    "/static/",
)

EXCLUDED_QUERY_MARKERS = (
    "transport=polling",
    "transport=websocket",
    "eio=",
)


class TestRequest(BaseModel):
    url: str = Field(min_length=1, max_length=MAX_TEST_URL_LEN)

    @validator("url")
    def validate_url(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("URL cannot contain null bytes")
        return value


class DecisionResponse(BaseModel):
    status: str
    reason: str
    probability: float


def get_stats_snapshot():
    with stats_lock:
        total = stats["total_requests"]
        blocked = stats["blocked"]
        blocked_percentage = round((blocked / total) * 100, 2) if total else 0.0
        return {
            "total_requests": total,
            "blocked": blocked,
            "allowed": stats["allowed"],
            "attack_types": dict(stats["attack_types"]),
            "blocked_percentage": blocked_percentage,
        }


def classify_attack(uri: str):
    uri_lower = uri.lower()
    if any(k in uri_lower for k in ["select", "union", "drop", "or 1=1", "and 1=1", "'"]):
        return "sqli"
    if any(k in uri_lower for k in ["script", "alert", "onerror", "onclick", "onload", "eval"]):
        return "xss"
    if any(k in uri_lower for k in ["../", "..\\", "etc/", "passwd", "shadow"]):
        return "path_traversal"
    if any(k in uri_lower for k in ["%00", "%2e", "&#"]):
        return "encoding"
    return "other"


def should_display_log(log_entry: dict) -> bool:
    """
    Determine if a log entry should be displayed in the frontend dashboard.
    
    Priority 1 (highest): EXCLUDE noisy URIs completely (socket.io, favicon, etc)
    Priority 2: INCLUDE blocked/suspicious requests from meaningful URIs
    Priority 3: INCLUDE allowed requests from meaningful URIs
    Priority 4 (lowest): EXCLUDE allowed requests from noisy URIs
    
    This function is applied ONLY when returning from /api/logs endpoint.
    All logs are stored internally in all_request_log for forensics.
    """
    uri = (log_entry.get("uri", "") or "").lower()
    status = log_entry.get("status", "allowed")
    method = (log_entry.get("method", "") or "").upper()
    
    # PRIORITY 1: Filter out noisy URI patterns FIRST (always exclude, regardless of status)
    # This prevents socket.io/favicon/assets from appearing even if marked as "blocked"
    if any(fragment in uri for fragment in EXCLUDED_URI_SUBSTRINGS):
        return False
    
    if any(marker in uri for marker in EXCLUDED_QUERY_MARKERS):
        return False
    
    if method == "OPTIONS":
        return False
    
    # PRIORITY 2: After noise filtering, include all meaningful traffic
    # API operations always visible
    if uri.startswith("/api/"):
        return True
    
    # Blocked requests (from non-noisy URIs) always visible for security
    if status == "blocked":
        return True
    
    # Allowed requests (from non-noisy URIs) visible
    if status == "allowed":
        return True
    
    # Fallback: include if not explicitly excluded
    return True


def log_request(ip: str, uri: str, method: str, status: str, reason: str, probability: float):
    """
    Store request log entry. NO filtering here - all requests stored for forensics.
    Filtering is applied only when returning logs from /api/logs endpoint.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ip": ip,
        "method": method,
        "uri": uri[:180],
        "status": status,
        "reason": reason,
        "probability": round(float(probability), 4),
    }

    with stats_lock:
        # Store ALL requests for forensic analysis and real-time flexibility
        all_request_log.appendleft(entry)

        # Update statistics (counts all requests, including noise)
        stats["total_requests"] += 1
        if status == "blocked":
            stats["blocked"] += 1
            stats["attack_types"][classify_attack(uri)] += 1
        else:
            stats["allowed"] += 1


def verify_api_token(request: Request):
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid authentication scheme")
    token = auth_header[7:]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API token")


def preprocess_request(method: str, uri: str):
    uri = uri[:MAX_URI_LEN]
    uri_norm = re.sub(r"\d+", "{ID}", uri)
    return f"{method} {uri_norm}"


def detect_suspicious_keywords(uri: str):
    uri_lower = uri.lower()
    matched = [k for k in SUSPICIOUS_KEYWORDS if k in uri_lower]
    return len(matched) > 0, matched


def detect_special_char_anomaly(uri: str):
    special_chars = sum(1 for c in uri if c in '!@#$%^&*()[]{};<>|\\"\'-,./?=')
    return special_chars > MAX_ALLOWED_SPECIAL_CHARS, special_chars


def detect_encoding_attacks(uri: str):
    patterns = [
        r"\\x[0-9a-f]{2}",
        r"%2e%2e|%252e%252e",
        r"%2f|%5c|%252f|%255c",
        r"%00|%2500",
        r"%3c|%3e|%22|%27|%3b|%2d%2d",
        r"\.\./|\.\\\\",
        r"&#[0-9]+;",
    ]
    return any(re.search(pattern, uri, re.IGNORECASE) for pattern in patterns)


def detect_injection_patterns(uri: str):
    patterns = [
        r"('\s*(AND|OR)\s*'?[^']*'?\s*=|\bOR\b.*=|AND\s*\d+=\d+)",
        r"(UNION\s+SELECT|UNION\s+ALL|UNION\s*/\*)",
        r"(DROP|DELETE|INSERT|UPDATE|EXEC|SCRIPT)\s+",
        r"(' ;\s*(DROP|DELETE|TRUNCATE|INSERT))",
        r"([`;&|]|\|\||&&|\n|\r)\s*(cat|ls|wget|curl|bash|sh|cmd)",
        r"(<\s*script[^>]*>|javascript:|onerror\s*=|onclick\s*=|onload\s*=|eval\()",
    ]
    return any(re.search(pattern, uri, re.IGNORECASE) for pattern in patterns)


def infer_malicious_probability(method: str, uri: str):
    if model is None or tokenizer is None:
        return 0.0, "model_unavailable"

    try:
        text = preprocess_request(method, uri)
        encoded = tokenizer.encode(text)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=1)
            return float(probs[0][1].item()), "ok"
    except Exception as exc:
        log_event("error", "model_inference_failed", error=str(exc), method=method, uri=uri[:128])
        return 0.0, "inference_error"


def evaluate_request(method: str, uri: str):
    decoded_uri = unquote_plus(unquote_plus(uri))
    normalized = decoded_uri.lower()

    direct_signatures = [
        "or 1=1",
        "and 1=1",
        "union select",
        "<script",
        "javascript:",
        "../",
        "..\\",
        "%00",
    ]
    if any(sig in normalized for sig in direct_signatures):
        return "blocked", "direct_signature_match", 1.0

    is_keyword_suspicious, keywords = detect_suspicious_keywords(decoded_uri)
    has_encoding_attack = detect_encoding_attacks(uri)
    has_injection = detect_injection_patterns(decoded_uri)
    is_char_anomaly, char_count = detect_special_char_anomaly(decoded_uri)
    has_signal = is_keyword_suspicious or has_encoding_attack or has_injection or is_char_anomaly

    if is_keyword_suspicious and has_injection:
        return "blocked", "rule_based_sqli_xss", 1.0

    if has_encoding_attack:
        return "blocked", "encoding_attack", 1.0

    malicious_prob, infer_state = infer_malicious_probability(method, decoded_uri)

    if infer_state == "inference_error":
        return "allowed", "fail_open_inference_error", 0.0

    if malicious_prob > AI_CONFIDENCE_THRESHOLD and has_signal:
        return "blocked", f"ai_high_confidence:{malicious_prob:.4f}", malicious_prob

    is_uncertain = UNCERTAINTY_THRESHOLD <= malicious_prob < AI_CONFIDENCE_THRESHOLD
    if is_uncertain and (is_keyword_suspicious or is_char_anomaly):
        return "blocked", f"uncertain_plus_anomaly:{malicious_prob:.4f}", malicious_prob

    if is_char_anomaly and is_keyword_suspicious:
        return "blocked", f"char_anomaly_plus_keywords:{char_count}", malicious_prob

    return "allowed", "safe", malicious_prob


def load_artifacts():
    global model, tokenizer
    tokenizer_path = "/app/model/weights/tokenizer.json"
    model_path = "/app/model/weights/waf_model.pth"

    try:
        tokenizer = HttpTokenizer(max_len=256)
        if os.path.exists(os.path.dirname(tokenizer_path)):
            tokenizer.load(tokenizer_path)

        model = WAFTransformer(num_classes=2)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            log_event("info", "model_weights_loaded", model_path=model_path)
        else:
            log_event("warning", "model_weights_missing", model_path=model_path)

        model.to(device)
        model.eval()
        log_event("info", "artifacts_loaded", tokenizer_model="bert-base-uncased")
    except Exception as exc:
        model = None
        tokenizer = None
        log_event("critical", "artifact_load_failed", error=str(exc))


@app.on_event("startup")
async def on_startup():
    load_artifacts()


@app.get("/")
def health_check():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "type": "SecureBERT",
        "confidence_threshold": AI_CONFIDENCE_THRESHOLD,
    }


@app.get("/api/stats")
def get_stats(request: Request):
    verify_api_token(request)
    return get_stats_snapshot()


@app.get("/api/logs")
def get_logs(request: Request):
    """
    Return filtered logs suitable for frontend dashboard display.
    - Applies should_display_log filter to all_request_log
    - Returns last 50-100 meaningful logs
    - Includes debug info: total stored vs returned
    """
    verify_api_token(request)
    
    with stats_lock:
        total_stored = len(all_request_log)
        
        # Filter logs and reverse to show most recent first
        filtered_logs = [
            log for log in all_request_log 
            if should_display_log(log)
        ]
        
        # Limit to last 100 meaningful logs for performance
        displayed_logs = filtered_logs[:100]
        
        # Calculate filtering stats
        filtered_count = len(filtered_logs)
        filtered_out = total_stored - filtered_count
        
        return {
            "logs": displayed_logs,
            "debug": {
                "total_stored": total_stored,
                "total_displayed": len(displayed_logs),
                "filtered_out": filtered_out,
                "filter_ratio": round((filtered_out / total_stored * 100), 1) if total_stored > 0 else 0.0,
            }
        }


@app.post("/api/test", response_model=DecisionResponse)
def test_url(req: TestRequest, request: Request):
    verify_api_token(request)
    uri = req.url
    method = "GET"
    ip = "dashboard-test"

    status, reason, probability = evaluate_request(method, uri)
    log_request(ip, uri, method, status, reason, probability)
    return DecisionResponse(status=status, reason=reason, probability=probability)


@app.api_route("/analyze", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def analyze_request(request: Request):
    # Receive metadata from nginx auth_request subrequest.
    original_uri = request.headers.get("X-Original-URI", "/")
    original_query = request.headers.get("X-Original-Query", "")
    original_method = request.headers.get("X-Original-Method", "GET")
    original_ip = request.headers.get("X-Original-IP", "0.0.0.0")

    full_uri = f"{original_uri}?{original_query}" if original_query else original_uri
    full_uri = full_uri[:MAX_URI_LEN]

    try:
        status, reason, probability = evaluate_request(original_method, full_uri)
        log_request(original_ip, full_uri, original_method, status, reason, probability)
        log_event(
            "warning" if status == "blocked" else "info",
            "request_decision",
            action=status,
            reason=reason,
            ip=original_ip,
            method=original_method,
            uri=full_uri[:180],
            probability=round(probability, 4),
        )

        # nginx auth_request accepts only 2xx to allow, 401/403 to deny.
        if status == "blocked":
            return Response(status_code=403)
        return Response(status_code=200)
    except Exception as exc:
        # Fail-open for availability: do not break normal traffic.
        log_event("critical", "analyze_unexpected_error", error=str(exc), ip=original_ip)
        return Response(status_code=200)
