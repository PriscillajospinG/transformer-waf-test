"""
WAF Main Service - Complete Integration
Reverse proxy with SecureBERT ML detection, rule engine, RL feedback, and anomaly detection
"""

from fastapi import FastAPI, Request, HTTPException, Response, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
import torch
import os
import sys
import time
import uuid
from typing import Optional, Dict
from datetime import datetime

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all detection engines
from model.transformer import WAFTransformer
from model.tokenizer import HttpTokenizer
from app.config import WAFConfig
from app.proxy import reverse_proxy
from app.rule_engine import rule_engine
from app.rl_engine import rl_engine
from app.bert_detector import bert_detector
from app.anomaly_engine import anomaly_engine, EmbeddingExtractor
from app.utils import (
    generate_request_id, extract_state_hash, extract_request_details,
    build_model_input_text, rate_limiter, get_timestamp_iso
)
import database

# Configure logging
logging.basicConfig(
    level=getattr(logging, WAFConfig.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("waf-service")

# Initialize FastAPI app
app = FastAPI(title="SecureBERT WAF", version="2.0.0")

# Global state
model = None
tokenizer = None
device = torch.device("cpu")

# ===== INITIALIZATION =====

def load_model_artifacts():
    """Load BERT model and tokenizer"""
    global model, tokenizer
    try:
        logger.info("Loading SecureBERT model and tokenizer...")
        
        TOKENIZER_PATH = f"{WAFConfig.MODEL_PATH}/tokenizer.json"
        MODEL_PATH = f"{WAFConfig.MODEL_PATH}/waf_model.pth"
        
        # Load tokenizer
        tokenizer = HttpTokenizer()
        try:
            tokenizer.load(TOKENIZER_PATH)
            logger.info("Tokenizer loaded from file")
        except:
            logger.info("Tokenizer not found locally, will initialize fresh")
        
        # Load model
        model = WAFTransformer(num_classes=2)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            logger.info("Model weights loaded from file")
        else:
            logger.warning("Model weights not found, using pretrained initialization")
        
        model.to(device)
        model.eval()
        
        # Setup BERT detector
        bert_detector.set_model(model, tokenizer)
        
        logger.info("Model artifacts loaded successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize WAF on startup"""
    logger.info("=" * 80)
    logger.info("🛡️  SecureBERT WAF Service Starting")
    logger.info("=" * 80)
    
    # Initialize database
    database.init_db()
    logger.info("✓ Database initialized")
    
# ===== HEALTH CHECK =====

@app.get("/")
@app.get("/health")
async def health_check():
    """WAF health check"""
    return JSONResponse({
        "status": "running",
        "service": "SecureBERT WAF",
        "model_loaded": model is not None,
        "version": "2.0.0",
        "features": {
            "ml_detection": model is not None,
            "rule_engine": True,
            "rl_learning": WAFConfig.RL_ENABLED,
            "anomaly_detection": WAFConfig.ANOMALY_ENABLED,
            "admin_api": WAFConfig.ADMIN_API_ENABLED
        }
    })

# ===== MAIN PROXY ENDPOINT =====

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_request(request: Request, path: str, background_tasks: BackgroundTasks):
    """
    Main WAF proxy endpoint
    Analyzes request through all detection layers before forwarding
    """
    
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        # Extract request details
        client_ip = request.client[0] if request.client else "0.0.0.0"
        method = request.method
        query_string = str(request.url.query) if request.url.query else ""
        full_uri = f"/{path}" + (f"?{query_string}" if query_string else "")
        headers = dict(request.headers)
        
        # Rate limiting
        if not rate_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return Response(content="Rate limit exceeded", status_code=429)
        
        # Read body safely
        body_bytes = await request.body()
        body_str = body_bytes.decode('utf-8', errors='ignore')[:1000]  # First 1000 chars
        
        logger.info(f"Request {request_id}: {method} {full_uri} from {client_ip}")
        
        # ===== DETECTION LAYER 1: Rule-Based Engine =====
        rule_score, detected_types = rule_engine.detect(full_uri, body_str, headers)
        
        # ===== DETECTION LAYER 2: BERT ML Model =====
        if model is not None and tokenizer is not None:
            model_text = build_model_input_text(method, full_uri, headers, body_str)
            benign_prob, malicious_prob, bert_confidence = bert_detector.detect(model_text)
            bert_score = malicious_prob
        else:
            benign_prob, malicious_prob, bert_confidence = 0.5, 0.5, 0.5
            bert_score = 0.5
        
        # ===== DETECTION LAYER 3: Anomaly Detection (Zero-Day) =====
        anomaly_score = 0.0
        is_anomaly = False
        anomaly_details = {}
        
        if WAFConfig.ANOMALY_ENABLED and model is not None:
            embedding = bert_detector.get_embedding(model_text)
            if embedding:
                is_anomaly, anomaly_score, anomaly_details = anomaly_engine.detect_anomaly(embedding)
        
        # ===== COMBINED DECISION LOGIC (Multi-Layer) =====
        combined_score = (0.4 * bert_score + 0.3 * rule_score + 0.3 * anomaly_score)
        should_block = False
        block_reasons = []
        
        # Decision rule 1: High BERT confidence
        if bert_score > WAFConfig.AI_CONFIDENCE_THRESHOLD:
            should_block = True
            block_reasons.append(f"AI(score={bert_score:.3f})")
        
        # Decision rule 2: Multiple detection types
        if len(detected_types) >= 2:
            should_block = True
            block_reasons.append(f"Rules({len(detected_types)}:{','.join(detected_types)})")
        
        # Decision rule 3: Anomaly detection + other signals
        if is_anomaly and WAFConfig.BLOCK_ON_ANOMALY:
            if bert_score > 0.5 or len(detected_types) > 0:
                should_block = True
                block_reasons.append(f"Anomaly+Signals")
        
        # Decision rule 4: Uncertain but suspicious
        if WAFConfig.BLOCK_ON_UNCERTAIN:
            if WAFConfig.UNCERTAINTY_THRESHOLD <= bert_score < WAFConfig.AI_CONFIDENCE_THRESHOLD:
                if len(detected_types) > 0 or anomaly_score > WAFConfig.ANOMALY_THRESHOLD:
                    should_block = True
                    block_reasons.append(f"Uncertain+Anomaly")
        
        # ===== RL POLICY APPLICATION =====
        state_hash = extract_state_hash(method, f"/{path}", bert_score, rule_score)
        rl_action = "allow"  # Default
        
        if WAFConfig.RL_ENABLED and model is not None:
            rl_action, rl_q = rl_engine.get_action(state_hash, bert_score, rule_score, method, f"/{path}")
            
            # RL can override decision (if trained with feedback)
            if rl_action == "block" and not should_block:
                logger.info(f"RL Override: blocking based on learned policy (q={rl_q:.4f})")
                should_block = True
                block_reasons.append(f"RL(q={rl_q:.3f})")
        
        # ===== LOGGING & DATABASE =====
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Extract details for logging
        request_details = extract_request_details(method, full_uri, headers, body_str)
        
        # Store in database
        actual_action = "block" if should_block else "allow"
        database.log_request(
            request_id=request_id,
            ip=client_ip,
            method=method,
            endpoint=f"/{path}",
            full_uri=full_uri,
            headers=headers,
            body_preview=body_str[:100],
            bert_score=bert_score,
            rule_score=rule_score,
            anomaly_score=anomaly_score,
            combined_score=combined_score,
            prediction="malicious" if bert_score > 0.5 else "benign",
            action=actual_action,
            blocked=should_block,
            block_reason=",".join(block_reasons) if block_reasons else None,
            processing_time_ms=processing_time_ms
        )
        
        # Store embedding for anomaly learning
        if not should_block and model is not None:
            embedding = bert_detector.get_embedding(model_text)
            if embedding:
                database.save_embedding(request_id, embedding, full_uri, method)
                anomaly_engine.add_benign_embedding(embedding, request_id)
        
        # ===== RESPONSE =====
        
        if should_block:
            logger.warning(f"🚫 BLOCKED {request_id}: {full_uri} | Reasons: {block_reasons}")
            
            # Build detailed block response
            block_detail = {
                "status": "blocked",
                "request_id": request_id,
                "timestamp": get_timestamp_iso(),
                "reasons": block_reasons,
                "scores": {
                    "bert": round(bert_score, 4),
                    "rule": round(rule_score, 4),
                    "anomaly": round(anomaly_score, 4),
                    "combined": round(combined_score, 4)
                },
                "detected_patterns": detected_types[:5] if detected_types else []
            }
            
            return JSONResponse(
                status_code=403,
                content=block_detail,
                headers={"X-WAF-Block-ID": request_id}
            )
        
        # Allow - forward to backend
        logger.info(f"✓ ALLOWED {request_id}: {full_uri} | Scores: BERT={bert_score:.3f} Rule={rule_score:.3f}")
        
        # Forward request to backend
        status, response_headers, response_body = await reverse_proxy.forward_request(
            method=method,
            path=f"/{path}",
            query=query_string,
            headers=headers,
            body=body_bytes if body_bytes else None
        )
        
        # Return backend response
        return Response(
            content=response_body,
            status_code=status,
            headers=response_headers
        )
    
    except Exception as e:
        logger.error(f"Error in proxy pipeline: {e}")
        processing_time_ms = (time.time() - start_time) * 1000
        database.log_request(
            request_id=request_id,
            ip=request.client[0] if request.client else "0.0.0.0",
            method=request.method,
            endpoint=f"/{path}",
            full_uri=f"/{path}",
            headers={},
            body_preview="",
            bert_score=0.0,
            rule_score=0.0,
            anomaly_score=0.0,
            combined_score=0.0,
            prediction="unknown",
            action="error",
            blocked=False,
            block_reason=str(e),
            processing_time_ms=processing_time_ms
        )
        
        return Response(content="WAF Processing Error", status_code=500)

# ===== ADMIN API ENDPOINTS =====

@app.get("/api/logs")
async def get_logs(limit: int = 100):
    """Get recent request logs"""
    logs = database.get_recent_logs(limit)
    return {"total": len(logs), "logs": logs}

@app.get("/api/stats")
async def get_stats():
    """Get aggregated statistics"""
    stats = database.get_stats()
    rl_stats = rl_engine.get_policy_stats() if WAFConfig.RL_ENABLED else {}
    anomaly_stats = anomaly_engine.get_stats() if WAFConfig.ANOMALY_ENABLED else {}
    
    return {
        "requests": stats,
        "rl_policy": rl_stats,
        "anomaly_detection": anomaly_stats,
        "configuration": {
            "ai_threshold": WAFConfig.AI_CONFIDENCE_THRESHOLD,
            "rule_threshold": WAFConfig.RULE_ENGINE_THRESHOLD,
            "anomaly_threshold": WAFConfig.ANOMALY_THRESHOLD
        }
    }

@app.post("/api/feedback")
async def submit_feedback(request: Request):
    """
    Submit admin feedback for a request
    Body: {
        "request_id": "uuid",
        "decision": "benign" or "malicious",
        "confidence": 0.0-1.0,
        "notes": "optional notes"
    }
    """
    try:
        data = await request.json()
        request_id = data.get("request_id")
        decision = data.get("decision")  # benign or malicious
        confidence = float(data.get("confidence", 1.0))
        notes = data.get("notes", "")
        
        if decision not in ["benign", "malicious"]:
            return JSONResponse({"error": "Invalid decision"}, status_code=400)
        
        database.save_feedback(request_id, decision, confidence, notes)
        
        # Update RL policy based on feedback
        if WAFConfig.RL_ENABLED:
            # Simplified: extract request details from logs
            logs = database.get_recent_logs(1000)
            req_log = next((log for log in logs if log['request_id'] == request_id), None)
            
            if req_log:
                action = "block" if req_log['blocked'] else "allow"
                rl_engine.apply_feedback(
                    request_id, decision, action,
                    req_log['bert_score'], req_log['rule_score'],
                    req_log['method'], req_log['endpoint']
                )
        
        return {"success": True, "message": f"Feedback recorded for {request_id}"}
    
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/api/qtable")
async def get_q_table(limit: int = 50):
    """Get RL Q-table (for debugging)"""
    if not WAFConfig.RL_ENABLED:
        return {"error": "RL not enabled"}
    
    q_entries = database.get_q_table(limit)
    return {
        "entries": len(q_entries),
        "table": q_entries,
        "policy_stats": rl_engine.get_policy_stats()
    }

@app.post("/api/config")
async def update_config(request: Request):
    """
    Update configuration threshold (at runtime)
    Body: {
        "threshold_name": "AI_CONFIDENCE_THRESHOLD",
        "value": 0.90
    }
    """
    try:
        data = await request.json()
        threshold = data.get("threshold_name")
        value = float(data.get("value"))
        
        if not hasattr(WAFConfig, threshold):
            return JSONResponse({"error": f"Unknown threshold: {threshold}"}, status_code=400)
        
        # Update in config (note: this is runtime only, doesn't persist)
        setattr(WAFConfig, threshold, value)
        logger.info(f"Config updated: {threshold} = {value}")
        
        return {"success": True, "message": f"{threshold} set to {value}"}
    
    except Exception as e:
        logger.error(f"Config update error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return WAFConfig.display()

# ===== DASHBOARD UI =====

@app.get("/dashboard")
async def dashboard():
    """Return simple dashboard HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SecureBERT WAF Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .header { color: #333; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }
            .card { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .stat { display: inline-block; margin: 10px;  }
            .stat-value { font-size: 24px; font-weight: bold; color: #0066cc; }
            .stat-label { color: #666; }
            button { padding: 10px 15px; background: #0066cc; color: white; border: none; border-radius: 3px; cursor: pointer; }
            button:hover { background: #0052a3; }
            #logs { max-height: 400px; overflow-y: auto; }
            .row { border-bottom: 1px solid #eee; padding: 8px; }
            .blocked { color: #d32f2f; font-weight: bold; }
            .allowed { color: #388e3c; }
        </style>
        <script>
            async function loadStats() {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('total-requests').textContent = data.requests.total_requests || 0;
                document.getElementById('blocked-requests').textContent = data.requests.blocked_requests || 0;
                document.getElementById('block-rate').textContent = (data.requests.block_rate || 0).toFixed(1) + '%';
                document.getElementById('avg-bert').textContent = (data.requests.average_bert_score || 0).toFixed(3);
            }
            
            async function loadLogs() {
                const res = await fetch('/api/logs?limit=20');
                const data = await res.json();
                let html = '';
                for (const log of data.logs || []) {
                    const blockClass = log.blocked ? 'blocked' : 'allowed';
                    const action = log.blocked ? '🚫 BLOCKED' : '✓ ALLOWED';
                    html += '<div class="row"><strong>' + action + '</strong> ' + log.method + ' ' + log.endpoint + ' | ' +
                            'BERT:' + (log.bert_score || 0).toFixed(3) + ' Rule:' + (log.rule_score || 0).toFixed(3) + '</div>';
                }
                document.getElementById('logs').innerHTML = html || '<p>No logs</p>';
            }
            
            setInterval(() => {
                loadStats();
                loadLogs();
            }, 5000);
            
            window.addEventListener('load', () => {
                loadStats();
                loadLogs();
            });
        </script>
    </head>
    <body>
        <div class="header">
            <h1>🛡️ SecureBERT WAF Dashboard</h1>
            <p>Real-time attack detection and blocking</p>
        </div>
        
        <div class="card">
            <h2>Statistics</h2>
            <div class="stat">
                <div class="stat-value" id="total-requests">0</div>
                <div class="stat-label">Total Requests</div>
            </div>
            <div class="stat">
                <div class="stat-value blocked" id="blocked-requests">0</div>
                <div class="stat-label">Blocked</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="block-rate">0%</div>
                <div class="stat-label">Block Rate</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="avg-bert">0.000</div>
                <div class="stat-label">Avg BERT Score</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Recent Requests</h2>
            <div id="logs">Loading...</div>
        </div>
        
        <div class="card">
            <h2>Actions</h2>
            <button onclick="loadStats()">Refresh Stats</button>
            <button onclick="location.href='/api/logs?limit=1000'">Export Logs (JSON)</button>
            <button onclick="location.href='/api/config'">View Config</button>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    logger.info("Starting WAF service...")
