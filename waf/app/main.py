from fastapi import FastAPI, Request, HTTPException, Response
import logging
import torch
import os
import sys

# Ensure we can import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import WAFTransformer
from model.tokenizer import HttpTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("waf-service")

app = FastAPI()

# Global Model & Tokenizer
model = None
tokenizer = None
device = torch.device("cpu") # Inference on CPU is fine for this scale

MAX_LEN = 64

def load_artifacts():
    global model, tokenizer
    try:
        # Paths are relative to WORKDIR /app in Docker
        TOKENIZER_PATH = "model/weights/tokenizer.json"
        MODEL_PATH = "model/weights/waf_model.pth"
        
        logger.info(f"Loading tokenizer from {TOKENIZER_PATH}...")
        tokenizer = HttpTokenizer()
        tokenizer.load(TOKENIZER_PATH)
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = WAFTransformer(vocab_size=1000, d_model=64, num_layers=2, num_classes=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        logger.info("Artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        # We don't crash, we just fail open (or closed) depending on policy
        # For this demo, we'll initialize dummy if failed, to avoid crash loop
        tokenizer = None
        model = None

@app.on_event("startup")
async def startup_event():
    load_artifacts()

def preprocess_request(method, uri, body=""):
    # Simple normalization similar to training
    # Ideally reuse RequestNormalizer but need to adapt for realtime single string
    # We'll do a simple heuristic here matching the synthetic generation
    
    # Heuristic: Replace numbers with {ID}
    import re
    uri_norm = re.sub(r'\d+', '{ID}', uri) 
    
    # Construct the sequence
    text = f"{method} {uri_norm}" 
    # Ignore body for this lightweight model demo, as synthetic data was URI focused
    
    return text

@app.get("/")
def health_check():
    return {"status": "running", "model_loaded": model is not None}

@app.api_route("/analyze", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def analyze_request(request: Request):
    """
    Analyzes the incoming request and decides whether to block or allow.
    Nginx sends the original request details via headers.
    """
    global model, tokenizer
    
    # Extract metadata from Nginx headers
    original_uri = request.headers.get("X-Original-URI", "/")
    original_method = request.headers.get("X-Original-Method", "GET")
    original_ip = request.headers.get("X-Original-IP", "0.0.0.0")
    
    # Fail open if model not ready
    if model is None or tokenizer is None:
        return Response(status_code=200)

    try:
        # Preprocess
        text = preprocess_request(original_method, original_uri)
        
        # Tokenize
        encoded = tokenizer.encode(text)
        ids = encoded.ids
        
        # Pad/Truncate
        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN]
        else:
            ids = ids + [0] * (MAX_LEN - len(ids))
            
        input_tensor = torch.tensor([ids], dtype=torch.long).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            malicious_prob = probs[0][1].item()
            
        logger.info(f"IP={original_ip} URI={original_uri} Pred={pred_class} MaliciousProb={malicious_prob:.4f}")
        
        if pred_class == 1: # Malicious
            logger.warning(f"BLOCKING Malicious Request: {original_uri}")
            return Response(content="Blocked by WAF", status_code=403)
            
        return Response(status_code=200)
        
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        # Fail Open
        return Response(status_code=200)
