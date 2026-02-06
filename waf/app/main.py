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
device = torch.device("cpu") 

def load_artifacts():
    global model, tokenizer
    try:
        # Paths are relative to WORKDIR /app in Docker
        TOKENIZER_PATH = "waf/model/weights/tokenizer.json"
        MODEL_PATH = "waf/model/weights/waf_model.pth"
        
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
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        tokenizer = None
        model = None

@app.on_event("startup")
async def startup_event():
    load_artifacts()

def preprocess_request(method, uri, body=""):
    # Normalize
    import re
    uri_norm = re.sub(r'\d+', '{ID}', uri) 
    text = f"{method} {uri_norm}" 
    return text

@app.get("/")
def health_check():
    return {"status": "running", "model_loaded": model is not None, "type": "SecureBERT"}

@app.api_route("/analyze", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def analyze_request(request: Request):
    global model, tokenizer
    
    original_uri = request.headers.get("X-Original-URI", "/")
    original_method = request.headers.get("X-Original-Method", "GET")
    original_ip = request.headers.get("X-Original-IP", "0.0.0.0")
    
    if model is None or tokenizer is None:
        return Response(status_code=200)

    try:
        text = preprocess_request(original_method, original_uri)
        
        # Tokenize (Returns dict of tensors on CPU)
        encoding = tokenizer.encode(text)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            malicious_prob = probs[0][1].item()
            
        logger.info(f"IP={original_ip} URI={original_uri} Pred={pred_class} MaliciousProb={malicious_prob:.4f}")
        
        if pred_class == 1:
            logger.warning(f"BLOCKING Malicious Request: {original_uri}")
            return Response(content="Blocked by SecureBERT WAF", status_code=403)
            
        return Response(status_code=200)
        
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        return Response(status_code=200)
