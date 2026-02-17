"""
Utility functions for WAF
"""

import hashlib
import json
import re
from typing import Dict, List, Tuple, Optional, Any
import time
from datetime import datetime
import uuid

def generate_request_id() -> str:
    """Generate unique request ID"""
    return str(uuid.uuid4())

def extract_state_hash(method: str, endpoint: str, bert_score: float, rule_score: float) -> str:
    """Create hash of request state for RL Q-learning"""
    # Bucket scores into discrete ranges for Q-learning state space
    bert_bucket = int(bert_score * 10) / 10  # 0.0, 0.1, 0.2, ... 1.0
    rule_bucket = int(rule_score * 10) / 10
    
    # Simplify endpoint to category
    endpoint_category = categorize_endpoint(endpoint)
    
    state_str = f"{method}:{endpoint_category}:{bert_bucket:.1f}:{rule_bucket:.1f}"
    return hashlib.sha256(state_str.encode()).hexdigest()[:16]

def categorize_endpoint(endpoint: str) -> str:
    """Categorize endpoint for RL state space"""
    if not endpoint:
        return "root"
    
    # Remove query parameters
    path = endpoint.split('?')[0]
    
    # Extract first path segment
    parts = [p for p in path.split('/') if p]
    if not parts:
        return "root"
    
    return parts[0]  # e.g., "api", "admin", "user", etc.

def extract_request_details(method: str, uri: str, headers: Dict, body: str = "") -> Dict[str, Any]:
    """Extract key details from request for analysis"""
    details = {
        'method': method,
        'uri': uri,
        'path': uri.split('?')[0],
        'query_string': uri.split('?')[1] if '?' in uri else '',
        'has_body': len(body) > 0,
        'body_length': len(body),
        'headers_count': len(headers),
        'special_char_count': sum(1 for c in uri if c in '!@#$%^&*()[]{};<>|\\"\'-,./?='),
        'encoded_chars_count': len(re.findall(r'%[0-9A-Fa-f]{2}', uri)),
        'sql_keywords': count_keywords(uri, ['union', 'select', 'drop', 'insert', 'delete', 'update', 'exec']),
        'xss_keywords': count_keywords(uri, ['script', 'alert', 'onclick', 'onerror', 'onload', 'eval']),
        'path_traversal_patterns': count_keywords(uri, ['../', '..\\', '..']),
        'has_cookie': 'Cookie' in headers,
        'has_user_agent': 'User-Agent' in headers,
    }
    return details

def count_keywords(text: str, keywords: List[str]) -> int:
    """Count occurrences of keywords in text"""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)

def build_model_input_text(method: str, uri: str, headers: Dict, body: str = "") -> str:
    """Build text representation for BERT input"""
    # Create normalized text from request
    text_parts = [
        method,
        uri.split('?')[0],  # Just the path
        uri.split('?')[1] if '?' in uri else '',  # Query params
        headers.get('User-Agent', 'unknown')[:50],
    ]
    
    if body:
        text_parts.append(body[:200])  # First 200 chars of body
    
    return " ".join(text_parts)

def normalize_uri(uri: str) -> str:
    """Normalize URI for comparison"""
    # Remove numbers to generalize
    normalized = re.sub(r'\d+', '{ID}', uri)
    # Normalize common patterns
    normalized = re.sub(r'[a-f0-9]{32}', '{HASH}', normalized)  # MD5
    normalized = re.sub(r'[a-f0-9]{40}', '{HASH}', normalized)  # SHA1
    return normalized

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def average_embeddings(embeddings: List[List[float]]) -> Optional[List[float]]:
    """Calculate average embedding from list of embeddings"""
    if not embeddings:
        return None
    
    dim = len(embeddings[0])
    avg = [sum(e[i] for e in embeddings) / len(embeddings) for i in range(dim)]
    return avg

def get_timestamp_iso() -> str:
    """Get current timestamp in ISO format"""
    return datetime.utcnow().isoformat()

class RateLimiter:
    """Simple rate limiter for IP addresses"""
    def __init__(self, max_requests: int = 100, window_sec: int = 60):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.requests = {}  # ip -> [(timestamp, count), ...]
    
    def is_allowed(self, ip: str) -> bool:
        """Check if IP is allowed to make request"""
        now = time.time()
        cutoff = now - self.window_sec
        
        if ip not in self.requests:
            self.requests[ip] = [(now, 1)]
            return True
        
        # Clean old requests
        self.requests[ip] = [(ts, count) for ts, count in self.requests[ip] if ts > cutoff]
        
        # Count recent requests
        recent_count = sum(count for ts, count in self.requests[ip])
        
        if recent_count >= self.max_requests:
            return False
        
        self.requests[ip].append((now, 1))
        return True

# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=1000, window_sec=60)
