"""
Configuration management for WAF
Reads from environment and CONFIG.env
"""

import os
from typing import Optional

class WAFConfig:
    """Central configuration for WAF"""
    
    # Core URLs
    TARGET_WEBSITE_URL = os.getenv("TARGET_WEBSITE_URL", "http://target-app:3000")
    WAF_LISTEN_HOST = os.getenv("WAF_LISTEN_HOST", "0.0.0.0")
    WAF_LISTEN_PORT = int(os.getenv("WAF_LISTEN_PORT", "8000"))
    
    # ML Model
    MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/weights")
    MODEL_NAME = os.getenv("MODEL_NAME", "SecureBERT")
    
    # Thresholds
    AI_CONFIDENCE_THRESHOLD = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.95"))
    UNCERTAINTY_THRESHOLD = float(os.getenv("UNCERTAINTY_THRESHOLD", "0.85"))
    RULE_ENGINE_THRESHOLD = float(os.getenv("RULE_ENGINE_THRESHOLD", "0.70"))
    ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.75"))
    COMBINED_THRESHOLD = float(os.getenv("COMBINED_THRESHOLD", "0.85"))
    
    # RL Configuration
    RL_ENABLED = os.getenv("RL_ENABLED", "true").lower() == "true"
    RL_EPSILON = float(os.getenv("RL_EPSILON", "0.1"))  # Exploration rate
    RL_ALPHA = float(os.getenv("RL_ALPHA", "0.1"))      # Learning rate
    RL_GAMMA = float(os.getenv("RL_GAMMA", "0.9"))      # Discount factor
    
    # Anomaly Detection
    ANOMALY_ENABLED = os.getenv("ANOMALY_ENABLED", "true").lower() == "true"
    ANOMALY_EMBEDDING_CACHE_SIZE = int(os.getenv("ANOMALY_EMBEDDING_CACHE_SIZE", "1000"))
    ANOMALY_SIMILARITY_THRESHOLD = float(os.getenv("ANOMALY_SIMILARITY_THRESHOLD", "0.7"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "/app/logs/waf.log")
    
    # Admin API
    ADMIN_API_ENABLED = os.getenv("ADMIN_API_ENABLED", "true").lower() == "true"
    ADMIN_API_PORT = int(os.getenv("ADMIN_API_PORT", "8001"))
    ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "change-me-in-production")
    
    # Request handling
    REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "30"))
    MAX_BODY_SIZE_BYTES = int(os.getenv("MAX_BODY_SIZE_BYTES", "10485760"))  # 10MB
    
    # Feature toggles
    BLOCK_ON_UNCERTAIN = os.getenv("BLOCK_ON_UNCERTAIN", "false").lower() == "true"
    BLOCK_ON_ANOMALY = os.getenv("BLOCK_ON_ANOMALY", "true").lower() == "true"
    LOG_ALLOWED_REQUESTS = os.getenv("LOG_ALLOWED_REQUESTS", "true").lower() == "true"
    
    # Alerts
    ALERT_EMAIL_ENABLED = os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true"
    ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "waf@example.com")
    ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "admin@example.com")
    ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.95"))
    
    @classmethod
    def display(cls):
        """Display current configuration"""
        config_dict = {k: getattr(cls, k) for k in dir(cls) if not k.startswith('_') and k.isupper()}
        return config_dict
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration"""
        if not cls.TARGET_WEBSITE_URL:
            print("ERROR: TARGET_WEBSITE_URL not set")
            return False
        
        if cls.AI_CONFIDENCE_THRESHOLD < 0 or cls.AI_CONFIDENCE_THRESHOLD > 1:
            print("ERROR: AI_CONFIDENCE_THRESHOLD must be between 0 and 1")
            return False
        
        return True
