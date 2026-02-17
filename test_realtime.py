#!/usr/bin/env python3
"""
Real-time WAF Testing
Monitor detection in action as requests are made
"""

import requests
import subprocess
import time
from threading import Thread
import sys

BASE_URL = "http://localhost:8080"

# Test cases: (description, path, expected_status)
TESTS = [
    ("✅ Home Page", "/", 200),
    ("✅ Benign Search", "/rest/products/search?q=apple", 200),
    ("❌ SQL Injection", "/rest/products/search?q=' OR 1=1 --", 403),
    ("❌ Zero-Day AND Variant", "/rest/products/search?q=1' AND '1'='1", 403),
    ("❌ XSS Attack", "/rest/products/search?q=<script>alert(1)</script>", 403),
    ("❌ Zero-Day XSS IMG", "/rest/products/search?q=<img src=x onerror=alert(1)>", 403),
    ("❌ Command Injection", "/?cmd=find / -name '*.sql'", 403),
    ("❌ Encoding Attack", "/files%2fetc%2fpasswd", 403),
]

def get_waf_logs():
    """Get recent WAF logs"""
    try:
        result = subprocess.run(
            ["docker-compose", "logs", "waf-service", "--tail=5"],
            capture_output=True,
            text=True,
            cwd="/Users/priscillajosping/Desktop/Mini Project/transformer-waf-test"
        )
        return result.stdout
    except:
        return ""

def test_request(desc, path, expected):
    """Test a single request"""
    print(f"\n{'='*70}")
    print(f"{desc}")
    print(f"   Path: {path}")
    print(f"   Expected: {expected}")
    
    try:
        response = requests.head(f"{BASE_URL}{path}", timeout=5, allow_redirects=False)
        status = response.status_code
        result = "✓" if status == expected else "✗"
        print(f"   Result: {status} {result}")
        
        # Show logs
        logs = get_waf_logs()
        if "BLOCKING" in logs or "ALLOWING" in logs:
            print(f"\n   WAF Decision:")
            for line in logs.split('\n'):
                if 'BLOCKING' in line or 'ALLOWING' in line or 'prob=' in line:
                    print(f"   {line[-150:]}")
        
        return status == expected
    except Exception as e:
        print(f"   Error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("🔍 REAL-TIME WAF DETECTION TEST")
    print("="*70)
    print(f"WAF Endpoint: {BASE_URL}")
    print(f"Detection Thresholds: AI=0.75, Uncertainty=0.65")
    print("="*70)
    
    passed = 0
    total = len(TESTS)
    
    for desc, path, expected in TESTS:
        if test_request(desc, path, expected):
            passed += 1
        time.sleep(1)  # Delay between tests
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 RESULTS: {passed}/{total} PASSED ({(passed/total)*100:.0f}%)")
    print("="*70)

if __name__ == "__main__":
    main()
