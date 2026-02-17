#!/usr/bin/env python3
"""
Zero-Day Attack Detection Test Suite
Tests the enhanced WAF against known and unknown attack patterns
"""

import requests
import time
import sys

BASE_URL = "http://localhost:8080"

def test_waf_zero_day():
    """
    Comprehensive test of zero-day protection capabilities
    """
    print("="*80)
    print("ZERO-DAY ATTACK DETECTION TEST SUITE")
    print("="*80)
    
    # Wait for server
    print(f"\n⏳ Waiting for services at {BASE_URL}...")
    for i in range(30):
        try:
            requests.get(BASE_URL, timeout=1)
            print("✅ Server is UP!\n")
            break
        except:
            time.sleep(2)
            print(".", end="", flush=True)
    else:
        print("\n❌ Timeout waiting for server.")
        sys.exit(1)

    # Test categories
    tests = {
        "BENIGN REQUESTS": [
            ("Home Page", "/", 200),
            ("Search Apple", "/rest/products/search?q=apple", 200),
            ("API Users", "/api/Users", 200),
            ("Contact Page", "/#/contact", 200),
        ],
        
        "KNOWN ATTACKS (Training Data)": [
            ("SQLi - OR 1=1", "/rest/products/search?q=' OR 1=1 --", 403),
            ("XSS - Script Tag", "/rest/products/search?q=<script>alert(1)</script>", 403),
            ("Path Traversal", "/../../../etc/passwd", 403),
        ],
        
        "ZERO-DAY VARIANTS (Never Seen)": [
            ("SQLi - AND Variant", "/rest/products/search?q=1' AND '1'='1", 403),
            ("SQLi - UNION Variant", "/rest/products/search?q=' UNION SELECT 1,2,3--", 403),
            ("XSS - IMG Tag", "/rest/products/search?q=<img src=x onerror=alert(1)>", 403),
            ("XSS - SVG Tag", "/rest/products/search?q=<svg onload=alert(1)>", 403),
            ("Path Traversal - Encoded", "/etc/passwd%00.jpg", 403),
            ("Path Traversal - Encoded Slashes", "/..%2f..%2fetc%2fpasswd", 403),
            ("Command Injection - Find", "/?cmd=find / -name '*.sql'", 403),
            ("SQLi - Obfuscated UNION", "/search?q=' UNION/**/SELECT 1--", 403),
        ],
        
        "ENCODING ATTACK DETECTION": [
            ("URL Encoding - %00", "/?q=%00", 403),
            ("URL Encoding - %2F (slash)", "/files%2fetc%2fpasswd", 403),
            ("Double Encoding", "/%252e%252e/etc/passwd", 403),
        ],
        
        "ANOMALY DETECTION": [
            ("Excessive Special Chars", "/search?param=!@#$%^&*()[]{};<>|\\\"'--", 403),
            ("SQL Keywords + Chars", "/api?q=SELECT * FROM users WHERE id=1' AND 1=1;--", 403),
        ]
    }

    total_passed = 0
    total_failed = 0
    
    for category, test_cases in tests.items():
        print(f"\n🧪 {category}")
        print("-" * 80)
        
        category_passed = 0
        category_failed = 0
        
        for name, path, expected_status in test_cases:
            url = f"{BASE_URL}{path}"
            try:
                response = requests.get(url, timeout=5)
                actual_status = response.status_code
                
                if actual_status == expected_status:
                    status_emoji = "✅" if expected_status == 200 else "❌"
                    print(f"{status_emoji} {name}")
                    print(f"   Path: {path}")
                    print(f"   Expected: {expected_status}, Got: {actual_status} ✓")
                    category_passed += 1
                    total_passed += 1
                else:
                    print(f"⚠️ {name}")
                    print(f"   Path: {path}")
                    print(f"   Expected: {expected_status}, Got: {actual_status} ✗")
                    category_failed += 1
                    total_failed += 1
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ {name} - Connection Error: {e}")
                category_failed += 1
                total_failed += 1
            
            time.sleep(0.1)  # Small delay between requests
        
        print(f"\n📊 Category Results: {category_passed} passed, {category_failed} failed")
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📊 Success Rate: {100*total_passed/(total_passed+total_failed):.1f}%")
    
    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED! Zero-day protection is working correctly!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total_failed} test(s) failed. Check WAF configuration.")
        sys.exit(1)

if __name__ == "__main__":
    test_waf_zero_day()
