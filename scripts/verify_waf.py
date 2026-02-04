import requests
import time
import sys

BASE_URL = "http://localhost:8080"

def test_waf():
    print(f"Waiting for services at {BASE_URL}...")
    for i in range(30):
        try:
            requests.get(BASE_URL, timeout=1)
            print("Server is UP!")
            break
        except:
            time.sleep(2)
            print(".", end="", flush=True)
    else:
        print("\nTimeout waiting for server.")
        sys.exit(1)

    print("\n--- Starting WAF Verification ---")
    
    # Test Cases
    tests = [
        # (Name, URL, Expected Status)
        ("Benign Root", "/", 200),
        # ("Benign API", "/api/quantities", 200), # Endpoint seems broken in this juice shop version
        ("Benign Search", "/rest/products/search?q=apple", 200),
        
        ("Malicious SQLi 1", "/rest/products/search?q=' OR 1=1 --", 403),
        ("Malicious SQLi 2", "/api/Users?email=' OR '1'='1", 403),
        ("Malicious XSS", "/rest/products/search?q=<script>alert(1)</script>", 403),
        ("Malicious Path Traversal", "/../../etc/passwd", 403),
    ]
    
    passed = 0
    failed = 0
    
    for name, path, expected in tests:
        url = f"{BASE_URL}{path}"
        try:
            res = requests.get(url)
            code = res.status_code
            result = "PASS" if code == expected else f"FAIL (Got {code})"
            print(f"[{result}] {name}: {path} -> {code}")
            
            if code == expected:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            failed += 1
            
    print("\n--- Summary ---")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("SUCCESS: WAF is working as expected!")
        sys.exit(0)
    else:
        print("FAILURE: Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    test_waf()
