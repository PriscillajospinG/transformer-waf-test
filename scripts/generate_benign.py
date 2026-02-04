import requests
import time
import random

BASE_URL = "http://localhost:8080"

PATHS = [
    "/",
    "/#/about",
    "/#/contact",
    "/#/login",
    "/#/register",
    "/api/Users",
    "/api/Products",
    "/rest/products/search?q=apple",
    "/rest/products/search?q=orange",
    "/api/Feedbacks",
    "/assets/public/images/uploads/meow.jpg",
    "/socket.io/?EIO=3&transport=polling&t=N8sl79O"
]

def generate_benign():
    print(f"Starting Benign Traffic Generation against {BASE_URL}...")
    session = requests.Session()
    
    for _ in range(100): # Generate 100 requests for verified test
        path = random.choice(PATHS)
        try:
            url = f"{BASE_URL}{path}"
            # Simulate real user agent
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
            }
            res = session.get(url, headers=headers)
            print(f"GET {path} - {res.status_code}")
            time.sleep(random.uniform(0.1, 0.5))
        except Exception as e:
            print(f"Error accessing {path}: {e}")

if __name__ == "__main__":
    generate_benign()
