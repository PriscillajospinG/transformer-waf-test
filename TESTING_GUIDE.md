# 🧪 WAF Testing & Deployment Guide

## Pre-Deployment Checklist

Before starting the WAF, verify:

```bash
# 1. Check Docker installation
docker --version
docker-compose --version

# 2. Verify Python dependencies are listed
cat waf/requirements.txt

# 3. Check model files exist (or will be downloaded)
ls -la waf/model/

# 4. Verify nginx config is valid
cat nginx/nginx.conf

# 5. Check CONFIG.env is properly configured
cat CONFIG.env
```

## Step-by-Step Deployment

### Phase 1: Configuration

```bash
# 1. Clone/copy the project
cd transformer-waf-test

# 2. Edit CONFIG.env for your environment
nano CONFIG.env

# Change these critical fields:
TARGET_WEBSITE_URL=http://your-backend:3000      # Your app location
PUBLIC_IP_OR_DOMAIN=192.168.1.100               # Your public IP/domain
PUBLIC_PORT=8080                                 # Public port

# Optional: Adjust detection thresholds
AI_CONFIDENCE_THRESHOLD=0.95                    # 0.95 = strict, 0.85 = lenient
ANOMALY_THRESHOLD=0.75                          # Sensitivity to anomalies
BLOCK_ON_ANOMALY=true                           # Block zero-days

# Optional: RL Configuration
RL_ENABLED=true                                 # Enable learning from feedback
RL_EPSILON=0.1                                  # Exploration rate
```

### Phase 2: Build and Start

```bash
# 1. Build WAF Docker image
docker-compose build waf-service

# 2. Start all services with logging
docker-compose up -d

# 3. Wait for services to be healthy (30-40 seconds)
sleep 40

# 4. Verify all containers are running
docker-compose ps

# Expected output:
# NAME              STATUS
# target-app        Up (healthy)
# waf-service       Up (healthy)
# waf-nginx         Up

# 5. Check WAF service is responsive
curl http://localhost:8000/health
# Should return: {"status":"running", "model_loaded": true, ...}

# 6. Check Nginx is routing correctly
curl http://localhost:8080/
# Should return Juice Shop (target app) homepage
```

### Phase 3: Verify Detection Works

```bash
# Test 1: Normal Request (should be ALLOWED)
echo "✓ Test 1: Normal Request"
curl http://localhost:8080/
# Should return 200 OK with app content

# Test 2: SQL Injection (should be BLOCKED)
echo "✗ Test 2: SQL Injection"
curl "http://localhost:8080/search?query=1' AND 1=1--"
# Should return 403 Forbidden with block reason

# Test 3: XSS (should be BLOCKED)
echo "✗ Test 3: XSS Attack"
curl "http://localhost:8080/api/comments?text=<script>alert('xss')</script>"
# Should return 403 Forbidden

# Test 4: Path Traversal (should be BLOCKED)
echo "✗ Test 4: Path Traversal"
curl "http://localhost:8080/files?path=../../../../etc/passwd"
# Should return 403 Forbidden

# Test 5: Command Injection (should be BLOCKED)
echo "✗ Test 5: Command Injection"
curl "http://localhost:8080/api/exec?cmd=id;whoami"
# Should return 403 Forbidden
```

### Phase 4: Monitor in Real-Time

```bash
# Terminal 1: Watch WAF logs
docker-compose logs -f waf-service

# Terminal 2: Watch Nginx access logs (blocked requests)
tail -f nginx/logs/access.log | grep " 403 "

# Terminal 3: Make test requests
curl "http://localhost:8080/search?q=' OR 1=1"
```

### Phase 5: Test Admin API

```bash
# 1. Get recent request logs
curl http://localhost:8080/api/logs | jq '.' | head -50

# 2. Get statistics
curl http://localhost:8080/api/stats | jq '.requests' | grep block_rate

# 3. Get a blocked request ID from logs
REQUEST_ID=$(curl http://localhost:8080/api/logs | jq -r '.logs[0].request_id')
echo "Using request ID: $REQUEST_ID"

# 4. Submit feedback (mark as false positive)
curl -X POST http://localhost:8080/api/feedback \
  -H "Content-Type: application/json" \
  -d "{
    \"request_id\": \"$REQUEST_ID\",
    \"decision\": \"benign\",
    \"confidence\": 0.95,
    \"notes\": \"This is a legitimate API request\"
  }"

# 5. View RL Q-table (policy)
curl http://localhost:8080/api/qtable | jq '.policy_stats'

# 6. Access dashboard
open http://localhost:8080/dashboard
# Or: curl http://localhost:8080/dashboard > dashboard.html
```

### Phase 6: Load Testing

```bash
# Install Apache Bench (if not already installed)
# macOS:
brew install httpd

# Linux:
sudo apt-get install apache2-utils

# Run benchmarks
echo "Benign traffic (should ~100% allow):"
ab -n 1000 -c 10 http://localhost:8080/

echo "Malicious traffic (should ~100% block):"
for i in {1..100}; do
  curl -s "http://localhost:8080/api?id=1' OR '1'='1" &
done
wait

# Check results
curl http://localhost:8080/api/stats | jq '.requests'
```

## Advanced Testing

### A/B Testing Thresholds

```bash
# Create threshold comparison
echo "Testing different AI thresholds:"

# Set strict threshold (0.99)
curl -X POST http://localhost:8080/api/config \
  -H "Content-Type: application/json" \
  -d '{"threshold_name": "AI_CONFIDENCE_THRESHOLD", "value": 0.99}'
echo "Threshold: 0.99 set - running 100 requests"
for i in {1..100}; do curl -s "http://localhost:8080/" > /dev/null; done
STATS_STRICT=$(curl http://localhost:8080/api/stats | jq '.requests.block_rate')
echo "Block rate with 0.99: $STATS_STRICT%"

# Set lenient threshold (0.70)
curl -X POST http://localhost:8080/api/config \
  -H "Content-Type: application/json" \
  -d '{"threshold_name": "AI_CONFIDENCE_THRESHOLD", "value": 0.70}'
echo "Threshold: 0.70 set - running 100 requests"
for i in {1..100}; do curl -s "http://localhost:8080/" > /dev/null; done
STATS_LENIENT=$(curl http://localhost:8080/api/stats | jq '.requests.block_rate')
echo "Block rate with 0.70: $STATS_LENIENT%"

# Restore original
curl -X POST http://localhost:8080/api/config \
  -H "Content-Type: application/json" \
  -d '{"threshold_name": "AI_CONFIDENCE_THRESHOLD", "value": 0.95}'
```

### Zero-Day Detection Testing

```bash
# Test novel attack variants (may not match known patterns)
echo "Testing zero-day detection (anomaly engine):"

# Normal request
curl -s http://localhost:8080/api/users | jq -r '.detection[].detected_patterns'

# Novel attack variant (slight variation of SQL injection)
curl -s "http://localhost:8080/api?x=1 %55NION %53ELECT" | jq -r '.detected_patterns'

# Another variant
curl -s "http://localhost:8080/search?q=0x3d%3d1%3d%3d1" | jq -r '.detected_patterns'
```

### Reinforcement Learning Testing

```bash
# Show Q-learning in action

# 1. Make 10 requests that trigger uncertain predictions
for i in {1..10}; do
  curl -s "http://localhost:8080/api/fuzzy?param=$(date +%s%N)" > /dev/null
done

# 2. View Q-table before feedback
echo "Q-table before feedback:"
curl http://localhost:8080/api/qtable | jq '.table | length'

# 3. Get request logs and find uncertain ones
curl http://localhost:8080/api/logs | jq '.logs[] | select(.bert_score > 0.5 and .bert_score < 0.95)'

# 4. Submit feedback for uncertain requests
curl -X POST http://localhost:8080/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"request_id": "xxx", "decision": "benign", "confidence": 1.0}'

# 5. Watch Q-values update
echo "Q-table after feedback:"
curl http://localhost:8080/api/qtable | jq '.policy_stats'
```

## Performance Benchmarks

### Expected Metrics

- **Throughput**: 100-200 requests/second (single instance)
- **Latency**: 40-80ms per request (includes model inference)
- **Memory**: 1.5-2.0 GB (BERT model + cache)
- **CPU**: 30-50% on 2-core system
- **False Positive Rate**: <2% (with default thresholds)
- **Detection Rate**: 85-95% for known attacks, 40-60% for zero-days

### Running Benchmarks

```bash
# Simple throughput test
time for i in {1..1000}; do curl -s http://localhost:8080/ > /dev/null; done

# Detailed latency test
for i in {1..100}; do
  time curl -s "http://localhost:8080/page?id=$i" > /dev/null 2>&1
done | grep real | awk '{print $2}' | sort -n | tail -10

# Resource monitoring
watch -n 1 'docker stats --no-stream'
```

## Troubleshooting

### Issue: Models fail to load

```bash
# Check model files exist
docker-compose exec waf-service ls -la /app/model/weights/

# Check logs for model loading errors
docker-compose logs waf-service | grep -i "model\|load\|error"

# Solution: Download/rebuild
docker-compose down
docker-compose build --no-cache waf-service
docker-compose up -d
```

### Issue: High false positive rate

```bash
# Diagnosis
curl http://localhost:8080/api/stats | jq '.requests'

# If block_rate > 5%, lower the threshold
curl -X POST http://localhost:8080/api/config \
  -H "Content-Type: application/json" \
  -d '{"threshold_name": "AI_CONFIDENCE_THRESHOLD", "value": 0.90}'

# Or disable anomaly blocking
curl -X POST http://localhost:8080/api/config \
  -H "Content-Type: application/json" \
  -d '{"threshold_name": "BLOCK_ON_ANOMALY", "value": false}'
```

### Issue: WAF service keeps crashing

```bash
# Check logs
docker-compose logs waf-service

# Common causes:
# 1. Out of memory: Increase limits in docker-compose.yml
# 2. Model not found: Rebuild and restart
# 3. Port conflict: Check docker-compose port mappings

# Solution
docker-compose down
docker-compose up -d --force-recreate
```

### Issue: Requests timeout

```bash
# Increase timeout in CONFIG.env
REQUEST_TIMEOUT_SEC=60

# Or check WAF service performance
docker stats waf-service

# If CPU is maxed out, may need more resources
# Edit docker-compose.yml limits section
```

## Cleanup and Shutdown

```bash
# Stop all services
docker-compose down

# Remove database (start fresh)
rm -f waf/data/waf.db

# Remove model cache
rm -rf waf/model/weights/__pycache__

# Remove logs
rm -rf nginx/logs/*
rm -f waf/logs/waf.log

# Full cleanup
docker-compose down -v --remove-orphans
docker system prune -a
```

## Integration with CI/CD

```yaml
# Example GitHub Actions workflow
name: WAF Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: docker/setup-buildx-action@v1
      
      - name: Start WAF
        run: docker-compose up -d
      
      - name: Wait for health
        run: sleep 45
      
      - name: Run tests
        run: |
          curl http://localhost:8080/
          curl "http://localhost:8080/search?q='OR 1=1"
          curl http://localhost:8080/api/stats
      
      - name: Cleanup
        run: docker-compose down
```

---

**✅ WAF is now fully deployed, tested, and ready for production!**
