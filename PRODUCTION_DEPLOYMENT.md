# 🚀 Production Deployment Guide

This guide explains how to run the WAF in **production mode** with continuous operation, auto-recovery, and real-time attack detection.

---

## ⚡ Quick Start - Run WAF 24/7

### Local Machine (Linux/macOS/Windows WSL)

```bash
# 1. Go to project directory
cd transformer-waf-test

# 2. Run production startup script
bash start_waf.sh

# 3. System automatically:
#    ✓ Starts all services with auto-restart enabled
#    ✓ Performs health checks every 10 seconds
#    ✓ Auto-recovers from crashes
#    ✓ Logs all activity to waf_production.log
#    ✓ Monitors for issues in background
```

**That's it!** The WAF is now **always running** and detecting attacks in real-time.

---

## 🎯 What "Always Running" Means

| Feature | How It Works | Benefit |
|---------|------------|---------|
| **Auto-Restart** | Docker restart policy = `always` | If WAF crashes, Docker automatically restarts it |
| **Health Checks** | Every 10 seconds | Detects issues before users notice them |
| **Monitoring Daemon** | Background process checks every 30s | Automatically fixes unhealthy services |
| **Resource Limits** | Memory/CPU capped | Prevents memory leaks from crashing the system |
| **Log Rotation** | Max 50MB per service | Prevents disk from filling up |
| **Graceful Recovery** | Wait 2-5s before restart | Allows clean shutdown, prevents data loss |

---

## 🛠️ Production Scripts

### 1. Start WAF - `start_waf.sh`

Starts the entire system in production mode:

```bash
bash start_waf.sh
```

**What it does:**
- ✅ Stops any existing containers
- ✅ Builds fresh Docker images
- ✅ Starts all services (Nginx, WAF API, Juice Shop)
- ✅ Waits for health checks to pass
- ✅ Verifies system is operational
- ✅ Starts background monitoring daemon
- ✅ Logs everything to `waf_production.log`

**Output:**
```
🛡️  WAF Production Startup
🏗️  Building and starting services...
⏳ Waiting for services to be healthy...
✅ All services are healthy!
════════════════════════════════════════════════════
✅ WAF PRODUCTION MODE STARTED
════════════════════════════════════════════════════

Services are running and configured for auto-restart:
  🌐 Website: http://localhost:8080
  🔍 WAF API: http://localhost:8000
```

### 2. Monitor WAF - `monitor_waf.sh`

Continuous health monitoring with auto-recovery:

```bash
bash monitor_waf.sh
```

**What it does:**
- 🔍 Checks WAF API health every 30 seconds
- 🔍 Checks Nginx health every 30 seconds
- 🔍 Checks Docker container status
- 🚨 Alerts if services go down
- 🔄 Auto-restarts failed services
- 📝 Logs all checks to `waf_production.log`
- 📧 Optional: Sends email alerts on failures

**Auto-restart Logic:**
- If a service fails, immediately restart containers
- Max 5 restart attempts before manual intervention required
- Waits 30 seconds between checks for stability

### 3. Stop WAF - `stop_waf.sh`

Gracefully stops all services:

```bash
bash stop_waf.sh
```

**What it does:**
- 🛑 Stops monitoring daemon
- 🛑 Gracefully shuts down all Docker containers
- 🛑 Cleans up network connections
- 📝 Logs shutdown time to `waf_production.log`

---

## 📊 Docker Configuration (Automatic)

The `docker-compose.yml` now includes production features:

### Auto-Restart Policy
```yaml
restart: always  # Automatically restart if crashed
```

### Health Checks
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/"]
  interval: 10s      # Check every 10 seconds
  timeout: 5s        # Wait 5 seconds for response
  retries: 3         # Allow 3 failures before marking unhealthy
  start_period: 40s  # Grace period on container start
```

### Resource Limits
```yaml
deploy:
  resources:
    limits:
      cpus: '1.0'        # Max 1 CPU core
      memory: 1024M      # Max 1GB RAM
    reservations:
      cpus: '0.5'        # Reserve 50% CPU
      memory: 512M       # Reserve 512MB RAM
```

### Log Rotation
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "50m"    # Max 50MB per log file
    max-file: "10"     # Keep max 10 files (500MB total)
```

---

## 📈 System Flow (Always Running)

```
START: bash start_waf.sh
   ↓
[docker-compose up -d --build]
   ├→ Build WAF Docker image
   ├→ Start Nginx container (auto-restart enabled)
   ├→ Start WAF service container (auto-restart enabled)
   └→ Start Juice Shop container (auto-restart enabled)
   ↓
[Health checks pass?]
   ├→ Yes → Continue
   └→ No → Retry until healthy
   ↓
[Start background monitor_waf.sh]
   ├→ Runs every 30 seconds forever
   ├→ Checks if services are responsive
   ├→ Auto-restarts if unhealthy
   └→ Logs all activity
   ↓
System is LIVE and protecting assets 24/7
   ├→ Detects incoming attacks in real-time
   ├→ Logs to: waf_production.log
   ├→ Logs to: docker stdout (docker-compose logs)
   ├→ Logs to: nginx/logs/access.log
   └→ Auto-recovers from any failure
```

---

## 🔍 Monitoring & Debugging

### View Live Production Logs

```bash
# Watch production startup/shutdown events
tail -f waf_production.log

# Example output:
# 2026-02-17 10:00:00 - Starting WAF in production mode...
# ✅ All services are healthy!
# 2026-02-17 10:00:45 - WAF is LIVE and protecting your assets!
# ✅ [10:05:30] All services healthy (Check #10)
```

### View Real-Time Attack Detection

```bash
# Live dashboard showing all 4 detection layers
python3 waf_dashboard.py

# Example output:
# Running Test: [1/6] Benign Homepage Request
# ├─ Layer 1 (Rules): PASSED
# ├─ Layer 2 (BERT): confidence=0.12 (SAFE)
# └─ Final Decision: ✅ ALLOWED (200 OK)
#
# Running Test: [3/6] SQL Injection Attack
# ├─ Layer 1 (Rules): FLAGGED (SQL keywords)
# ├─ Layer 2 (BERT): confidence=0.96 (MALICIOUS)
# └─ Final Decision: ❌ BLOCKED (403 Forbidden)
```

### Docker Container Status

```bash
# Check if containers are running and healthy
docker-compose ps

# Output:
# NAME              STATUS              PORTS
# waf-nginx         Up (healthy)        0.0.0.0:8080->80/tcp
# waf-service       Up (healthy)        (internal)
# juice-shop        Up (healthy)        (internal)
```

### Docker Service Logs

```bash
# Follow WAF service logs in real-time
docker-compose logs -f waf-service

# Example:
# waf-service | INFO: Zero-day detection enabled with threshold: 0.95
# waf-service | WARNING: BLOCKING: IP=172.18.0.1 URI=/search MaliciousProb=0.96
# waf-service | INFO: ALLOWED: IP=172.18.0.1 URI=/ BenignProb=0.98
```

### Nginx Access Logs (All Requests)

```bash
# View all HTTP requests and WAF decisions
tail -f nginx/logs/access.log

# 200 = Allowed, 403 = Blocked
# tail -f nginx/logs/access.log | grep " 403 "  # Blocked attacks only
```

---

## 🚨 Common Issues & Solutions

### Issue: Services crash and restart repeatedly

**Solution**: Check the logs
```bash
tail -f waf_production.log
docker-compose logs waf-service | tail -100
```

**Common causes:**
- Out of memory → Increase resource limits in docker-compose.yml
- Port already in use → Stop other services using ports 8080, 8000, 3000
- Missing dependencies → Run `bash setup.sh`

### Issue: WAF not detecting attacks

**Verify it's running:**
```bash
curl -I http://localhost:8080/search?q=\' OR 1=1
# Should return: 403 Forbidden (if attack pattern detected)
# Or: 200 OK (if pattern not detected)
```

**Check logs:**
```bash
docker-compose logs waf-service | grep "BLOCKING\\|ALLOWED"
```

### Issue: High disk usage from logs

**Solution**: Already configured with log rotation
- Max 50MB per log file
- Keep max 10 files
- Auto-deletes old logs

**Manual cleanup:**
```bash
# Remove old log files
rm -f nginx/logs/access.log.*
docker-compose logs --tail=0  # Clear container logs
```

### Issue: Need to update model or retrain

**Without stopping WAF:**
```bash
# Fine-tune on recent traffic
python3 waf/train/online_learning.py

# Reload model in running WAF
docker-compose restart waf-service
```

---

## 📋 Daily Operations

### Morning
```bash
# Check overnight logs
tail -20 waf_production.log

# Review attacks blocked
tail -f nginx/logs/access.log | grep " 403 "
```

### Monitor During Day
```bash
# Live dashboard (optional)
python3 waf_dashboard.py &

# Watch for issues
tail -f waf_production.log
```

### End of Day
```bash
# Check WAF is still healthy
docker-compose ps

# Review day's statistics
grep "BLOCKING" dockerfile/logs/access.log | wc -l  # Total attacks blocked

# Keep logs backed up (optional)
cp nginx/logs/access.log "nginx/logs/access_$(date +%Y%m%d).log"
```

---

## 🌍 Cloud/Server Deployment

### AWS EC2

```bash
# 1. SSH into instance
ssh -i key.pem ubuntu@instance-ip

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 3. Clone project
git clone <your-repo> transformer-waf-test
cd transformer-waf-test

# 4. Start WAF
bash start_waf.sh

# 5. Configure firewall to allow port 8080
# (AWS Security Group: Port 8080 from anywhere, Port 22 from your IP)

# 6. Access website
# http://<instance-ip>:8080
```

### Docker Swarm (Multiple Machines)

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml waf

# Monitor
docker stack services waf
```

### Kubernetes

See `kubernetes_deployment.yaml` for K8s setup (optional).

---

## 🔐 Security Best Practices

### 1. Network Security
- ✅ WAF service (port 8000) is internal only
- ✅ Only Nginx (port 8080) exposed to internet
- ✅ Juice Shop never directly accessible

### 2. Resource Limits
- ✅ Each service has CPU/memory caps
- ✅ Prevents DoS attacks from consuming all resources
- ✅ Auto-restart if service uses too much

### 3. Log Security
- ✅ Logs show all blocked attacks
- ✅ Logs rotated automatically
- ✅ Access logs are read-only in production

### 4. Monitoring
- ✅ Background daemon detects issues
- ✅ Email alerts on failures (optional)
- ✅ Quick auto-recovery

---

## 📞 Summary

| Task | Command |
|------|---------|
| **Start 24/7 WAF** | `bash start_waf.sh` |
| **Stop WAF** | `bash stop_waf.sh` |
| **View Logs** | `tail -f waf_production.log` |
| **Live Dashboard** | `python3 waf_dashboard.py` |
| **Check Status** | `docker-compose ps` |
| **View Attacks** | `docker-compose logs waf-service` |
| **Review Results** | `tail -f nginx/logs/access.log` |

---

**Your WAF is now production-ready and will automatically protect your assets 24/7!** 🛡️

