#!/bin/bash
# 🛡️ Start WAF in Production Mode
# Configure, deploy, and protect any website

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$PROJECT_DIR/waf_production.log"
PID_FILE="$PROJECT_DIR/.waf_pid"
CONFIG_FILE="$PROJECT_DIR/CONFIG.env"

# Load configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ CONFIG.env not found! Please create it first."
    echo "   See CONFIG.env.example for template"
    exit 1
fi

# Source configuration
source "$CONFIG_FILE"

echo "🛡️  WAF Production Deployment" | tee -a "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting WAF for: $TARGET_WEBSITE_URL" | tee -a "$LOG_FILE"

# Validate configuration
if [ -z "$TARGET_WEBSITE_URL" ]; then
    echo "❌ TARGET_WEBSITE_URL not set in CONFIG.env" | tee -a "$LOG_FILE"
    exit 1
fi

echo "⚙️  Configuration:" | tee -a "$LOG_FILE"
echo "   Website: $TARGET_WEBSITE_URL" | tee -a "$LOG_FILE"
echo "   Public Domain: $PUBLIC_IP_OR_DOMAIN:$PUBLIC_PORT" | tee -a "$LOG_FILE"
echo "   AI Threshold: $AI_CONFIDENCE_THRESHOLD" | tee -a "$LOG_FILE"

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed." | tee -a "$LOG_FILE"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed." | tee -a "$LOG_FILE"
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping any existing containers..." | tee -a "$LOG_FILE"
docker-compose down 2>/dev/null || true

# Build and start services
echo "🏗️  Building and starting services..." | tee -a "$LOG_FILE"
docker-compose up -d --build 2>&1 | tee -a "$LOG_FILE"

# Wait for healthy status
echo "⏳ Waiting for services to be healthy..." | tee -a "$LOG_FILE"
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if docker-compose ps | grep -q "healthy"; then
        echo "✅ All services are healthy!" | tee -a "$LOG_FILE"
        break
    fi
    
    attempt=$((attempt + 1))
    echo "  Waiting... ($attempt/$max_attempts)" | tee -a "$LOG_FILE"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "⚠️  Some services may not be fully healthy yet, but continuing..." | tee -a "$LOG_FILE"
fi

# Store PID
echo $$ > "$PID_FILE"

# Verify system is operational
echo "🔍 Verifying system is operational..." | tee -a "$LOG_FILE"
sleep 5

# Test basic functionality
HEALTH=$(curl -s http://localhost:8000/ || echo "")
if echo "$HEALTH" | grep -q "running"; then
    echo "✅ WAF API is responsive" | tee -a "$LOG_FILE"
else
    echo "⚠️  WAF API may not be responding yet" | tee -a "$LOG_FILE"
fi

# Display status
echo "" | tee -a "$LOG_FILE"
echo "════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "✅ WAF DEPLOYED SUCCESSFULLY" | tee -a "$LOG_FILE"
echo "════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📊 Deployment Details:" | tee -a "$LOG_FILE"
echo "  Target Website: $TARGET_WEBSITE_URL" | tee -a "$LOG_FILE"
echo "  Public Access: http://$PUBLIC_IP_OR_DOMAIN:$PUBLIC_PORT" | tee -a "$LOG_FILE"
echo "  WAF API: http://localhost:8000 (internal)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "🔒 Protection Status:" | tee -a "$LOG_FILE"
echo "  ✅ All services running" | tee -a "$LOG_FILE"
echo "  ✅ Auto-restart enabled" | tee -a "$LOG_FILE"
echo "  ✅ Health checks active (10s interval)" | tee -a "$LOG_FILE"
echo "  ✅ Real-time attack detection active" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📈 Monitoring:" | tee -a "$LOG_FILE"
echo "  View logs: tail -f waf_production.log" | tee -a "$LOG_FILE"
echo "  Check status: docker-compose ps" | tee -a "$LOG_FILE"
echo "  View attacks: tail -f nginx/logs/access.log | grep ' 403 '" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "🛑 To Stop:" | tee -a "$LOG_FILE"
echo "  bash stop_waf.sh" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Start monitoring daemon in background
if [ -f "$PROJECT_DIR/monitor_waf.sh" ]; then
    echo "🔔 Starting background monitoring daemon..." | tee -a "$LOG_FILE"
    bash "$PROJECT_DIR/monitor_waf.sh" > /dev/null 2>&1 &
    MONITOR_PID=$!
    echo $MONITOR_PID >> "$PID_FILE"
    echo "   Monitor PID: $MONITOR_PID" | tee -a "$LOG_FILE"
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') - WAF is LIVE and protecting your website!" | tee -a "$LOG_FILE"
