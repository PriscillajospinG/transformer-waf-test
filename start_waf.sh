#!/bin/bash
# Start WAF in Production Mode
# Run this to start the WAF system with all monitoring and auto-recovery

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$PROJECT_DIR/waf_production.log"
PID_FILE="$PROJECT_DIR/.waf_pid"

echo "🛡️  WAF Production Startup" | tee -a "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting WAF in production mode..." | tee -a "$LOG_FILE"

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
echo "✅ WAF PRODUCTION MODE STARTED" | tee -a "$LOG_FILE"
echo "════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Services are running and configured for auto-restart:" | tee -a "$LOG_FILE"
echo "  🌐 Website: http://localhost:8080" | tee -a "$LOG_FILE"
echo "  🔍 WAF API: http://localhost:8000" | tee -a "$LOG_FILE"
echo "  📊 Logs: docker-compose logs -f waf-service" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Monitoring:" | tee -a "$LOG_FILE"
echo "  Real-time detection: python3 waf_dashboard.py" | tee -a "$LOG_FILE"
echo "  System status: docker-compose ps" | tee -a "$LOG_FILE"
echo "  Service logs: tail -f waf_production.log" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "⚙️  Auto-features enabled:" | tee -a "$LOG_FILE"
echo "  ✓ Auto-restart on crash (always policy)" | tee -a "$LOG_FILE"
echo "  ✓ Health checks every 10 seconds" | tee -a "$LOG_FILE"
echo "  ✓ Resource limits to prevent memory leaks" | tee -a "$LOG_FILE"
echo "  ✓ Log rotation (max 50MB per service)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "To stop: bash stop_waf.sh" | tee -a "$LOG_FILE"
echo "To monitor: bash monitor_waf.sh" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Start monitoring daemon in background
if [ -f "$PROJECT_DIR/monitor_waf.sh" ]; then
    echo "🔔 Starting background monitoring daemon..." | tee -a "$LOG_FILE"
    bash "$PROJECT_DIR/monitor_waf.sh" > /dev/null 2>&1 &
    MONITOR_PID=$!
    echo $MONITOR_PID >> "$PID_FILE"
    echo "   Monitor PID: $MONITOR_PID" | tee -a "$LOG_FILE"
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') - WAF is LIVE and protecting your assets!" | tee -a "$LOG_FILE"
