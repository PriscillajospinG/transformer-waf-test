#!/bin/bash
# Stop WAF Production Mode
# Gracefully stops all services

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$PROJECT_DIR/waf_production.log"
PID_FILE="$PROJECT_DIR/.waf_pid"

echo "🛑 Stopping WAF services..." | tee -a "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Shutdown initiated" | tee -a "$LOG_FILE"

# Kill monitoring daemon if running
if [ -f "$PID_FILE" ]; then
    while IFS= read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Stopping monitor PID $pid..." | tee -a "$LOG_FILE"
            kill "$pid" 2>/dev/null || true
        fi
    done < "$PID_FILE"
    rm "$PID_FILE"
fi

# Graceful shutdown of containers
echo "🔄 Stopping Docker containers (graceful shutdown)..." | tee -a "$LOG_FILE"
docker-compose down 2>&1 | tee -a "$LOG_FILE"

echo "✅ WAF services stopped successfully" | tee -a "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Shutdown complete" | tee -a "$LOG_FILE"
