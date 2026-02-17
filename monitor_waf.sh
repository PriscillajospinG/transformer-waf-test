#!/bin/bash
# Monitor WAF Production System
# Continuously monitors health and auto-restarts if needed

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$PROJECT_DIR/waf_production.log"

echo "🔔 WAF Monitoring Daemon Started"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Monitoring started" >> "$LOG_FILE"

# Configuration
CHECK_INTERVAL=30  # Check every 30 seconds
ALERT_EMAIL=""     # Optional: set to email for alerts
RESTART_ATTEMPTS=0
MAX_RESTART_ATTEMPTS=5

# Function to check service health
check_services() {
    local unhealthy=0
    
    # Check WAF API
    if ! curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "⚠️  WAF API is down!" >> "$LOG_FILE"
        unhealthy=$((unhealthy + 1))
    fi
    
    # Check Nginx
    if ! curl -s http://localhost:8080/ > /dev/null 2>&1; then
        echo "⚠️  Nginx is down!" >> "$LOG_FILE"
        unhealthy=$((unhealthy + 1))
    fi
    
    # Check Docker status
    if ! docker-compose ps | grep -q "Up"; then
        echo "⚠️  Some containers are not running!" >> "$LOG_FILE"
        unhealthy=$((unhealthy + 1))
    fi
    
    return $unhealthy
}

# Function to send alert
send_alert() {
    local message="$1"
    
    echo "🚨 ALERT: $message" | tee -a "$LOG_FILE"
    
    # Optional: Send email alert
    if [ -n "$ALERT_EMAIL" ]; then
        echo "Alert: $message" | mail -s "WAF Alert" "$ALERT_EMAIL" 2>/dev/null || true
    fi
}

# Function to restart services
restart_services() {
    echo "🔄 Attempting to restart services..." >> "$LOG_FILE"
    
    RESTART_ATTEMPTS=$((RESTART_ATTEMPTS + 1))
    
    if [ $RESTART_ATTEMPTS -gt $MAX_RESTART_ATTEMPTS ]; then
        send_alert "Max restart attempts ($MAX_RESTART_ATTEMPTS) reached - manual intervention required!"
        exit 1
    fi
    
    docker-compose down 2>&1 >> "$LOG_FILE"
    sleep 5
    docker-compose up -d --build 2>&1 >> "$LOG_FILE"
    
    echo "✅ Restart attempt #$RESTART_ATTEMPTS completed" >> "$LOG_FILE"
    sleep 10
}

# Main monitoring loop
loop_count=0
last_status="healthy"

while true; do
    loop_count=$((loop_count + 1))
    timestamp=$(date '+%H:%M:%S')
    
    check_services
    service_status=$?
    
    if [ $service_status -eq 0 ]; then
        # All services healthy
        if [ "$last_status" != "healthy" ]; then
            echo "✅ [$timestamp] Services recovered" >> "$LOG_FILE"
            last_status="healthy"
            RESTART_ATTEMPTS=0  # Reset counter
        fi
        
        # Log status periodically
        if [ $((loop_count % 10)) -eq 0 ]; then
            echo "✅ [$timestamp] All services healthy (Check #$loop_count)" >> "$LOG_FILE"
        fi
    else
        # Services unhealthy
        if [ "$last_status" != "unhealthy" ]; then
            send_alert "Services detected unhealthy at $timestamp"
            last_status="unhealthy"
        fi
        
        restart_services
    fi
    
    sleep $CHECK_INTERVAL
done
