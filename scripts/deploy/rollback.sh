#!/usr/bin/env bash
# ============================================================
# SAM3 Health Check & Monitoring Script
# ─────────────────────────────────────────────────────────────
# Run via cron or systemd timer for continuous monitoring.
# Usage: ./scripts/deploy/monitor.sh [--alert]
# ============================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────
API_URL="${API_URL:-http://localhost:8001}"
ALERT_SCRIPT="${ALERT_SCRIPT:-}"
LOG_FILE="${LOG_FILE:-/var/log/sam3-health.log}"
MAX_RESPONSE_TIME=5000  # milliseconds

# ── Colours ───────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── Functions ─────────────────────────────────────────────────
log() {
    local level="$1"
    local msg="$2"
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${timestamp} [$level] $msg"

    if [[ -n "$LOG_FILE" ]]; then
        echo "${timestamp} [$level] $msg" >> "$LOG_FILE"
    fi
}

alert() {
    local msg="$1"
    log "ALERT" "$msg"

    if [[ -n "$ALERT_SCRIPT" ]] && [[ -x "$ALERT_SCRIPT" ]]; then
        "$ALERT_SCRIPT" "$msg"
    elif command -v curl &>/dev/null; then
        # Discord/Slack webhook support via env vars
        if [[ -n "${DISCORD_WEBHOOK:-}" ]]; then
            curl -fsSL -X POST "$DISCORD_WEBHOOK" \
                -H "Content-Type: application/json" \
                -d "{\"content\": \"⚠️ SAM3 Alert: $msg\"}" 2>/dev/null || true
        fi
    fi
}

check_api() {
    local response_time http_code
    response_time=$(curl -sf -o /dev/null -w "%{time_total}" "$API_URL/health" 2>/dev/null)
    http_code=$(curl -sf -o /dev/null -w "%{http_code}" "$API_URL/health" 2>/dev/null || echo "000")

    local response_ms
    response_ms=$(python3 -c "print(int(${response_time:-999} * 1000))" 2>/dev/null || echo "9999")

    if [[ "$http_code" != "200" ]]; then
        alert "API unhealthy: HTTP $http_code (expected 200)"
        return 1
    fi

    if (( response_ms > MAX_RESPONSE_TIME )); then
        alert "API slow: ${response_ms}ms (threshold: ${MAX_RESPONSE_TIME}ms)"
        return 1
    fi

    log "OK" "API healthy (HTTP $http_code, ${response_ms}ms)"
    return 0
}

check_postgres() {
    if docker exec sam3-postgres-prod pg_isready -U sam3user -d sam3 &>/dev/null; then
        log "OK" "PostgreSQL healthy"
        return 0
    else
        alert "PostgreSQL unhealthy"
        return 1
    fi
}

check_milvus() {
    if curl -sf "http://localhost:19530/healthz" | grep -q "OK" 2>/dev/null; then
        log "OK" "Milvus healthy"
        return 0
    else
        alert "Milvus unhealthy"
        return 1
    fi
}

check_minio() {
    if docker exec minio-prod mc ready local &>/dev/null; then
        log "OK" "MinIO healthy"
        return 0
    else
        alert "MinIO unhealthy"
        return 1
    fi
}

check_disk_space() {
    local threshold=85
    local usage
    usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')

    if (( usage > threshold )); then
        alert "Disk usage critical: ${usage}% (threshold: ${threshold}%)"
        return 1
    fi

    log "OK" "Disk usage: ${usage}%"
    return 0
}

check_memory() {
    local threshold=90
    local usage
    usage=$(free | awk '/Mem:/ {printf "%.0f", $3/$2 * 100}')

    if (( usage > threshold )); then
        alert "Memory usage high: ${usage}% (threshold: ${threshold}%)"
        return 1
    fi

    log "OK" "Memory usage: ${usage}%"
    return 0
}

# ── Main ───────────────────────────────────────────────────────
do_alert=false
if [[ "${1:-}" == "--alert" ]]; then
    do_alert=true
fi

all_ok=true

check_api       || all_ok=false
check_postgres  || all_ok=false
check_milvus    || all_ok=false
check_minio     || all_ok=false
check_disk_space || all_ok=false
check_memory    || all_ok=false

if $all_ok; then
    log "INFO" "All checks passed"
    exit 0
else
    log "ERROR" "One or more checks failed"
    exit 1
fi
