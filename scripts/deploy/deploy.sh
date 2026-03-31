#!/usr/bin/env bash
# ============================================================
# SAM3 Deployment Script
# ─────────────────────────────────────────────────────────────
# Deploys the SAM3 application to a self-hosted server.
#
# Usage:
#   ./scripts/deploy/deploy.sh [--rollback] [--status] [--logs]
#
# Prerequisites:
#   - Docker and Docker Compose installed on target server
#   - SSH access configured
#   - .env.prod file present in the project root
#   - GitHub Container Registry access (ghcr.io)
# ============================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Colour

log_info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ── Variables ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_PATH="${DEPLOY_PATH:-/opt/sam3}"
REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE_NAME="${IMAGE_NAME:-ghcr.io/${GITHUB_REPOSITORY_OWNER:-owner}/sam3}"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.prod.yml"
ENV_FILE="$PROJECT_ROOT/.env.prod"

# ── Parse arguments ──────────────────────────────────────────
ACTION="${1:-deploy}"

# ── Helpers ────────────────────────────────────────────────────
print_header() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
}

check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check Docker
    if ! command -v docker &>/dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    log_ok "Docker: $(docker --version)"

    # Check Docker Compose
    if ! docker compose version &>/dev/null; then
        log_error "Docker Compose v2 is not available."
        exit 1
    fi
    log_ok "Docker Compose: $(docker compose version)"

    # Check .env.prod exists
    if [[ ! -f "$ENV_FILE" ]]; then
        log_error ".env.prod not found at: $ENV_FILE"
        log_info "Copy .env.prod.example to .env.prod and fill in the values."
        exit 1
    fi
    log_ok ".env.prod found"

    # Check docker-compose.prod.yml exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "docker-compose.prod.yml not found at: $COMPOSE_FILE"
        exit 1
    fi
    log_ok "docker-compose.prod.yml found"

    # Check registry login
    if ! docker registry inspect "$REGISTRY" &>/dev/null 2>&1; then
        log_warn "Not logged in to $REGISTRY. Run:"
        log_warn "  echo \$GITHUB_TOKEN | docker login $REGISTRY -u GITHUB_USERNAME --password-stdin"
    else
        log_ok "$REGISTRY login OK"
    fi
}

pull_image() {
    local tag="${1:-latest}"
    log_info "Pulling image: ${IMAGE_NAME}:${tag}"
    if ! docker pull "${IMAGE_NAME}:${tag}"; then
        log_error "Failed to pull image. Check your registry credentials."
        exit 1
    fi
    docker tag "${IMAGE_NAME}:${tag}" "sam3-api:latest"
    log_ok "Image pulled and tagged successfully"
}

stop_services() {
    log_info "Stopping existing services..."
    docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    log_ok "Services stopped"
}

start_services() {
    log_info "Starting services with docker-compose..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    log_ok "Services started"
}

wait_healthy() {
    local max_wait=120
    local waited=0
    local interval=5

    log_info "Waiting for services to become healthy (max ${max_wait}s)..."

    while (( waited < max_wait )); do
        local status
        status=$(docker compose -f "$COMPOSE_FILE" ps --format json 2>/dev/null | \
            jq -r 'select(.Service != "minio-setup" and .Service != "attu" and .Service != "nginx") | .Health' 2>/dev/null | \
            sort -u | tr '\n' ' ')

        if [[ "$status" == *"healthy"* ]] && [[ ! "$status" =~ "starting" ]]; then
            log_ok "All critical services are healthy"
            return 0
        fi

        echo -n "."
        sleep $interval
        waited=$((waited + interval))
    done

    log_warn "Timeout waiting for services to be healthy"
    docker compose -f "$COMPOSE_FILE" ps
    return 1
}

health_check() {
    print_header "Health Check"
    local api_port
    api_port=$(grep -E '^API_PORT' "$ENV_FILE" | cut -d= -f2 2>/dev/null || echo "8001")

    local status
    status=$(curl -sf -o /dev/null -w "%{http_code}" "http://localhost:${api_port}/health" 2>/dev/null || echo "000")

    if [[ "$status" == "200" ]]; then
        log_ok "API health check: HTTP $status"
    else
        log_error "API health check failed: HTTP $status"
        return 1
    fi

    # Check critical services
    local postgres_ok milvus_ok minio_ok
    postgres_ok=$(docker exec sam3-postgres-prod pg_isready -U sam3user -d sam3 &>/dev/null && echo "ok" || echo "fail")
    milvus_ok=$(curl -sf "http://localhost:19530/healthz" &>/dev/null && echo "ok" || echo "fail")
    minio_ok=$(docker exec minio-prod mc ready local &>/dev/null && echo "ok" || echo "fail")

    log_info "PostgreSQL: $postgres_ok  |  Milvus: $milvus_ok  |  MinIO: $minio_ok"

    if [[ "$postgres_ok" != "ok" ]] || [[ "$milvus_ok" != "ok" ]]; then
        log_error "One or more critical services are not healthy"
        return 1
    fi

    return 0
}

deploy() {
    print_header "Deploying SAM3 to Production"

    check_prerequisites

    log_info "Deployment path: $DEPLOY_PATH"
    log_info "Image: ${IMAGE_NAME}:latest"
    echo ""

    # Pull latest image
    pull_image "latest"

    # Stop old services
    stop_services

    # Copy updated files
    log_info "Syncing configuration files to $DEPLOY_PATH..."
    mkdir -p "$DEPLOY_PATH"
    cp "$COMPOSE_FILE" "$DEPLOY_PATH/docker-compose.prod.yml"
    cp "$ENV_FILE" "$DEPLOY_PATH/.env.prod"

    # Load new image
    log_info "Loading new image..."
    docker pull "${IMAGE_NAME}:latest"

    # Start services
    start_services

    # Wait and health check
    wait_healthy || true
    health_check

    print_header "Deployment Summary"
    log_ok "SAM3 is deployed!"
    log_info "API:        http://localhost:8001"
    log_info "Health:     http://localhost:8001/health"
    log_info "Swagger:     http://localhost:8001/docs"
    log_info "MinIO:      http://localhost:9001"
    log_info "Attu:       http://localhost:8000"
    echo ""
}

rollback() {
    print_header "Rolling Back to Previous Version"

    log_warn "This will restart services with the currently loaded image."

    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d

    wait_healthy || true
    health_check

    log_ok "Rollback complete"
}

status() {
    print_header "Service Status"

    echo ""
    docker compose -f "$COMPOSE_FILE" ps
    echo ""
    health_check
}

logs() {
    local service="${2:-sam3-api}"
    log_info "Showing logs for: $service (Ctrl+C to exit)"
    docker compose -f "$COMPOSE_FILE" logs -f "$service"
}

cleanup() {
    print_header "Cleanup Old Images"

    log_info "Removing dangling images..."
    docker image prune -af --filter "until=24h"

    log_info "Removing stopped containers..."
    docker container prune -f

    log_ok "Cleanup complete"
}

# ── Main ───────────────────────────────────────────────────────
case "$ACTION" in
    deploy)
        deploy
        ;;
    rollback)
        rollback
        ;;
    status)
        status
        ;;
    logs)
        logs "$@"
        ;;
    cleanup)
        cleanup
        ;;
    health)
        health_check
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|status|logs [service]|cleanup|health}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Full deployment to production (default)"
        echo "  rollback - Restart with currently loaded image"
        echo "  status   - Show service status"
        echo "  logs     - Show logs (default: sam3-api)"
        echo "  cleanup  - Remove old Docker images and containers"
        echo "  health   - Run health check only"
        exit 1
        ;;
esac
