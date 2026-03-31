#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
CI_IMAGE_TAG="${SAM3_LOCAL_IMAGE_TAG:-sam3:ci-local}"
RELEASE_IMAGE_TAG="${SAM3_RELEASE_IMAGE_TAG:-sam3:release-local}"
VENV_BIN="$PROJECT_ROOT/.venv/bin"
COMPOSE_PROJECT_NAME="${SAM3_COMPOSE_PROJECT_NAME:-sam3-ci-local}"
CI_POSTGRES_PORT="${SAM3_CI_POSTGRES_PORT:-15432}"
CI_MILVUS_PORT="${SAM3_CI_MILVUS_PORT:-19531}"
CI_MINIO_API_PORT="${SAM3_CI_MINIO_API_PORT:-19000}"
CI_MINIO_CONSOLE_PORT="${SAM3_CI_MINIO_CONSOLE_PORT:-19001}"
CI_ATTU_PORT="${SAM3_CI_ATTU_PORT:-18000}"
CI_API_PORT="${SAM3_CI_API_PORT:-18001}"
COMPOSE_ENV=(
    "COMPOSE_PROJECT_NAME=$COMPOSE_PROJECT_NAME"
    "SAM3_DOCKER_NETWORK=${SAM3_DOCKER_NETWORK:-${COMPOSE_PROJECT_NAME}-network}"
    "SAM3_POSTGRES_CONTAINER=${SAM3_POSTGRES_CONTAINER:-${COMPOSE_PROJECT_NAME}-postgres}"
    "SAM3_MILVUS_CONTAINER=${SAM3_MILVUS_CONTAINER:-${COMPOSE_PROJECT_NAME}-milvus}"
    "SAM3_MINIO_CONTAINER=${SAM3_MINIO_CONTAINER:-${COMPOSE_PROJECT_NAME}-minio}"
    "SAM3_ETCD_CONTAINER=${SAM3_ETCD_CONTAINER:-${COMPOSE_PROJECT_NAME}-etcd}"
    "SAM3_ATTU_CONTAINER=${SAM3_ATTU_CONTAINER:-${COMPOSE_PROJECT_NAME}-attu}"
    "SAM3_MINIO_CLIENT_CONTAINER=${SAM3_MINIO_CLIENT_CONTAINER:-${COMPOSE_PROJECT_NAME}-minio-client}"
    "SAM3_API_CONTAINER=${SAM3_API_CONTAINER:-${COMPOSE_PROJECT_NAME}-api}"
    "POSTGRES_HOST_PORT=$CI_POSTGRES_PORT"
    "MILVUS_HOST_PORT=$CI_MILVUS_PORT"
    "MINIO_API_HOST_PORT=$CI_MINIO_API_PORT"
    "MINIO_CONSOLE_HOST_PORT=$CI_MINIO_CONSOLE_PORT"
    "ATTU_HOST_PORT=$CI_ATTU_PORT"
    "API_HOST_PORT=$CI_API_PORT"
)

phase="${1:-all}"

log() {
    printf '[local-ci] %s\n' "$1"
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$1" >&2
        exit 1
    fi
}

require_venv_tool() {
    if [[ ! -x "$VENV_BIN/$1" ]]; then
        printf 'Missing required tool in .venv: %s\n' "$VENV_BIN/$1" >&2
        printf 'Run `make install` first.\n' >&2
        exit 1
    fi
}

cleanup_compose() {
    compose --profile api down -v --remove-orphans >/dev/null 2>&1 || true
}

compose() {
    env "${COMPOSE_ENV[@]}" docker compose -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" "$@"
}

wait_for_http() {
    local url="$1"
    local max_attempts="${2:-30}"
    local sleep_seconds="${3:-2}"

    for attempt in $(seq 1 "$max_attempts"); do
        if curl -sf "$url" >/dev/null; then
            return 0
        fi
        sleep "$sleep_seconds"
    done

    return 1
}

run_workflow_lint() {
    log "Validating GitHub workflows"

    if command -v actionlint >/dev/null 2>&1; then
        actionlint -color
        return
    fi

    require_cmd docker
    docker run --rm \
        -v "$PROJECT_ROOT:/repo" \
        -w /repo \
        rhysd/actionlint:latest \
        -config-file /repo/.github/actionlint.yaml \
        -color
}

run_quality() {
    require_venv_tool ruff
    require_venv_tool pyright
    log "Running lint, format, and type checks"
    if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
        "$VENV_BIN/ruff" check src/ tests/ --output-format=github
    else
        "$VENV_BIN/ruff" check src/ tests/
    fi
    "$VENV_BIN/ruff" check --select=I --ignore=I001 src/ tests/
    "$VENV_BIN/ruff" format --check tests/
    "$VENV_BIN/pyright" src/
}

run_unit() {
    require_venv_tool pytest
    log "Running unit tests"
    "$VENV_BIN/pytest" tests/ -m unit -v --tb=short --no-header -q
}

run_integration() {
    require_cmd docker
    require_venv_tool pytest
    trap cleanup_compose RETURN

    log "Starting integration infrastructure"
    compose up -d --wait postgres etcd minio milvus

    log "Running integration tests"
    env \
        POSTGRES_HOST=127.0.0.1 \
        POSTGRES_PORT="$CI_POSTGRES_PORT" \
        POSTGRES_DB=sam3 \
        POSTGRES_USER=sam3user \
        POSTGRES_PASSWORD=sam3pass \
        MILVUS_URI="http://127.0.0.1:$CI_MILVUS_PORT" \
        MINIO_ENDPOINT="127.0.0.1:$CI_MINIO_API_PORT" \
        MINIO_SECURE=false \
        MINIO_ACCESS_KEY=minioadmin \
        MINIO_SECRET_KEY=minioadmin \
        "$VENV_BIN/pytest" tests/ -m "integration and not slow and not gpu" -v --tb=short --no-header -q
}

run_image_smoke() {
    require_cmd docker
    require_cmd curl
    trap cleanup_compose RETURN

    if [[ "${SAM3_SKIP_IMAGE_BUILD:-0}" != "1" ]]; then
        log "Building production image for local smoke test"
        docker build --target production -t "$CI_IMAGE_TAG" "$PROJECT_ROOT"
    fi

    log "Starting isolated API smoke stack ($COMPOSE_PROJECT_NAME) with image tag $CI_IMAGE_TAG"
    (
        export ENV="${ENV:-staging}"
        export SAM3_IMAGE_TAG="$CI_IMAGE_TAG"
        compose --profile api up -d --wait sam3-api
    )

    if ! wait_for_http "http://127.0.0.1:$CI_API_PORT/health" 30 2; then
        compose logs sam3-api || true
        printf 'API failed health check during image smoke test\n' >&2
        exit 1
    fi
}

run_release_preflight() {
    require_cmd docker
    require_cmd curl
    require_venv_tool pytest
    trap cleanup_compose RETURN

    log "Running release preflight tests"
    "$VENV_BIN/pytest" tests/ -m "not slow and not gpu and not integration" -v --tb=short

    if [[ "${SAM3_SKIP_IMAGE_BUILD:-0}" != "1" ]]; then
        log "Building production image for local release preflight"
        docker build --target production -t "$RELEASE_IMAGE_TAG" "$PROJECT_ROOT"
    fi

    log "Starting isolated release smoke stack ($COMPOSE_PROJECT_NAME) with image tag $RELEASE_IMAGE_TAG"
    (
        export ENV=prod
        export SAM3_IMAGE_TAG="$RELEASE_IMAGE_TAG"
        compose --profile api up -d --wait sam3-api
    )

    if ! wait_for_http "http://127.0.0.1:$CI_API_PORT/health" 30 2; then
        compose logs sam3-api || true
        printf 'Release smoke container failed health check\n' >&2
        exit 1
    fi
}

case "$phase" in
    workflow-lint)
        run_workflow_lint
        ;;
    quality)
        run_quality
        ;;
    unit)
        run_unit
        ;;
    integration)
        run_integration
        ;;
    image-smoke)
        run_image_smoke
        ;;
    release-preflight)
        run_release_preflight
        ;;
    all)
        run_workflow_lint
        run_quality
        run_unit
        run_integration
        run_image_smoke
        ;;
    *)
        cat <<'EOF' >&2
Usage: ./scripts/local-ci.sh {workflow-lint|quality|unit|integration|image-smoke|release-preflight|all}
EOF
        exit 1
        ;;
esac
