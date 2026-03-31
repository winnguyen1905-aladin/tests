# ============================================================
# SAM3 Makefile — Development shortcuts
# ============================================================
# Run `make help` to see all targets.
# ============================================================

.PHONY: help install dev test test-unit test-integration test-gpu \
        build build-dev lint format clean logs logs-api \
        up down restart health deploy

# ── Colours ──────────────────────────────────────────────────
GREEN  := $(shell tput -Tscreen bold 2>/dev/null && tput setaf 2 || echo "")
CYAN  := $(shell tput -Tscreen bold 2>/dev/null && tput setaf 6 || echo "")
YELLOW:= $(shell tput -Tscreen bold 2>/dev/null && tput setaf 3 || echo "")
RESET := $(shell tput -Tscreen sgr0 2>/dev/null || echo "")

# ── Default ──────────────────────────────────────────────────
.DEFAULT_GOAL := help

# ── Help ─────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(CYAN)%-22s$(RESET) %s\n", $$1, $$2}'

# ── Development ──────────────────────────────────────────────
install: ## Install dependencies (creates venv + installs)
	uv sync --all-extras --dev

dev: ## Start development server with hot-reload
	docker compose -f docker-compose.yml up -d --wait
	python main.py

build: ## Build production Docker image
	docker build --target production --tag sam3-api:latest .

build-dev: ## Build development Docker image
	docker build --target development --tag sam3-api:dev .

# ── Testing ──────────────────────────────────────────────────
test: ## Run all tests (unit + integration)
	uv run pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	uv run pytest tests/ -m unit -v --tb=short

test-integration: ## Run integration tests (requires infra)
	docker compose -f docker-compose.yml up -d --wait
	uv run pytest tests/ -m integration -v --tb=short

test-gpu: ## Run GPU tests (requires GPU runner)
	docker compose -f docker-compose.yml up -d --wait
	uv run pytest tests/ -m gpu -v --tb=short

test-coverage: ## Run tests with coverage report
	uv run pytest tests/ -m unit -v --cov=src --cov-report=term-missing --cov-report=html --tb=short

test-fast: ## Run fast unit tests only
	uv run pytest tests/ -m "unit and not slow" -v --tb=short

# ── Code Quality ──────────────────────────────────────────────
lint: ## Run linter (Ruff)
	uv run ruff check src/ tests/

lint-fix: ## Run linter with auto-fix
	uv run ruff check --fix src/ tests/

format: ## Format code (Ruff)
	uv run ruff format src/ tests/

typecheck: ## Type check (Pyright)
	uv run pyright src/

security: ## Run security checks (Bandit)
	uv run bandit -r src/

check: lint format typecheck ## Run all checks

# ── Docker / Infrastructure ──────────────────────────────────
up: ## Start all infrastructure services
	docker compose -f docker-compose.yml up -d --wait
	@echo "Services ready. Run 'make dev' to start the API."

down: ## Stop all services
	docker compose -f docker-compose.yml down --remove-orphans

restart: down up ## Restart all services

logs: ## Show all logs
	docker compose -f docker-compose.yml logs -f

logs-api: ## Show API logs only
	docker compose -f docker-compose.yml logs -f sam3-api

logs-postgres: ## Show PostgreSQL logs
	docker compose -f docker-compose.yml logs -f sam3-postgres

logs-milvus: ## Show Milvus logs
	docker compose -f docker-compose.yml logs -f milvus

ps: ## Show service status
	docker compose -f docker-compose.yml ps

health: ## Run health check
	@echo "API:";  curl -sf http://localhost:8001/health | python -m json.tool || echo "FAIL"; \
	echo "Milvus:"; curl -sf http://localhost:19530/healthz || echo "FAIL"

# ── Production ──────────────────────────────────────────────
deploy: ## Deploy to production server (requires .env.prod)
	@./scripts/deploy/deploy.sh deploy

rollback: ## Rollback production deployment
	@./scripts/deploy/deploy.sh rollback

deploy-status: ## Check production status
	@./scripts/deploy/deploy.sh status

# ── Maintenance ──────────────────────────────────────────────
clean: ## Clean build artifacts and cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ .ruff_cache/
	@echo "Clean complete"

clean-images: ## Remove old Docker images
	docker image prune -af --filter "until=24h"

prune: clean clean-images ## Full cleanup (cache + old images)

# ── CI helpers (used by GitHub Actions) ──────────────────────
ci-test-unit:
	uv run pytest tests/ -m unit -v --tb=short --no-header -q

ci-test-integration:
	docker compose -f docker-compose.yml up -d --wait
	uv run pytest tests/ -m integration -v --tb=short --no-header -q
	docker compose -f docker-compose.yml down -v

ci-build:
	docker build --target production --tag sam3-api:latest .
