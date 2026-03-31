# ============================================================
# SAM3 - Multi-stage production Dockerfile
# Stage 1: Builder   — installs deps, downloads models
# Stage 2: Production — minimal runtime image
# Stage 3: Development — with hot-reload tools (tag: sam3:dev)
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies (no unnecessary packages in runtime image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libturbojpeg0-dev \
    && rm -rf /var/lib/apt/lists/* \
    && find /usr/lib/python3.12/ -name "*.pyc" -delete \
    && find /usr/lib/python3.12/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Copy only dependency files first (layer cache optimisation)
COPY pyproject.toml ./

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (cpu + cuda variants handled by pip)
# Core + dev dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir \
        "pydantic>=2.0" \
        "pydantic-settings>=2.0" \
        fastapi \
        uvicorn[standard] \
        "uvloop>=0.17" \
        "httptools>=0.5" \
        "dependency-injector>=4.0" \
        pytest \
        pytest-asyncio \
        pytest-cov \
        "pillow>=10.0" \
        "numpy>=1.24" \
        opencv-python-headless \
        "pyyaml>=6.0" \
        "matplotlib>=3.7" \
        "ftfy>=6.1" \
        "regex>=2023.0" \
        httpx \
        aiofiles

# ── Stage 2: Production ──────────────────────────────────────
FROM python:3.12-slim AS production

# Security: run as non-root user
RUN groupadd --gid 1000 sam3 && useradd --uid 1000 --gid sam3 --shell /bin/bash --create-home sam3

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && find /usr/lib/python3.12/ -name "*.pyc" -delete \
    && find /usr/lib/python3.12/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Copy source code
COPY --chown=sam3:sam3 . .

# Environment defaults (overridden at runtime via --env-file)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    TOKENIZERS_PARALLELISM=false \
    LOG_LEVEL=INFO \
    ENV=prod \
    PYTHONOPTIMIZE=1

USER sam3

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

# Expose API port
EXPOSE 8001

# Run with uvicorn (4 workers, binds to 0.0.0.0)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]

# ── Stage 3: Development ──────────────────────────────────────
FROM python:3.12-slim AS development

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libturbojpeg0-dev \
        nano \
        vim \
        htop \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata + source before editable install
COPY pyproject.toml README.md ./
COPY src ./src

# Install torch CPU-only FIRST (from PyTorch CPU index) so pip skips it
# during editable install and does NOT pull in the ~2 GB CUDA stack.
# The CPU wheel has no nvidia-*/cuda* transitive dependencies.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.4.0" \
        "torchvision>=0.19.0" && \
    pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir \
        "uvloop>=0.17" \
        "httptools>=0.5"

# Development environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=DEBUG \
    ENV=dev

# Development command with hot-reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload", "--reload-dir", "src", "--reload-dir", "."]
