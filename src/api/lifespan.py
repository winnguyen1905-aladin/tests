#!/usr/bin/env python3
"""
Lifespan management for SAM3 API.

Handles application startup/shutdown:
- ``initialize_application_resources()`` for PostgreSQL (``init_db``)
- ``shutdown_application_resources()`` for DB disconnect and GPU cleanup
Service instances are provided by the container (get_ingestion_service/get_verification_service).
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from src.service.ingestionService import IngestionService
from src.service.verificationService import VerificationService

logger = logging.getLogger(__name__)

# =============================================================================
# Service getters (delegate to container)
# =============================================================================

def get_ingestion_service() -> Optional[IngestionService]:
    """Get ingestion service from the DI container."""
    try:
        from src.config.containers import container
        return container.ingestion_service()
    except Exception as e:
        logger.warning(f"Container not ready: {e}")
        return None


def get_verification_service() -> Optional[VerificationService]:
    """Get verification service from the DI container."""
    try:
        from src.config.containers import container
        return container.verification_service()
    except Exception as e:
        logger.warning(f"Container not ready: {e}")
        return None

_INGEST_CONCURRENCY = 1  # Default, can be overridden by env var
_ingest_semaphore: Optional[asyncio.Semaphore] = None
# Single-thread executor: SAM3SemanticPredictor is NOT thread-safe.
_ingest_executor: Optional[ThreadPoolExecutor] = None


def get_ingest_semaphore() -> Optional[asyncio.Semaphore]:
    """Get the ingest semaphore for concurrency control."""
    return _ingest_semaphore


def get_ingest_executor() -> Optional[ThreadPoolExecutor]:
    """Get the ingest executor for running ingestion tasks."""
    return _ingest_executor


@asynccontextmanager
async def lifespan(_app):
    """Initialize and cleanup via DI container.

    Startup: ``initialize_application_resources()`` (database); services from the container on use.
    Shutdown: ``shutdown_application_resources()`` (DB + GPU cleanup).
    """
    global _ingest_semaphore, _ingest_executor

    import os

    logger.info("Starting SAM3 Tree Identification API...")
    logger.info(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

    _INGEST_CONCURRENCY = int(os.environ.get("INGEST_CONCURRENCY", "1"))
    _ingest_semaphore = asyncio.Semaphore(_INGEST_CONCURRENCY)
    _ingest_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
    logger.info(f"Ingest concurrency limit: {_INGEST_CONCURRENCY} (set INGEST_CONCURRENCY env to change)")

    # ── 1. Validate vector dimension before any DB work ──────────────────────
    # get_vector_dim() returns the fallback (384) when called at import time
    # before config is ready.  validate_vector_dim() checks that the fallback
    # matches the real postgres_vector_dim from config — raises if they differ.
    # This must run here, not at import time, because config may not be ready yet.
    try:
        from src.repository.entityModels import validate_vector_dim

        dim = validate_vector_dim()
        logger.info(f"✓ Vector dimension validated: {dim}")
    except RuntimeError as e:
        logger.error(f"Vector dimension mismatch: {e}")
        raise

    try:
        from src.config.containers import initialize_application_resources

        initialize_application_resources()
        logger.info("✓ Application resources initialized (PostgreSQL / init_db)")
    except Exception as e:
        logger.error(f"Failed to initialize application resources: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    yield

    logger.info("Shutting down application resources...")
    try:
        from src.config.containers import shutdown_application_resources

        shutdown_application_resources()
    except Exception as e:
        logger.warning(f"Error during application shutdown: {e}")
    logger.info("Shutdown complete")
