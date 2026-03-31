#!/usr/bin/env python3
"""
SAM3 API module.

Contains route handlers, helpers, and lifespan management.
"""

from .helpers import (
    parse_timestamp,
    create_time_filter,
    NumpyEncoder,
    convert_numpy_types,
    cleanup_gpu_memory,
    ApiEnvelope,
    _envelope_json_response,
)

from .lifespan import (
    lifespan,
    get_ingestion_service,
    get_verification_service,
    get_ingest_semaphore,
    get_ingest_executor,
)

from .debugRoutes import router as debug_router
from .trees import router as trees_router

__all__ = [
    # Helpers
    "parse_timestamp",
    "create_time_filter",
    "NumpyEncoder",
    "convert_numpy_types",
    "cleanup_gpu_memory",
    # Envelope
    "ApiEnvelope",
    "_envelope_json_response",
    # Lifespan
    "lifespan",
    "get_ingestion_service",
    "get_verification_service",
    "get_ingest_semaphore",
    "get_ingest_executor",
    # Routers
    "debug_router",
    "trees_router",
]
