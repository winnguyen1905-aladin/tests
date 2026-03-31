#!/usr/bin/env python3
"""
Helper utilities for SAM3 API.

Contains shared utility functions for timestamp parsing, type conversion,
GPU memory management, JSON encoding, and the uniform API response envelope.
"""

import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.config.appConfig import get_config

logger = logging.getLogger(__name__)

# =============================================================================
# Timestamp Helper Functions
# =============================================================================

THREE_MONTHS_SECONDS = 90 * 24 * 60 * 60  # ±3 months in seconds


def parse_timestamp(timestamp_str: str) -> int:
    """Parse ISO 8601 timestamp string to Unix epoch seconds.

    Args:
        timestamp_str: ISO 8601 formatted timestamp (e.g., "2025-06-01T10:30:00Z")

    Returns:
        Unix epoch seconds as int

    Raises:
        ValueError: If the timestamp string is invalid
    """
    if not timestamp_str:
        raise ValueError("Timestamp string is empty")

    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return int(dt.timestamp())
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}") from e


def create_time_filter(query_timestamp: int, range_seconds: int = THREE_MONTHS_SECONDS) -> dict:
    """Create a time filter with ±range_seconds from query timestamp.

    Args:
        query_timestamp: Query timestamp in Unix epoch seconds
        range_seconds: Time range in seconds (default: ±3 months)

    Returns:
        Dict with captured_at_min and captured_at_max
    """
    return {
        'captured_at_min': query_timestamp - range_seconds,
        'captured_at_max': query_timestamp + range_seconds,
    }


# =============================================================================
# JSON Encoding Utilities
# =============================================================================


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        # Handle all numpy scalar types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            val = float(obj)
            # Handle inf/nan values
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            # Also handle inf/nan in arrays
            arr = obj.tolist()
            return arr
        # Handle regular Python types that might be inf/nan
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        if isinstance(obj, int):
            return int(obj)
        if isinstance(obj, bool):
            return bool(obj)
        return super().default(obj)


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types.

    Args:
        obj: Object to convert (can be dict, list, numpy type, etc.)

    Returns:
        Object with all numpy types converted to Python native types
    """
    # Handle None
    if obj is None:
        return None

    # Handle numpy types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        # Handle inf/nan values - convert to None for JSON compatibility
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        # Recursively convert array elements
        return [convert_numpy_types(x) for x in obj.tolist()]

    # Handle built-in Python types
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, bool):
        return bool(obj)

    # Handle containers
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]

    return obj


# =============================================================================
# GPU Memory Management
# =============================================================================


def cleanup_gpu_memory():
    """Release GPU memory aggressively based on model_mode.

    Always calls gc.collect() and torch.cuda.empty_cache() to prevent memory fragmentation.
    Only skips torch.cuda.reset_peak_memory_stats() in ultra mode for performance.
    """
    try:
        app_config = get_config()

        if torch.cuda.is_available():
            # Always do basic cleanup to prevent memory fragmentation
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            if app_config.model_mode == "lite":
                # Lite mode: aggressive cleanup including peak memory stats
                torch.cuda.reset_peak_memory_stats()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
                logger.info(f"GPU memory cleared (mode={app_config.model_mode})")
            else:
                # Ultra mode: skip peak memory stats for performance
                logger.debug(f"GPU memory cleanup (mode={app_config.model_mode})")
    except Exception as e:
        logger.warning(f"Error during GPU cleanup: {e}")


# =============================================================================
# API Response Envelope
# =============================================================================


class ApiEnvelope(BaseModel):
    """Uniform JSON envelope for every HTTP response from this app."""

    status_code: int = Field(..., description="Logical status; mirrors HTTP status code")
    message: str = Field(..., description="Human-readable summary")
    error: str | Dict[str, Any] | List[Any] | None = Field(
        default=None,
        description="Error payload when failed; null on success",
    )
    data: Dict[str, Any] | None = Field(default=None, description="Response payload; null on error")

    model_config = ConfigDict(extra="forbid")


def _envelope_json_response(
    http_status: int,
    message: str,
    error: str | Dict[str, Any] | List[Any] | None = None,
    data: Dict[str, Any] | None = None,
) -> JSONResponse:
    """Build a ``JSONResponse`` whose body is always an :class:`ApiEnvelope`."""
    body = ApiEnvelope(
        status_code=http_status,
        message=message,
        error=error,
        data=data,
    ).model_dump()
    return JSONResponse(status_code=http_status, content=body)
