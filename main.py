#!/usr/bin/env python3
"""FastAPI entry point for SAM3 Tree Identification System."""

import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch

from fastapi import FastAPI, HTTPException, Request, Depends, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dependency_injector.wiring import inject, Provide
from src.config.containers import Container, container

from src.config import setup_logging, get_config  # Unified config module
from src.api import (
    lifespan,
    parse_timestamp,
    create_time_filter,
    convert_numpy_types,
    cleanup_gpu_memory,
    get_ingest_semaphore,
    get_ingest_executor,
    debug_router,
    trees_router,
    ApiEnvelope,
    _envelope_json_response,
)
from src.service.ingestionService import IngestionService
from src.service.verificationService import VerificationService

# Set CUDA environment variables BEFORE importing any CUDA libs
# Check if using AMD ROCm (expandable_segments not supported)
is_rocm = False
try:
    is_rocm = getattr(torch.version, 'hip', None) is not None
except Exception:
    pass

if is_rocm:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
else:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'

if 'TOKENIZERS_PARALLELISM' not in os.environ:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Initialise logging early (reads LOG_LEVEL / LOG_FILE env vars).
# Third-party noisy loggers are silenced inside setup_logging().
setup_logging()

logger = logging.getLogger(__name__)

# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(title="SAM3 Tree Identification API", lifespan=lifespan)


# =============================================================================
# Service reference for testing (allows mocking via patch('main.ingestion_service'))
# The /ingest and /verify endpoints use container.ingestion_service() directly,
# but this reference exists so existing tests that patch main.ingestion_service
# can still work (the mock replaces the factory reference used by the endpoint).
# =============================================================================
ingestion_service = container.ingestion_service
verification_service = container.verification_service

# Wire DI container so routes can use Depends(Provide[Container.xxx])
# Wire all service modules for @inject pattern
container.wire(modules=[
    "main",
    "src.service.preprocessorService",
    "src.service.ingestionService",
    "src.service.verificationService",
    "src.service.hierarchicalMatchingService",
])

# Read CORS allowed origins from environment variable (before app initialization)
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Warn if CORS origins not explicitly set in production
if "ALLOWED_ORIGINS" not in os.environ:
    logger.warning(
        "ALLOWED_ORIGINS not set in environment. Using default '{_raw_origins}'. "
        "Set ALLOWED_ORIGINS explicitly for production to prevent CSRF vulnerabilities."
    )

# Mount static files for ingest-data directory
# This allows serving images from http://localhost:8001/ingest-data/...
ingest_data_path = Path(__file__).parent / "ingest-data"
if ingest_data_path.exists():
    # Validate path is a directory and resolve symlinks to prevent path traversal
    try:
        resolved_path = ingest_data_path.resolve()
        if not resolved_path.is_dir():
            logger.warning(f"ingest-data path is not a directory: {resolved_path}, skipping mount")
        elif not str(resolved_path).startswith(str(Path(__file__).parent.resolve())):
            logger.warning(f"ingest-data path is outside project root: {resolved_path}, skipping mount")
        else:
            logger.info(f"Mounting static files from: {resolved_path}")
            app.mount("/ingest-data", StaticFiles(directory=str(resolved_path)), name="ingest-data")
    except Exception as e:
        logger.warning(f"Error validating ingest-data path: {e}, skipping mount")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Middleware
# =============================================================================

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Simple API key authentication middleware."""
    # Exempt paths that don't need authentication
    exempt_paths = ["/", "/health", "/docs", "/openapi.json", "/redoc", "/debug/features", "/debug/postgres"]
    if request.url.path in exempt_paths:
        return await call_next(request)

    # Check for API key
    api_key = os.environ.get("API_KEY")
    if api_key:
        provided_key = request.headers.get("X-API-Key")
        if provided_key != api_key:
            body = ApiEnvelope(
                status_code=401,
                message="Unauthorized",
                error="Unauthorized: Invalid or missing X-API-Key header",
                data=None,
            ).model_dump()
            return JSONResponse(status_code=401, content=body)

    return await call_next(request)


# =============================================================================
# Include Routers
# =============================================================================

app.include_router(debug_router)
app.include_router(trees_router)


@app.exception_handler(HTTPException)
async def _http_exception_envelope(request: Request, exc: HTTPException) -> JSONResponse:
    """Wrap ``HTTPException`` payloads in :class:`ApiEnvelope`."""
    detail = exc.detail
    if isinstance(detail, str):
        message = detail
        error: Union[str, Dict[str, Any], List[Any]] = detail
    elif isinstance(detail, list):
        message = "Request validation error"
        error = {"errors": detail}
    else:
        message = str(detail)
        error = detail if isinstance(detail, dict) else {"detail": detail}
    return _envelope_json_response(exc.status_code, message, error=error, data=None)


@app.exception_handler(RequestValidationError)
async def _validation_exception_envelope(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Wrap FastAPI request validation errors in :class:`ApiEnvelope` (HTTP 422)."""
    # exc.errors() returns list of ErrorWrapper objects — convert to plain dicts for JSON serialization
    errors = [
        {
            "type": e.get("type"),
            "loc": e.get("loc"),
            "msg": e.get("msg"),
            "input": e.get("input"),
        }
        for e in exc.errors()
    ]
    return _envelope_json_response(
        422,
        "Validation failed",
        error={"errors": errors},
        data=None,
    )


# =============================================================================
# Form-Data Models
# =============================================================================
# Reusable sub-models for time_series and metadata fields.


class TimeSeriesForm(BaseModel):
    """Form-data representation of time_series."""

    latitude: float = Field(..., description="GPS latitude coordinate")
    longitude: float = Field(..., description="GPS longitude coordinate")
    timestamp: int = Field(..., description="Unix epoch milliseconds")
    heading: float = Field(..., description="Compass heading in degrees")
    pitch: float = Field(..., description="Camera pitch in degrees")
    roll: float = Field(..., description="Camera roll in degrees")


class MetadataForm(BaseModel):
    """Form-data representation of metadata (device signature)."""

    deviceId: str = Field(..., description="Device identifier")
    nonce: str = Field(..., description="Nonce for signature verification")
    signature: str = Field(..., description="Base64-encoded cryptographic signature")


class IngestPayloadForm(BaseModel):
    """Single JSON object for form field ``payload`` on ingest routes (plus file upload)."""

    image_id: Optional[str] = Field(None, description="Unique image identifier")
    tree_id: str = Field(..., description="Tree identifier")
    time_series: TimeSeriesForm = Field(..., description="GPS, timestamp, orientation")
    metadata: Optional[MetadataForm] = Field(None, description="Device signature (optional)")


class VerifyPayloadForm(BaseModel):
    """Single JSON object for form field ``payload`` on verify routes (plus file upload)."""

    time_series: TimeSeriesForm = Field(..., description="GPS, timestamp, orientation")
    metadata: Optional[MetadataForm] = Field(None, description="Device signature (optional)")


# =============================================================================
# Standard API envelope & verification DTOs
# =============================================================================


class VerifyMultipartFormSchema(BaseModel):
    """Logical multipart fields for ``POST /verify`` (image file is separate)."""

    payload: Optional[str] = Field(None, description="JSON {time_series, metadata?}")
    time_series: Optional[str] = Field(None, description="JSON string when payload omitted")
    metadata: Optional[str] = Field(None, description="Optional JSON string")
    known_tree_id: Optional[str] = Field(None, description="Restrict search to one tree")
    radius: Optional[float] = Field(None, description="Geo radius override (metres)")


class VerifyRequestPayload(BaseModel):
    """Structured verify request after text fields are parsed (binary image is separate)."""

    time_series: TimeSeriesForm
    metadata: Optional[MetadataForm] = None
    known_tree_id: Optional[str] = None
    radius: Optional[float] = None


class MatchTimingPayload(BaseModel):
    """Timing breakdown from hierarchical matching."""

    model_config = ConfigDict(extra="forbid")
    dino_search: float = 0.0
    superpoint_match: float = 0.0
    geometric_verify: float = 0.0
    texture_match: float = 0.0
    total: float = 0.0


class BestMatchPayload(BaseModel):
    """Best candidate row; extra keys from ``MatchingResult.to_dict()`` are allowed."""

    model_config = ConfigDict(extra="allow")
    query_id: Optional[str] = None
    candidate_id: Optional[str] = None
    tree_id: Optional[str] = None
    dino_similarity: Optional[float] = None
    superpoint_matches: Optional[int] = None
    superpoint_inliers: Optional[int] = None
    superpoint_match_ratio: Optional[float] = None
    ransac_inlier_ratio: Optional[float] = None
    bark_texture_similarity: Optional[float] = None
    homography: Optional[Any] = None
    fundamental: Optional[Any] = None
    reprojection_error: Optional[float] = None
    final_score: Optional[float] = None
    timing: Optional[MatchTimingPayload] = None
    captured_at: Optional[int] = None
    confidence: Optional[float] = None
    score: Optional[float] = None


class VerifyResultData(BaseModel):
    """``data`` object for verify responses (backward-compatible fields)."""

    model_config = ConfigDict(extra="forbid")
    status: str = Field(
        ...,
        description="Semantic outcome: matched | no_match | probable_match | possible_match | error",
    )
    decision: str
    confidence: float = 0.0
    best_match: Optional[Dict[str, Any]] = None
    matched_tree_id: Optional[str] = None
    reason: str = ""
    source: Optional[str] = Field(
        default=None,
        description="Set to 'transparent' for verify-transparent only",
    )


def _verify_data_status(decision: str, matched_tree_id: Optional[str]) -> str:
    """Map service ``decision`` + ``matched_tree_id`` to ``data.status`` (never 'unknown')."""
    d = str(decision or "").strip().upper()
    if d == "ERROR":
        return "error"
    if d == "MATCH" and matched_tree_id not in (None, ""):
        return "matched"
    if d == "NO_MATCH":
        return "no_match"
    if d == "PROBABLE_MATCH":
        return "probable_match"
    if d == "POSSIBLE_MATCH":
        return "possible_match"
    if matched_tree_id is not None:
        return "no_match"
    return "no_match"


def _normalize_best_match(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not raw or not isinstance(raw, dict):
        return None
    try:
        return BestMatchPayload.model_validate(raw).model_dump(mode="json")
    except Exception:
        return convert_numpy_types(raw)


def _verify_result_from_service(
    result: Dict[str, Any],
    *,
    source: Optional[str] = None,
) -> VerifyResultData:
    """Build :class:`VerifyResultData` from hierarchical matcher output."""
    result = convert_numpy_types(result)
    decision = str(result.get("decision", "NO_MATCH"))
    reason = str(result.get("reason", ""))

    best_match_raw = result.get("best_match")
    if best_match_raw and isinstance(best_match_raw, dict):
        formatted_best_match = _normalize_best_match(best_match_raw)
        confidence = float(best_match_raw.get("confidence", 0.0))
    else:
        formatted_best_match = None
        confidence = 0.0

    matched_tree_id: Optional[str] = None
    if formatted_best_match and isinstance(formatted_best_match, dict):
        tid = formatted_best_match.get("tree_id")
        matched_tree_id = str(tid) if tid is not None else None

    data_status = _verify_data_status(decision, matched_tree_id)

    return VerifyResultData(
        status=data_status,
        decision=decision,
        confidence=confidence,
        best_match=formatted_best_match,
        matched_tree_id=matched_tree_id,
        reason=reason,
        source=source,
    )


# =============================================================================
# Helpers for form-data parsing
# =============================================================================


def _parse_form_field(model_cls: type[BaseModel], raw_value: str) -> BaseModel:
    """Parse a JSON string field (sent as multipart form text) into a Pydantic model."""
    try:
        return model_cls.model_validate_json(raw_value)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid JSON in field: {e}",
        )


def _build_gps_angle_from_timeseries(ts: TimeSeriesForm) -> dict:
    """Convert TimeSeriesForm (heading/pitch/roll) to the gps_angle dict expected by services."""
    # heading → hor_angle, pitch stays as pitch, roll → ver_angle (standard naming)
    return {
        "latitude": ts.latitude,
        "longitude": ts.longitude,
        "hor_angle": ts.heading,
        "pitch": ts.pitch,
        "ver_angle": ts.roll,
    }


def _resolve_ingest_form_models(
    payload: Optional[str],
    image_id: Optional[str],
    tree_id: Optional[str],
    time_series: Optional[str],
    metadata: Optional[str],
) -> tuple[str, Optional[str], TimeSeriesForm, Optional[MetadataForm]]:
    """Resolve ingest fields from either a single ``payload`` JSON or discrete form fields."""
    if payload and payload.strip():
        body = IngestPayloadForm.model_validate_json(payload)
        return body.tree_id, body.image_id, body.time_series, body.metadata
    if not tree_id or not time_series:
        raise HTTPException(
            status_code=422,
            detail=(
                "Provide multipart field `payload` (JSON with image_id, tree_id, time_series, metadata) "
                "or the fields `tree_id` and `time_series` (time_series as JSON string)."
            ),
        )
    ts = _parse_form_field(TimeSeriesForm, time_series)
    meta: Optional[MetadataForm] = None
    if metadata:
        meta = _parse_form_field(MetadataForm, metadata)
    return tree_id, image_id, ts, meta


def _resolve_verify_form_models(
    payload: Optional[str],
    time_series: Optional[str],
    metadata: Optional[str],
) -> tuple[TimeSeriesForm, Optional[MetadataForm]]:
    """Resolve verify fields from either ``payload`` JSON or discrete ``time_series`` / ``metadata``."""
    if payload and payload.strip():
        body = VerifyPayloadForm.model_validate_json(payload)
        return body.time_series, body.metadata
    if not time_series:
        raise HTTPException(
            status_code=422,
            detail=(
                "Provide multipart field `payload` (JSON with time_series, metadata) "
                "or the field `time_series` (JSON string)."
            ),
        )
    ts = _parse_form_field(TimeSeriesForm, time_series)
    meta: Optional[MetadataForm] = None
    if metadata:
        meta = _parse_form_field(MetadataForm, metadata)
    return ts, meta


# =============================================================================
# API Routes
# =============================================================================

@app.get("/")
async def root():
    """API information."""
    return _envelope_json_response(
        200,
        "OK",
        error=None,
        data={
            "name": "SAM3 Tree Identification API",
            "version": "1.0.0",
            "endpoints": [
                {"path": "/", "method": "GET", "description": "API info"},
                {"path": "/health", "method": "GET", "description": "Health check"},
                {"path": "/ingest", "method": "POST", "description": "Ingest tree (multipart form-data)"},
                {"path": "/ingest-transparent", "method": "POST", "description": "Ingest pre-segmented image (multipart form-data)"},
                {"path": "/verify", "method": "POST", "description": "Verify tree (multipart form-data)"},
                {"path": "/verify-transparent", "method": "POST", "description": "Verify pre-segmented image (multipart form-data)"},
                {"path": "/debug/features", "method": "GET", "description": "Debug stored features"},
                {"path": "/debug/postgres", "method": "GET", "description": "Debug PostgreSQL"},
                {"path": "/cleanup-gpu", "method": "POST", "description": "Release GPU memory"},
            ],
        },
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return _envelope_json_response(
        200,
        "All systems operational",
        error=None,
        data={"status": "healthy", "message": "All systems operational"},
    )


# ============================================================================
# INGEST ENDPOINTS
# ============================================================================
# Flow to PostgreSQL:
# 1. POST /ingest -> ingestion_service.ingest_raw()
#    -> IngestionService.ingest_raw()
#    -> IngestionService.ingest()
#    -> STEP 2: DINO global feature extraction
#    -> STEP 3: SuperPoint local feature extraction
#    -> STEP 4: Store local features in MinIO
#    -> STEP 5: Store global features in PostgreSQL
# ============================================================================

# ============================================================================
# INGEST ENDPOINT — multipart/form-data
# ============================================================================


@app.post("/ingest")
async def ingest(
    request: Request,
    # Either send ``payload`` (one JSON object) or discrete fields below.
    payload: Optional[str] = Form(
        None,
        description=(
            "Optional: JSON object {image_id?, tree_id, time_series, metadata?} "
            "(same logical shape as your API contract; nested objects as JSON, not strings)"
        ),
    ),
    image_id: Optional[str] = Form(None, description="Unique image identifier (auto-generated if omitted)"),
    tree_id: Optional[str] = Form(None, description="Tree identifier (required if payload omitted)"),
    time_series: Optional[str] = Form(
        None,
        description="JSON string of TimeSeries (required if payload omitted)",
    ),
    metadata: Optional[str] = Form(None, description="JSON string of Metadata (optional)"),
):
    """Ingest a tree image via multipart/form-data.

    **Multipart form-data (not query params).** Nested objects are either:

    - **Option A — single field:** ``payload`` = JSON
      ``{"image_id","tree_id","time_series":{...},"metadata":{...}}``
    - **Option B — flat fields:** ``tree_id``, ``time_series`` (JSON string), optional ``metadata`` (JSON string), optional ``image_id``

    Plus an **image file** on any other form field name (binary upload).

    ``time_series`` object keys: latitude, longitude, timestamp (epoch ms), heading, pitch, roll.
    ``metadata`` object keys: deviceId, nonce, signature.
    """
    tree_id_resolved, image_id_resolved, ts, _meta = _resolve_ingest_form_models(
        payload, image_id, tree_id, time_series, metadata
    )
    gps_angle = _build_gps_angle_from_timeseries(ts)
    # timestamp from time_series is Unix epoch milliseconds
    captured_at_ms = ts.timestamp

    # ── Extract image bytes from multipart form ──────────────────────────
    form = await request.form()
    image_bytes: Optional[bytes] = None
    resolved_image_id = image_id_resolved

    for field_name, field_value in form.items():
        # Skip the text fields we already parsed
        if field_name in ("payload", "image_id", "tree_id", "time_series", "metadata"):
            continue
        if hasattr(field_value, "read"):
            image_bytes = await field_value.read()
            if not resolved_image_id:
                resolved_image_id = field_name
            break

    if image_bytes is None:
        raise HTTPException(
            status_code=400,
            detail="No image file found in form data. Upload an image file as a binary form field.",
        )

    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 20MB)")

    # Decode image
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. Send a valid JPEG or PNG.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    # Auto-generate image_id if not provided
    if not resolved_image_id:
        import uuid
        resolved_image_id = f"auto-{uuid.uuid4().hex[:8]}"

    try:
        import asyncio

        _ingestion_service = ingestion_service()
        if _ingestion_service is None:
            raise HTTPException(status_code=503, detail="Ingestion service not available")

        _ingest_semaphore = get_ingest_semaphore()
        _ingest_executor = get_ingest_executor()
        if not _ingest_semaphore or not _ingest_executor:
            raise HTTPException(status_code=503, detail="Ingest executor not initialized")

        loop = asyncio.get_running_loop()
        async with _ingest_semaphore:
            result = await loop.run_in_executor(
                _ingest_executor,
                lambda: _ingestion_service.ingest_raw(
                    image=img,
                    image_id=resolved_image_id,
                    tree_id=tree_id_resolved,
                    gps_angle=gps_angle,
                    captured_at=captured_at_ms,
                ),
            )

        msg = result.message if hasattr(result, "message") else "OK"
        return _envelope_json_response(
            200,
            msg,
            error=None,
            data={
                "success": True,
                "imageId": resolved_image_id,
                "treeId": tree_id_resolved,
                "features_extracted": {
                    "global_dim": result.feature_info.get("global_dim", 0),
                    "local_keypoints": result.feature_info.get("n_keypoints", 0),
                    "local_dim": result.feature_info.get("descriptor_dim", 0),
                },
                "storage_keys": result.storage_keys if hasattr(result, "storage_keys") else {},
                "message": msg,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ============================================================================
# VERIFY ENDPOINT — multipart/form-data
# ============================================================================


@app.post("/verify")
@inject
async def verify(
    request: Request,
    verification_service: VerificationService = Depends(Provide[Container.verification_service]),
    payload: Optional[str] = Form(
        None,
        description="Optional: JSON object {time_series, metadata?} (nested objects as JSON)",
    ),
    known_tree_id: Optional[str] = Form(None, description="Optional: restrict search to specific tree"),
    radius: Optional[float] = Form(None, description="Search radius in meters (overrides config)"),
    time_series: Optional[str] = Form(None, description="JSON string of TimeSeries (required if payload omitted)"),
    metadata: Optional[str] = Form(None, description="JSON string of Metadata (optional)"),
):
    """Verify tree identity via multipart/form-data.

    **Option A:** ``payload`` = JSON ``{"time_series":{...},"metadata":{...}}``

    **Option B:** ``time_series`` (JSON string), optional ``metadata`` (JSON string).

    Optional: ``known_tree_id``, ``radius``. Plus an **image file** on any other form field.

    Response body is always an :class:`ApiEnvelope`; verification fields live under ``data``.
    """
    # Reference schema for OpenAPI (multipart file is not part of this model).
    _ = VerifyMultipartFormSchema(
        payload=payload,
        time_series=time_series,
        metadata=metadata,
        known_tree_id=known_tree_id,
        radius=radius,
    )

    try:
        ts, _meta = _resolve_verify_form_models(payload, time_series, metadata)
        _ = VerifyRequestPayload(
            time_series=ts,
            metadata=_meta,
            known_tree_id=known_tree_id,
            radius=radius,
        )
        lat, lon = ts.latitude, ts.longitude
        heading, pitch, roll = ts.heading, ts.pitch, ts.roll
        query_timestamp_ms = ts.timestamp

        form = await request.form()
        image_bytes: Optional[bytes] = None
        for field_name, field_value in form.items():
            if field_name in ("payload", "known_tree_id", "radius", "time_series", "metadata"):
                continue
            if hasattr(field_value, "read"):
                image_bytes = await field_value.read()
                break

        if image_bytes is None:
            raise HTTPException(status_code=400, detail="No image file found in form data.")

        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large (max 20MB)")

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            if nparr.size == 0:
                raise HTTPException(status_code=400, detail="Invalid image data")
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Could not decode image.")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image data")

        mask = None
        app_cfg = get_config()
        effective_radius = radius if radius is not None else app_cfg.geo_radius_meters

        geo_filter = {
            "radius_meters": effective_radius,
            "latitude": lat,
            "longitude": lon,
        }
        angle_filter = {
            "hor_angle_min": heading - app_cfg.hor_angle_range,
            "hor_angle_max": heading + app_cfg.hor_angle_range,
            "ver_angle_min": roll - app_cfg.ver_angle_range,
            "ver_angle_max": roll + app_cfg.ver_angle_range,
            "pitch_min": (pitch - app_cfg.pitch_range) if pitch is not None else None,
            "pitch_max": (pitch + app_cfg.pitch_range) if pitch is not None else None,
        }

        query_timestamp_sec = query_timestamp_ms / 1000.0
        time_filter = create_time_filter(query_timestamp_sec)

        result = await verification_service.verify(
            image=img,
            mask=mask,
            known_tree_id=known_tree_id,
            geo_filter=geo_filter,
            angle_filter=angle_filter,
            time_filter=time_filter,
        )

        if not isinstance(result, dict):
            raise TypeError(f"verification_service.verify returned {type(result)!r}, expected dict")

        result = convert_numpy_types(result)
        verify_data = _verify_result_from_service(result, source=None)

        return _envelope_json_response(
            200,
            "Verification complete",
            error=None,
            data=verify_data.model_dump(mode="json"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return _envelope_json_response(
            500,
            "Verification failed",
            error={"type": type(e).__name__, "detail": str(e)},
            data=None,
        )


def _mask_from_checkerboard_jpeg(bgr: np.ndarray) -> np.ndarray:
    """Infer a foreground mask from a JPEG whose background is a checkerboard pattern.

    Image editors render transparent areas as alternating squares of two distinct
    colours when saving without an alpha channel.  Empirically those squares are:
      • near-white  (all channels > 230) — JPEG-compressed white tiles
      • near-black  (all channels < 15)  — JPEG-compressed black tiles

    Algorithm
    ---------
    1. Colour threshold: mark near-white OR near-black pixels as background.
    2. Morphological open (erode → dilate): removes stray single-pixel noise
       from both the tree body and the background.
    3. Morphological close: bridge checkerboard holes *inside* the tree body
       (kernel large enough to span one checkerboard tile, ~16–20 px).
    4. Keep only the largest connected foreground component — this is the tree.
       Any stray foreground fragments that leaked into corner background areas
       during closing are removed automatically.
    5. Guard: if the entire image is classified as background, raise HTTPException 422.

    Returns
    -------
    mask : np.ndarray, uint8, shape (H, W)
        255 for subject (tree) pixels, 0 for background.

    Raises
    ------
    HTTPException 422 if no foreground pixels are detected.
    """
    # ── Step 1: Colour-based background detection ─────────────────────────────
    # Use int16 to avoid uint8 wrap-around during comparison.
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)

    is_white_bg = (b > 230) & (g > 230) & (r > 230)
    is_black_bg = (b < 15)  & (g < 15)  & (r < 15)
    is_bg = is_white_bg | is_black_bg

    foreground = (~is_bg).astype(np.uint8) * 255

    # ── Step 2: Morphological open — remove stray noise before closing ────────
    # A 5×5 kernel is sufficient to remove isolated single-tile noise without
    # significantly shrinking the tree body.  Running open BEFORE close ensures
    # that the subsequent close operation targets the real tree boundary.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel_open)

    # ── Step 3: Morphological close — fill checkerboard tiles inside the tree ─
    # A kernel of 21×21 spans one checkerboard tile (~16 px) with some margin.
    # Two iterations bridge tiles that are separated by one intervening tile.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # ── Step 4: Keep the single largest connected component ───────────────────
    # After closing, any background corner pixels that were accidentally pulled
    # into the mask form *small* isolated islands far from the tree body.
    # Keeping only the largest component (the tree) removes them.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    if num_labels <= 1:
        # num_labels == 1 means only the background label exists —
        # the entire image was classified as background (e.g. near-white bark
        # or near-black shadows on the tree).  Raise instead of silently returning
        # a zero mask that would produce a meaningless DINO embedding.
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not detect tree foreground from checkerboard background. "
                "Image may not have a checkerboard-pattern background or tree pixels "
                "are all near-white / near-black."
            ),
        )
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    foreground = (labels == largest_label).astype(np.uint8) * 255

    return foreground


def _decode_transparent_png(image_bytes: bytes) -> tuple:
    """Decode a pre-background-removed image into (bgr_image, binary_mask).

    Supports three input types:

    1. **PNG / WebP with alpha channel (BGRA)** – preferred.
       The alpha channel is used directly as the segmentation mask.
       alpha > 0  → foreground (255)
       alpha == 0 → background (0)

    2. **JPEG with checkerboard background** – common export from background-
       removal tools that cannot save transparency.
       The alternating near-black / near-white checkerboard squares are detected
       automatically via colour thresholds and the tree pixels are kept.

    3. **Any other 3-channel image** – treated as full-foreground (pass-through).

    IMPORTANT: The returned ``bgr`` contains ORIGINAL pixel values (including any
    checkerboard background pixels in JPEG path).  Background pixels are NOT
    replaced with black.  Downstream ``ingest()`` applies the mask internally,
    matching the SAM3 path behaviour.

    Args:
        image_bytes: Raw bytes of the image file.

    Returns:
        (bgr, mask_gray): bgr is (H, W, 3) uint8 BGR with original pixel values;
        mask_gray is (H, W) uint8 with 255 for foreground pixels and 0 for
        background pixels.

    Raises:
        HTTPException 400 if the bytes cannot be decoded as an image.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    if nparr.size == 0:
        raise HTTPException(status_code=400, detail="Empty image data")

    # IMREAD_UNCHANGED preserves the alpha channel (4-channel BGRA for PNGs/WebP)
    img_full = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img_full is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not decode image. "
                "Send a PNG/WebP with transparency (RGBA) or a JPEG whose background "
                "was exported as a checkerboard pattern."
            ),
        )

    if img_full.ndim == 2:
        # Grayscale — convert to BGR, full foreground mask
        bgr = cv2.cvtColor(img_full, cv2.COLOR_GRAY2BGR)
        mask = np.full(img_full.shape, 255, dtype=np.uint8)

    elif img_full.shape[2] == 4:
        # BGRA — use alpha channel directly as mask (best quality)
        bgr = img_full[:, :, :3].copy()
        alpha = img_full[:, :, 3]
        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)

    else:
        # 3-channel: no alpha available → infer mask from checkerboard pattern.
        # This handles JPEGs exported from background-removal tools.
        # _mask_from_checkerboard_jpeg now handles the full open→close pipeline
        # and raises HTTPException 422 when no foreground is detected.
        bgr = img_full.copy()
        mask = _mask_from_checkerboard_jpeg(bgr)

    # ── Return original BGR + mask (NOT pre-blackened) ─────────────────────
    # CRITICAL: do NOT replace background pixels with black before passing to
    # ingest().  The downstream ingest() pipeline (apply_mask → crop → prepare_
    # for_dino) must operate on the ORIGINAL pixel values, just like the SAM3
    # path does.  Pre-blackening corrupts the bounding-box calculation and the
    # pixel values that DINO sees, producing a different embedding vector.
    #
    # SAM3 path:         original_jpeg → apply_mask(original, sam3_mask) → crop → DINO
    # Transparent path:   original_bgr  → apply_mask(original, mask)       → crop → DINO
    # Both now produce identical downstream input.
    return bgr, mask


# =============================================================================
# /ingest-transparent
# =============================================================================

# =============================================================================
# INGEST-TRANSPARENT ENDPOINT — multipart/form-data
# =============================================================================

@app.post("/ingest-transparent")
async def ingest_transparent(
    request: Request,
    payload: Optional[str] = Form(
        None,
        description="Optional: JSON object {image_id?, tree_id, time_series, metadata?}",
    ),
    image_id: Optional[str] = Form(None, description="Unique image identifier (auto-generated if omitted)"),
    tree_id: Optional[str] = Form(None, description="Tree identifier (required if payload omitted)"),
    time_series: Optional[str] = Form(None, description="JSON string of TimeSeries (required if payload omitted)"),
    metadata: Optional[str] = Form(None, description="JSON string of Metadata (optional)"),
):
    """Ingest a pre-segmented (transparent-background) tree image via multipart/form-data.

    Uses ``_decode_transparent_png`` to extract the foreground mask directly from the
    uploaded image — **no SAM3 segmentation is run**.

    Supported input formats:
      1. **PNG / WebP with alpha channel (BGRA)** – alpha > 0 → foreground, alpha == 0 → background.
      2. **JPEG with checkerboard background** – alternating near-black / near-white tiles
         are detected and the tree mask is reconstructed via morphological operations.
      3. **Any other 3-channel image** – treated as full-foreground (pass-through).

    The returned BGR preserves original pixel values; downstream ``ingest()`` applies
    the mask internally, producing feature vectors identical to the SAM3-segmented path.

    Same form contract as ``/ingest``: optional ``payload`` JSON or flat ``tree_id`` +
    ``time_series`` (+ optional ``metadata``, ``image_id``).
    """
    tree_id_resolved, image_id_resolved, ts, _meta = _resolve_ingest_form_models(
        payload, image_id, tree_id, time_series, metadata
    )
    gps_angle = _build_gps_angle_from_timeseries(ts)
    captured_at_ms = ts.timestamp

    # ── Extract image bytes from multipart form ──────────────────────────
    form = await request.form()
    image_bytes: Optional[bytes] = None
    resolved_image_id = image_id_resolved

    for field_name, field_value in form.items():
        if field_name in ("payload", "image_id", "tree_id", "time_series", "metadata"):
            continue
        if hasattr(field_value, "read"):
            image_bytes = await field_value.read()
            if not resolved_image_id:
                resolved_image_id = field_name
            break

    if image_bytes is None:
        raise HTTPException(status_code=400, detail="No image file found in form data.")

    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 20MB)")

    # ── Decode transparent PNG → BGR + mask ──────────────────────────────
    bgr, mask = _decode_transparent_png(image_bytes)

    # Guard: empty mask (all-alpha transparent image or all-background JPEG)
    if not np.any(mask):
        raise HTTPException(
            status_code=422,
            detail="Decoded mask is empty — no foreground pixels found in image",
        )

    if not resolved_image_id:
        import uuid as _uuid
        resolved_image_id = f"auto-{_uuid.uuid4().hex[:8]}"

    # Build metadata dict for ingestion service (includes captured_at unlike /ingest which passes it separately)
    final_metadata = {**gps_angle, "captured_at": captured_at_ms}

    try:
        import asyncio

        _ingestion_service = ingestion_service()
        if _ingestion_service is None:
            raise HTTPException(status_code=503, detail="Ingestion service not available")

        _ingest_semaphore = get_ingest_semaphore()
        _ingest_executor = get_ingest_executor()
        if not _ingest_semaphore or not _ingest_executor:
            raise HTTPException(status_code=503, detail="Ingest executor not initialized")

        loop = asyncio.get_running_loop()
        async with _ingest_semaphore:
            # Call ingest() DIRECTLY — bypasses SAM3 since mask is already available
            result = await loop.run_in_executor(
                _ingest_executor,
                lambda: _ingestion_service.ingest(
                    image=bgr,
                    mask=mask,
                    image_id=resolved_image_id,
                    tree_id=tree_id_resolved,
                    metadata=final_metadata,
                ),
            )

        msg = result.message if hasattr(result, "message") else "OK"
        return _envelope_json_response(
            200,
            msg,
            error=None,
            data={
                "success": True,
                "imageId": resolved_image_id,
                "treeId": tree_id_resolved,
                "source": "transparent",
                "features_extracted": {
                    "global_dim": result.feature_info.get("global_dim", 0),
                    "local_keypoints": result.feature_info.get("n_keypoints", 0),
                    "local_dim": result.feature_info.get("descriptor_dim", 0),
                },
                "storage_keys": result.storage_keys if hasattr(result, "storage_keys") else {},
                "message": msg,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/ingest-transparent] error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# =============================================================================
# VERIFY-TRANSPARENT ENDPOINT — multipart/form-data
# =============================================================================

@app.post("/verify-transparent")
@inject
async def verify_transparent(
    request: Request,
    verification_service: VerificationService = Depends(Provide[Container.verification_service]),
    payload: Optional[str] = Form(
        None,
        description="Optional: JSON object {time_series, metadata?}",
    ),
    known_tree_id: Optional[str] = Form(None, description="Optional: restrict search to specific tree"),
    radius: Optional[float] = Form(None, description="Search radius in metres (overrides config)"),
    time_series: Optional[str] = Form(None, description="JSON string of TimeSeries (required if payload omitted)"),
    metadata: Optional[str] = Form(None, description="JSON string of Metadata (optional)"),
):
    """Verify a pre-segmented (transparent-background) tree image via multipart/form-data.

    Uses ``_decode_transparent_png`` to extract the foreground mask directly from the
    uploaded image — **no SAM3 segmentation is run**.

    Supported input formats:
      1. **PNG / WebP with alpha channel (BGRA)** – alpha > 0 → foreground, alpha == 0 → background.
      2. **JPEG with checkerboard background** – alternating near-black / near-white tiles
         are detected and the tree mask is reconstructed via morphological operations.
      3. **Any other 3-channel image** – treated as full-foreground (pass-through).

    The returned BGR preserves original pixel values; downstream ``verify()`` applies
    the mask internally, producing feature vectors identical to the SAM3-segmented path.

    Same form contract as ``/verify``: optional ``payload`` JSON or ``time_series``
    (+ optional ``metadata``), plus image file.

    Response body is always an :class:`ApiEnvelope`; verification fields live under ``data``.
    """
    _ = VerifyMultipartFormSchema(
        payload=payload,
        time_series=time_series,
        metadata=metadata,
        known_tree_id=known_tree_id,
        radius=radius,
    )

    try:
        ts, _meta = _resolve_verify_form_models(payload, time_series, metadata)
        _ = VerifyRequestPayload(
            time_series=ts,
            metadata=_meta,
            known_tree_id=known_tree_id,
            radius=radius,
        )
        lat, lon = ts.latitude, ts.longitude
        heading, pitch_val, roll = ts.heading, ts.pitch, ts.roll
        query_timestamp_ms = ts.timestamp

        form = await request.form()
        image_bytes: Optional[bytes] = None
        for field_name, field_value in form.items():
            if field_name in ("payload", "known_tree_id", "radius", "time_series", "metadata"):
                continue
            if hasattr(field_value, "read"):
                image_bytes = await field_value.read()
                break

        if image_bytes is None:
            raise HTTPException(status_code=400, detail="No image file found in form data.")

        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large (max 20MB)")

        bgr, mask = _decode_transparent_png(image_bytes)

        if not np.any(mask):
            raise HTTPException(
                status_code=422,
                detail="Decoded mask is empty — no foreground pixels found in image",
            )

        app_cfg = get_config()
        effective_radius = radius if radius is not None else app_cfg.geo_radius_meters

        geo_filter = {
            "radius_meters": effective_radius,
            "latitude": lat,
            "longitude": lon,
        }
        angle_filter = {
            "hor_angle_min": heading - app_cfg.hor_angle_range,
            "hor_angle_max": heading + app_cfg.hor_angle_range,
            "ver_angle_min": roll - app_cfg.ver_angle_range,
            "ver_angle_max": roll + app_cfg.ver_angle_range,
            "pitch_min": (pitch_val - app_cfg.pitch_range) if pitch_val is not None else None,
            "pitch_max": (pitch_val + app_cfg.pitch_range) if pitch_val is not None else None,
        }
        query_timestamp_sec = query_timestamp_ms / 1000.0
        time_filter = create_time_filter(query_timestamp_sec)

        result = await verification_service.verify(
            image=bgr,
            mask=mask,
            known_tree_id=known_tree_id,
            geo_filter=geo_filter,
            angle_filter=angle_filter,
            time_filter=time_filter,
        )

        if not isinstance(result, dict):
            raise TypeError(f"verification_service.verify returned {type(result)!r}, expected dict")

        result = convert_numpy_types(result)
        verify_data = _verify_result_from_service(result, source="transparent")

        return _envelope_json_response(
            200,
            "Verification complete",
            error=None,
            data=verify_data.model_dump(mode="json"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/verify-transparent] error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return _envelope_json_response(
            500,
            "Verification failed",
            error={"type": type(e).__name__, "detail": str(e)},
            data=None,
        )


@app.post("/cleanup-gpu")
async def cleanup_gpu():
    """Release GPU memory."""
    cleanup_gpu_memory()
    return _envelope_json_response(
        200,
        "GPU memory released",
        error=None,
        data={"success": True, "message": "GPU memory released"},
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Verbose DEBUG logging when run directly
    setup_logging(level="DEBUG")

    # Auto reload when Python files change
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=[str(Path(__file__).parent / "src")],
    )
