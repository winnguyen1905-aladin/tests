#!/usr/bin/env python3
"""
config.py - Unified Configuration & Logging for the SAM3 Project

Provides:
  - Centralised logging setup (call `setup_logging()` once at startup)
  - Pydantic-based Settings loaded from environment / .env file
  - AppConfig bridge for backward compatibility (via `init_config()`)

Environment variables (subset):
    ENV                 - dev | staging | prod  (default: dev)
    LOG_LEVEL           - DEBUG | INFO | WARNING | ERROR  (default: INFO)
    LOG_FILE            - Optional path to a rotating log file
    MILVUS_URI          - Milvus connection URI
    MINIO_ENDPOINT      - MinIO endpoint
    MINIO_ACCESS_KEY    - MinIO access key
    MINIO_SECRET_KEY    - MinIO secret key
    SAM3_MODEL_PATH     - Path to SAM3 model weights
    SAM3_DEVICE         - Device for SAM3 (cuda/cpu)
    DINO_DEVICE         - Device for DINO (cuda/cpu)
    DINO_HF_TOKEN       - HuggingFace token for gated DINOv3 (or use HF_TOKEN)
    MODEL_MODE          - ultra | lite
    INGEST_CONCURRENCY  - Max concurrent ingest requests
    CORS_ORIGINS        - Comma-separated list of allowed origins
    API_VERSION         - API version prefix
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# AppConfig Lazy Import (to avoid torch dependency at module load time)
# ---------------------------------------------------------------------------

def _get_app_config_helper():
    """Lazy import of appConfig to avoid torch dependency at module load time."""
    from .appConfig import get_config as _get_app_config
    return _get_app_config


# Re-export get_config from appConfig for backward compatibility
def get_config():
    """Return the global AppConfig singleton.

    This function is re-exported from appConfig for backward compatibility.
    New code should prefer using Settings directly.
    """
    return _get_app_config_helper()()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_QUIET_LOGGERS = [
    "urllib3.connectionpool",
    "urllib3.util.retry",
    "transformers.image_processing_utils",
    "pymilvus",
    "grpc",
    "PIL",
    "matplotlib",
]

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-42s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_logging_configured = False  # guard against redundant calls


def setup_logging(
    level: str | None = None,
    log_file: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """Configure the root logger for the entire application.

    Call once at startup (e.g. inside the FastAPI lifespan or ``__main__``).
    All subsequent ``logging.getLogger(__name__)`` calls in ``src/`` modules
    will automatically inherit this configuration.

    Args:
        level:        Override LOG_LEVEL env var (e.g. ``"DEBUG"``).
                      Falls back to ``$LOG_LEVEL``, then ``"INFO"``.
        log_file:     Path to a rotating log file. Falls back to ``$LOG_FILE``.
                      When ``None``, logs go to *stderr* only.
        max_bytes:    Maximum size of a single log file before rotation.
        backup_count: Number of rotated backup files to keep.
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    # Resolve log level
    raw_level = level or os.environ.get("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, raw_level.upper(), logging.INFO)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler (stderr)
    handlers: list[logging.Handler] = []
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console.setLevel(numeric_level)
    handlers.append(console)

    # Optional rotating file handler
    resolved_log_file = log_file or os.environ.get("LOG_FILE")
    if resolved_log_file:
        log_path = Path(resolved_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)

    # Apply to root logger
    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()  # drop any handlers added by earlier basicConfig calls
    for h in handlers:
        root.addHandler(h)

    # Silence noisy third-party loggers
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    logging.getLogger(__name__).info(
        "Logging initialised — level=%s%s",
        raw_level.upper(),
        f", file={resolved_log_file}" if resolved_log_file else "",
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file.

    All fields can be overridden by the corresponding environment variable
    (case-insensitive). Unknown variables are silently ignored.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ─────────────────────────────────────────────────────────
    env: Literal["dev", "staging", "prod"] = "dev"

    # ── API ──────────────────────────────────────────────────────────────────
    api_version: str = "v1"
    api_title: str = "SAM3 Tree Identification API"
    cors_origins: str = "*"          # comma-separated for multiple origins
    cors_allow_credentials: bool = True

    # ── Milvus ───────────────────────────────────────────────────────────────
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "tree_features"
    milvus_vector_dim: int = 384  # dinov3-vitb16-pretrain-lvd1689m: ViT-Base 21M, 384 dims
    milvus_top_k: int = 100
    milvus_search_top_k: int = 100
    milvus_search_nprobe: int = 32
    milvus_timeout: int = 30         # seconds
    milvus_pool_size: int = 10

    # ── Vector store ─────────────────────────────────────────────────────────
    vector_store_type: Literal["milvus", "postgres"] = "postgres"

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "sam3"
    postgres_user: str = "sam3user"
    postgres_password: str = "sam3pass"
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20
    postgres_vector_dim: int = 384  # DINO model output dimension (384 dims)
    postgres_vector_index_type: str = "hnsw"
    postgres_vector_m: int = 16
    postgres_vector_ef_construction: int = 64

    # ── MinIO ────────────────────────────────────────────────────────────────
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "tree-features"
    minio_secure: bool = False
    minio_timeout: int = 30

    # ── SAM3 ─────────────────────────────────────────────────────────────────
    sam3_model_path: str = "sam3.pt"
    sam3_device: str = "cuda"
    sam3_confidence: float = 0.05
    sam3_image_size: int = 644

    # ── DINO ─────────────────────────────────────────────────────────────────
    dino_device: str = "cuda"
    # Use DINO_HF_TOKEN, or standard HF_TOKEN / HUGGING_FACE_HUB_TOKEN (see resolve helper).
    dino_hf_token: str = ""
    dino_model_type: str = "dinov3-vitb16-pretrain-lvd1689m"
    dino_image_size: int = 644

    # ── SuperPoint & LightGlue ───────────────────────────────────────────────
    sp_device: str = "cuda"
    sp_max_keypoints: int = 4096
    sp_max_dimension: int = 1280  # Max image pixel dimension for SuperPoint preprocessing
    lg_device: str = "cuda"
    lg_filter_threshold: float = 0.5
    lg_confidence: float = 0.1

    # ── DINO extras ──────────────────────────────────────────────────────────
    dino_use_multi_gpu: bool = False
    dino_gpu_ids: Optional[List[int]] = None
    dino_enable_memory_optimization: bool = True
    dino_use_gradient_checkpointing: bool = False
    dino_use_4bit_quantization: bool = False

    # ── Matching & filters ───────────────────────────────────────────────────
    coarse_threshold: float = 0.5
    inlier_threshold: int = 8
    min_matches: int = 15
    min_inliers: int = 7
    geo_radius_meters: float = 10.0
    hor_angle_range: float = 30.0
    ver_angle_range: float = 60.0
    pitch_range: float = 75.0

    # ── Concurrency ──────────────────────────────────────────────────────────
    ingest_concurrency: int = 4
    ingest_thread_workers: int = 1   # must be 1 for SAM3 (not thread-safe)

    # ── GPU / model mode ─────────────────────────────────────────────────────
    model_mode: Literal["ultra", "lite"] = "ultra"

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Feature flags ────────────────────────────────────────────────────────
    enable_graph_api: bool = True
    enable_batch_ingestion: bool = True

    # ── Helpers ──────────────────────────────────────────────────────────────
    @model_validator(mode="after")
    def _validate_vector_dims_match(self) -> "Settings":
        """Keep vector widths aligned across stores for consistent feature flow."""
        if int(self.milvus_vector_dim) != int(self.postgres_vector_dim):
            raise ValueError(
                "MILVUS_VECTOR_DIM must match POSTGRES_VECTOR_DIM "
                f"(got milvus_vector_dim={self.milvus_vector_dim}, "
                f"postgres_vector_dim={self.postgres_vector_dim})"
            )
        return self

    def get_cors_origins_list(self) -> List[str]:
        """Return CORS origins as a list."""
        if self.cors_origins == "*" or not self.cors_origins:
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",")]

    def is_production(self) -> bool:
        """Return ``True`` when running in production."""
        return self.env == "prod"

    def is_development(self) -> bool:
        """Return ``True`` when running in development."""
        return self.env == "dev"


def resolve_huggingface_token_for_dino(settings: Optional[Settings] = None) -> str:
    """Return the first non-empty HF token for DINO downloads.

    Order: ``DINO_HF_TOKEN`` (from Settings), ``HF_TOKEN``, ``HUGGING_FACE_HUB_TOKEN``.
    Values are stripped; empty strings are ignored.

    DINOv3 checkpoints are *gated*: the token must belong to an account that has
    accepted the model license on the Hugging Face model page.
    """
    s = settings if settings is not None else get_settings()
    for raw in (
        s.dino_hf_token,
        os.environ.get("HF_TOKEN"),
        os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    ):
        if raw is None:
            continue
        t = str(raw).strip()
        if t:
            return t
    return ""


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings singleton.

    Clear with ``get_settings.cache_clear()`` in tests.
    """
    return Settings()


def reload_settings() -> Settings:
    """Invalidate the settings cache and return a fresh instance."""
    get_settings.cache_clear()
    return get_settings()


# ---------------------------------------------------------------------------
# AppConfig bridge  (backward-compatible)
# ---------------------------------------------------------------------------

def create_app_config_from_settings(settings: Optional[Settings] = None):
    """Build an ``AppConfig`` instance from a ``Settings`` object.

    Bridges the new Pydantic-based ``Settings`` with the legacy ``AppConfig``
    class for backward compatibility. All env-backed config comes from Settings
    (no os.getenv in appConfig layer).

    Args:
        settings: Optional ``Settings`` instance; uses the cached singleton
                  when not provided.

    Returns:
        A fully configured ``AppConfig`` instance.

    Raises:
        ImportError: If ``appConfig`` module is not available.
    """
    # Lazy import to avoid torch dependency at module load time
    from .appConfig import AppConfig

    if settings is None:
        settings = get_settings()

    return AppConfig(
        # Paths (base_dir, data_dir keep AppConfig defaults)
        model_path=settings.sam3_model_path,
        # SAM3
        sam3_device=settings.sam3_device,
        sam3_confidence=settings.sam3_confidence,
        sam3_image_size=settings.sam3_image_size,
        # DINO
        dino_device=settings.dino_device,
        dino_hf_token=resolve_huggingface_token_for_dino(settings),
        dino_model_type=settings.dino_model_type,
        dino_image_size=settings.dino_image_size,
        dino_use_multi_gpu=settings.dino_use_multi_gpu,
        dino_gpu_ids=settings.dino_gpu_ids,
        dino_enable_memory_optimization=settings.dino_enable_memory_optimization,
        dino_use_gradient_checkpointing=settings.dino_use_gradient_checkpointing,
        dino_use_4bit_quantization=settings.dino_use_4bit_quantization,
        # SuperPoint
        sp_device=settings.sp_device,
        sp_max_keypoints=settings.sp_max_keypoints,
        sp_max_dimension=settings.sp_max_dimension,
        # LightGlue
        lg_device=settings.lg_device,
        lg_filter_threshold=settings.lg_filter_threshold,
        lg_confidence=settings.lg_confidence,
        # Model mode
        model_mode=settings.model_mode,
        # Vector store
        vector_store_type=settings.vector_store_type,
        # Milvus
        milvus_uri=settings.milvus_uri,
        milvus_collection=settings.milvus_collection,
        milvus_vector_dim=settings.milvus_vector_dim,
        milvus_top_k=settings.milvus_top_k,
        milvus_search_top_k=settings.milvus_search_top_k,
        milvus_search_nprobe=settings.milvus_search_nprobe,
        # PostgreSQL
        postgres_host=settings.postgres_host,
        postgres_port=settings.postgres_port,
        postgres_db=settings.postgres_db,
        postgres_user=settings.postgres_user,
        postgres_password=settings.postgres_password,
        postgres_pool_size=settings.postgres_pool_size,
        postgres_max_overflow=settings.postgres_max_overflow,
        postgres_vector_dim=settings.postgres_vector_dim,
        postgres_vector_index_type=settings.postgres_vector_index_type,
        postgres_vector_m=settings.postgres_vector_m,
        postgres_vector_ef_construction=settings.postgres_vector_ef_construction,
        # MinIO
        minio_endpoint=settings.minio_endpoint,
        minio_access_key=settings.minio_access_key,
        minio_secret_key=settings.minio_secret_key,
        minio_bucket=settings.minio_bucket,
        minio_secure=settings.minio_secure,
        # Matching & filters
        coarse_threshold=settings.coarse_threshold,
        inlier_threshold=settings.inlier_threshold,
        min_matches=settings.min_matches,
        min_inliers=settings.min_inliers,
        geo_radius_meters=settings.geo_radius_meters,
        hor_angle_range=settings.hor_angle_range,
        ver_angle_range=settings.ver_angle_range,
        pitch_range=settings.pitch_range,
        # Logging
        verbose=not settings.is_production(),
        log_level=settings.log_level,
    )


def init_config():
    """Initialise and return an ``AppConfig`` built from current ``Settings``.

    Recommended entry-point for application startup code that still relies on
    the legacy ``AppConfig`` interface.

    Returns:
        A configured ``AppConfig`` instance.
    """
    return create_app_config_from_settings(get_settings())
