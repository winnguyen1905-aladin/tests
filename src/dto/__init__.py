#!/usr/bin/env python3
"""
SAM3 Data Transfer Objects (DTOs).

Contains Pydantic models for API request/response handling.
"""

# Common DTOs
from .common import (
    HealthResponse,
    ErrorResponse,
    RootResponse,
    FeatureInfo,
    StorageKeys,
    IngestResultItem,
)

# Tree DTOs
try:
    from .tree import (
        TreeCreateRequest,
        TreeUpdateRequest,
        TreePatchRequest,
        TreeResponse,
        TreeListQuery,
        TreeListData,
        TreeEvidenceResponse,
        TreeEvidenceListData,
    )
except ImportError:
    pass  # tree.py may not exist in all environments

# Ingestion DTOs
from .ingestion import (
    GpsAngle,
    IngestQueryParams,
    IngestResponse,
    BatchIngestRequest,
    BatchIngestResponse,
    BoxIngestItem,
    BoxIngestRequest,
    BoxIngestResponse,
    TransparentIngestResponse,
)

# Verification DTOs
from .verification import (
    GeoFilter,
    AngleFilter,
    TimeFilter,
    VerifyQueryParams,
    MatchCandidateDTO,
    BestMatchDTO,
    VerifyResponse,
    VerifyErrorResponse,
    DebugFeaturesResponse,
    DebugPostgresResponse,
)

__all__ = [
    # Common
    "HealthResponse",
    "ErrorResponse",
    "RootResponse",
    "FeatureInfo",
    "StorageKeys",
    "IngestResultItem",
    # Tree
    "TreeCreateRequest",
    "TreeUpdateRequest",
    "TreePatchRequest",
    "TreeResponse",
    "TreeListQuery",
    "TreeListData",
    "TreeEvidenceResponse",
    "TreeEvidenceListData",
    # Ingestion
    "GpsAngle",
    "IngestQueryParams",
    "IngestResponse",
    "BatchIngestRequest",
    "BatchIngestResponse",
    "BoxIngestItem",
    "BoxIngestRequest",
    "BoxIngestResponse",
    "TransparentIngestResponse",
    # Verification
    "GeoFilter",
    "AngleFilter",
    "TimeFilter",
    "VerifyQueryParams",
    "MatchCandidateDTO",
    "BestMatchDTO",
    "VerifyResponse",
    "VerifyErrorResponse",
    "DebugFeaturesResponse",
    "DebugPostgresResponse",
]
