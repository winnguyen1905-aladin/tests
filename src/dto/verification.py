#!/usr/bin/env python3
"""
Verification DTOs for SAM3 API.

Contains request/response models for tree verification endpoints.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class GeoFilter(BaseModel):
    """Geospatial filter for verification queries."""
    radius_meters: float = Field(description="Search radius in meters")
    latitude: float = Field(description="Center latitude")
    longitude: float = Field(description="Center longitude")


class AngleFilter(BaseModel):
    """Angle filter for verification queries."""
    hor_angle_min: float = Field(description="Minimum horizontal angle")
    hor_angle_max: float = Field(description="Maximum horizontal angle")
    ver_angle_min: float = Field(description="Minimum vertical angle")
    ver_angle_max: float = Field(description="Maximum vertical angle")
    pitch_min: Optional[float] = Field(default=None, description="Minimum pitch angle")
    pitch_max: Optional[float] = Field(default=None, description="Maximum pitch angle")


class TimeFilter(BaseModel):
    """Time filter for verification queries."""
    captured_at_min: int = Field(description="Minimum capture timestamp (epoch seconds)")
    captured_at_max: int = Field(description="Maximum capture timestamp (epoch seconds)")


class VerifyQueryParams(BaseModel):
    """Query parameters for /verify endpoint."""
    known_tree_id: Optional[str] = Field(default=None, description="Optional: restrict search to specific tree")
    latitude: float = Field(description="Query latitude (required)")
    longitude: float = Field(description="Query longitude (required)")
    hor_angle: float = Field(description="Query horizontal viewing angle (required)")
    ver_angle: float = Field(description="Query vertical viewing angle (required)")
    captured_at: str = Field(description="Query image capture time (ISO 8601)")
    pitch: Optional[float] = Field(default=None, description="Query camera pitch angle (optional)")
    radius: Optional[float] = Field(default=None, description="Search radius in meters (optional)")


class MatchCandidateDTO(BaseModel):
    """A candidate match from verification."""
    image_id: str = Field(description="Image identifier")
    tree_id: str = Field(description="Tree identifier")
    similarity_score: float = Field(description="Similarity score")
    rank: int = Field(description="Rank in results")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class BestMatchDTO(BaseModel):
    """Best match result from verification."""
    tree_id: str = Field(description="Matched tree identifier")
    image_id: str = Field(description="Matched image identifier")
    confidence: float = Field(description="Match confidence score")
    similarity: Optional[float] = Field(default=None, description="Similarity score")
    match_count: Optional[int] = Field(default=None, description="Number of matched keypoints")
    inlier_count: Optional[int] = Field(default=None, description="Number of inliers")
    geo_distance: Optional[float] = Field(default=None, description="Geodesic distance in meters")
    angle_diff: Optional[float] = Field(description="Angle difference")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class VerifyResponse(BaseModel):
    """Response from /verify endpoint."""
    status: str = Field(description="Verification status")
    decision: str = Field(description="Match decision (MATCH/NO_MATCH/ERROR)")
    confidence: float = Field(description="Confidence score")
    best_match: Optional[Dict[str, Any]] = Field(default=None, description="Best match details")
    matched_tree_id: Optional[str] = Field(default=None, description="Matched tree ID at top level")
    reason: str = Field(description="Decision reason")


class VerifyErrorResponse(BaseModel):
    """Error response from /verify endpoint."""
    success: bool = Field(default=False, description="Always false for errors")
    decision: str = Field(description="Decision status")
    error: str = Field(description="Error message")
    message: str = Field(description="Human-readable message")


class DebugFeaturesResponse(BaseModel):
    """Debug response for feature information."""
    success: bool = Field(description="Success status")
    total_entities: int = Field(description="Total entities in database")
    total_features: int = Field(description="Total features stored")
    features_by_tree: Dict[str, int] = Field(description="Features grouped by tree")
    message: str = Field(description="Status message")


class DebugPostgresResponse(BaseModel):
    """Debug response for PostgreSQL status."""
    success: bool = Field(description="Success status")
    database: str = Field(description="Database name")
    evidence_count: int = Field(description="Number of evidence records")
    vector_dim: int = Field(description="Vector dimension")
    column_types: Dict[str, str] = Field(description="Column type mappings")
