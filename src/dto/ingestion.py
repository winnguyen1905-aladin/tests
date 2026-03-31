#!/usr/bin/env python3
"""
Ingestion DTOs for SAM3 API.

Contains request/response models for tree image ingestion endpoints.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
import json


class GpsAngle(BaseModel):
    """GPS coordinates and viewing angles for a tree image."""
    latitude: float = Field(description="Latitude coordinate")
    longitude: float = Field(description="Longitude coordinate")
    hor_angle: float = Field(description="Horizontal viewing angle in degrees")
    ver_angle: float = Field(description="Vertical viewing angle in degrees")
    pitch: Optional[float] = Field(default=None, description="Camera pitch angle (optional)")


class IngestQueryParams(BaseModel):
    """Query parameters for /ingest endpoint."""
    imageId: str = Field(description="Unique identifier for the image")
    treeId: str = Field(description="Tree identifier")
    latitude: float = Field(description="Latitude coordinate (required)")
    longitude: float = Field(description="Longitude coordinate (required)")
    hor_angle: float = Field(description="Horizontal viewing angle in degrees (required)")
    ver_angle: float = Field(description="Vertical viewing angle in degrees (required)")
    captured_at: str = Field(description="Capture timestamp (ISO 8601, e.g. 2025-06-01T10:30:00Z)")


class IngestResponse(BaseModel):
    """Response from /ingest endpoint."""
    success: bool = Field(description="Whether ingestion succeeded")
    imageId: str = Field(description="Image identifier")
    treeId: str = Field(description="Tree identifier")
    features_extracted: Dict[str, int] = Field(description="Feature extraction info")
    storage_keys: Dict[str, str] = Field(description="Storage keys for features")
    message: str = Field(description="Result message")


class BatchIngestRequest(BaseModel):
    """Request body for batch ingestion endpoint."""
    imageIds: List[str] = Field(description="List of image identifiers")
    treeIds: List[str] = Field(description="List of tree identifiers")
    batch_size: int = Field(default=8, description="GPU batch size")

    @field_validator('imageIds', 'treeIds')
    @classmethod
    def validate_lists_match(cls, v, info):
        if 'imageIds' in info.field_name or 'treeIds' in info.field_name:
            # Will be validated at a higher level
            return v
        return v


class BatchIngestResponse(BaseModel):
    """Response from batch ingestion endpoint."""
    success: bool = Field(description="Overall success status")
    total: int = Field(description="Total images in batch")
    succeeded: int = Field(description="Number of successful ingestions")
    failed: int = Field(description="Number of failed ingestions")
    results: List[Dict[str, Any]] = Field(description="Individual ingestion results")


class BoxIngestItem(BaseModel):
    """Per-image metadata for box-prompt ingestion."""
    imageId: Optional[str] = Field(default=None, description="Image identifier (auto-generated if omitted)")
    treeId: Optional[str] = Field(default=None, description="Tree identifier (auto-generated if omitted)")
    label: Optional[str] = Field(default=None, description="Tree label")
    box_coordinates: Optional[List[List[int]]] = Field(default=None, description="Box coordinates [[x1,y1,x2,y2],...]")
    timestamp: Optional[str] = Field(default=None, description="Capture timestamp")
    latitude: Optional[float] = Field(default=None, description="Latitude coordinate")
    longitude: Optional[float] = Field(default=None, description="Longitude coordinate")
    confidence: Optional[float] = Field(default=None, description="Detection confidence")
    counter: Optional[int] = Field(default=None, description="Image counter")
    signature: Optional[str] = Field(default=None, description="Device signature")
    device_id: Optional[str] = Field(default=None, description="Device identifier")
    nonce: Optional[str] = Field(default=None, description="Nonce for deduplication")


class BoxIngestRequest(BaseModel):
    """Request body for box-prompt batch ingestion."""
    items: List[BoxIngestItem] = Field(description="Per-image metadata objects")
    batch_size: int = Field(default=8, description="GPU batch size")

    @classmethod
    def from_json_string(cls, json_str: str) -> "BoxIngestRequest":
        """Parse from JSON string."""
        data = json.loads(json_str)
        return cls(items=[BoxIngestItem(**item) for item in data])


class BoxIngestResponse(BaseModel):
    """Response from box-prompt batch ingestion."""
    success: bool = Field(description="Overall success status")
    total: int = Field(description="Total images in batch")
    succeeded: int = Field(description="Number of successful ingestions")
    failed: int = Field(description="Number of failed ingestions")
    results: List[Dict[str, Any]] = Field(description="Individual ingestion results")


class TransparentIngestResponse(BaseModel):
    """Response from transparent ingestion endpoint."""
    success: bool = Field(description="Whether ingestion succeeded")
    imageId: str = Field(description="Image identifier")
    treeId: str = Field(description="Tree identifier")
    mask_shape: Optional[Dict[str, int]] = Field(default=None, description="Segmentation mask shape")
    features_extracted: Dict[str, int] = Field(description="Feature extraction info")
    storage_keys: Dict[str, str] = Field(description="Storage keys for features")
    message: str = Field(description="Result message")
