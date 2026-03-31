#!/usr/bin/env python3
"""
Common DTOs for SAM3 API.

Contains shared request/response models used across multiple endpoints.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Health status (healthy/degraded/unhealthy)")
    message: str = Field(description="Human-readable message")


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(description="Error type or code")
    message: str = Field(description="Human-readable error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")


class RootResponse(BaseModel):
    """API root information response."""
    name: str = Field(description="API name")
    version: str = Field(description="API version")
    endpoints: List[Dict[str, Any]] = Field(description="Available endpoints")


class FeatureInfo(BaseModel):
    """Feature extraction information."""
    global_dim: int = Field(description="Global feature dimension")
    local_keypoints: int = Field(description="Number of local keypoints")
    local_dim: int = Field(description="Local descriptor dimension")


class StorageKeys(BaseModel):
    """Storage keys for ingested data."""
    global_features: Optional[str] = Field(default=None, description="MinIO key for global features")
    local_features: Optional[str] = Field(default=None, description="MinIO key for local features")
    image: Optional[str] = Field(default=None, description="MinIO key for processed image")


class IngestResultItem(BaseModel):
    """Individual result item from batch ingestion."""
    imageId: str = Field(description="Image identifier")
    treeId: str = Field(description="Tree identifier")
    success: bool = Field(description="Whether ingestion succeeded")
    message: str = Field(description="Result message")
    features: Optional[Dict[str, Any]] = Field(default=None, description="Extracted features info")
