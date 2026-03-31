#!/usr/bin/env python3
"""
Tree DTOs for SAM3 API.

Contains request/response models for tree table endpoints.
"""

from __future__ import annotations

import re
import struct
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


_POINT_WKT_RE = re.compile(
    r"^\s*(?:SRID=\d+;)?POINT\s*\(\s*([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s*\)\s*$",
    re.IGNORECASE,
)


def _parse_point_wkb(raw: bytes) -> Tuple[Optional[float], Optional[float]]:
    """Parse 2D WKB/EWKB point bytes into latitude/longitude."""
    if len(raw) < 1 + 4 + 16:
        return (None, None)

    byte_order = raw[0]
    if byte_order == 0:
        endian = ">"
    elif byte_order == 1:
        endian = "<"
    else:
        return (None, None)

    geom_type = struct.unpack(f"{endian}I", raw[1:5])[0]
    offset = 5

    # EWKB SRID flag
    if geom_type & 0x20000000:
        if len(raw) < offset + 4 + 16:
            return (None, None)
        offset += 4

    if (geom_type & 0xFF) != 1:
        return (None, None)

    longitude, latitude = struct.unpack(f"{endian}dd", raw[offset:offset + 16])
    return (float(latitude), float(longitude))


def _extract_lat_lon(location: Any) -> Tuple[Optional[float], Optional[float]]:
    """Extract latitude/longitude from PostGIS-compatible point values."""
    if location is None:
        return (None, None)

    raw = getattr(location, "data", location)

    if isinstance(raw, memoryview):
        raw = raw.tobytes()

    if isinstance(raw, bytes):
        return _parse_point_wkb(raw)

    if isinstance(raw, str):
        if all(c in "0123456789abcdefABCDEF" for c in raw) and len(raw) % 2 == 0:
            try:
                return _parse_point_wkb(bytes.fromhex(raw))
            except ValueError:
                pass

        match = _POINT_WKT_RE.match(raw)
        if match:
            longitude = float(match.group(1))
            latitude = float(match.group(2))
            return (latitude, longitude)

    return (None, None)


class TreeCreateRequest(BaseModel):
    """Request body for creating a tree record."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Tree identifier",
    )
    region_code: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Region code",
    )
    farm_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Farm identifier",
    )
    geohash_7: str = Field(
        ...,
        min_length=7,
        max_length=7,
        pattern=r"^[a-z0-9]+$",
        description="Geohash-7 of the tree location (7 lowercase alphanumeric chars)",
    )
    latitude: Optional[float] = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="GPS latitude used to build the PostGIS location",
    )
    longitude: Optional[float] = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="GPS longitude used to build the PostGIS location",
    )
    row_idx: Optional[int] = Field(default=None, ge=0, description="Grid row index")
    col_idx: Optional[int] = Field(default=None, ge=0, description="Grid column index")
    codebook_id: str = Field(
        default="codebook_v1",
        min_length=1,
        max_length=128,
        description="Codebook version",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tree metadata",
    )
    captured_at: Optional[datetime] = Field(
        default=None,
        description="Capture timestamp",
    )

    @model_validator(mode="after")
    def validate_all_fields(self) -> "TreeCreateRequest":
        """Validate required strings are non-empty and lat/lon are paired."""
        if not self.id or not self.id.strip():
            raise ValueError("id cannot be empty or blank")
        if not self.region_code or not self.region_code.strip():
            raise ValueError("region_code cannot be empty or blank")
        if not self.farm_id or not self.farm_id.strip():
            raise ValueError("farm_id cannot be empty or blank")
        if self.codebook_id is not None and not self.codebook_id.strip():
            raise ValueError("codebook_id cannot be blank")
        # geohash_7 already validated by field constraints; enforce lowercase
        if self.geohash_7 != self.geohash_7.lower():
            raise ValueError("geohash_7 must be lowercase")
        # Metadata: if provided as non-dict, fail
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        # Lat/lon must be both present or both absent
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        return self


class TreeUpdateRequest(BaseModel):
    """Request body for fully updating a tree record."""

    model_config = ConfigDict(extra="forbid")

    region_code: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Region code",
    )
    farm_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Farm identifier",
    )
    geohash_7: str = Field(
        ...,
        min_length=7,
        max_length=7,
        pattern=r"^[a-z0-9]+$",
        description="Geohash-7 of the tree location (7 lowercase alphanumeric chars)",
    )
    latitude: Optional[float] = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="GPS latitude used to build the PostGIS location",
    )
    longitude: Optional[float] = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="GPS longitude used to build the PostGIS location",
    )
    row_idx: Optional[int] = Field(default=None, ge=0, description="Grid row index")
    col_idx: Optional[int] = Field(default=None, ge=0, description="Grid column index")
    codebook_id: str = Field(
        default="codebook_v1",
        min_length=1,
        max_length=128,
        description="Codebook version",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tree metadata",
    )
    captured_at: Optional[datetime] = Field(
        default=None,
        description="Capture timestamp",
    )

    @model_validator(mode="after")
    def validate_all_fields(self) -> "TreeUpdateRequest":
        """Validate required strings are non-empty and lat/lon are paired."""
        if not self.region_code or not self.region_code.strip():
            raise ValueError("region_code cannot be empty or blank")
        if not self.farm_id or not self.farm_id.strip():
            raise ValueError("farm_id cannot be empty or blank")
        if self.codebook_id is not None and not self.codebook_id.strip():
            raise ValueError("codebook_id cannot be blank")
        if self.geohash_7 != self.geohash_7.lower():
            raise ValueError("geohash_7 must be lowercase")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        return self


class TreePatchRequest(BaseModel):
    """Request body for partially updating a tree record.

    Any subset of fields may be provided. Fields not sent in the request are
    left unchanged on the server. At least one field must be present.
    """

    model_config = ConfigDict(extra="forbid")

    region_code: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Region code",
    )
    farm_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Farm identifier",
    )
    geohash_7: Optional[str] = Field(
        default=None,
        min_length=7,
        max_length=7,
        pattern=r"^[a-z0-9]+$",
        description="Geohash-7 of the tree location (7 lowercase alphanumeric chars)",
    )
    latitude: Optional[float] = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="GPS latitude used to build the PostGIS location",
    )
    longitude: Optional[float] = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="GPS longitude used to build the PostGIS location",
    )
    row_idx: Optional[int] = Field(
        default=None,
        ge=0,
        description="Grid row index",
    )
    col_idx: Optional[int] = Field(
        default=None,
        ge=0,
        description="Grid column index",
    )
    codebook_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Codebook version",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tree metadata",
    )
    captured_at: Optional[datetime] = Field(
        default=None,
        description="Capture timestamp",
    )

    @model_validator(mode="after")
    def validate_patch_payload(self) -> "TreePatchRequest":
        """Require at least one field is provided and lat/lon are paired."""
        if not self.model_fields_set:
            raise ValueError("at least one field must be provided in the request body")

        # Validate non-empty strings when explicitly set
        for field_name in ("region_code", "farm_id", "codebook_id"):
            val = getattr(self, field_name, None)
            if val is not None and (not isinstance(val, str) or not val.strip()):
                raise ValueError(f"{field_name} cannot be blank")

        # geohash_7 must be lowercase when provided
        if self.geohash_7 is not None and self.geohash_7 != self.geohash_7.lower():
            raise ValueError("geohash_7 must be lowercase")

        # metadata must be a dict when provided
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")

        # Lat/lon: if one is sent, both must be sent together
        sent_coords = {"latitude", "longitude"} & self.model_fields_set
        if sent_coords and sent_coords != {"latitude", "longitude"}:
            raise ValueError("latitude and longitude must be provided together")

        if sent_coords == {"latitude", "longitude"}:
            if (self.latitude is None) != (self.longitude is None):
                raise ValueError("latitude and longitude must be provided together")

        return self


class TreeResponse(BaseModel):
    """Response payload for a single tree record."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Tree identifier")
    region_code: str = Field(description="Region code")
    farm_id: str = Field(description="Farm identifier")
    geohash_7: str = Field(description="Geohash-7 of the tree location")
    latitude: Optional[float] = Field(default=None, description="GPS latitude")
    longitude: Optional[float] = Field(default=None, description="GPS longitude")
    row_idx: Optional[int] = Field(default=None, description="Grid row index")
    col_idx: Optional[int] = Field(default=None, description="Grid column index")
    codebook_id: Optional[str] = Field(default=None, description="Codebook version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Tree metadata")
    captured_at: Optional[datetime] = Field(default=None, description="Capture timestamp")
    created_at: datetime = Field(description="Record creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

    @classmethod
    def from_record(cls, tree: Any) -> "TreeResponse":
        """Build a response DTO from a repository record or ORM object."""
        metadata = getattr(tree, "metadata", None)
        if metadata is None:
            metadata = getattr(tree, "tree_metadata", None)
        if metadata is None:
            metadata = {}

        latitude, longitude = _extract_lat_lon(getattr(tree, "location", None))
        geohash = getattr(tree, "geohash_7", None)
        if geohash is None:
            geohash = getattr(tree, "geohash7")

        return cls(
            id=getattr(tree, "id"),
            region_code=getattr(tree, "region_code"),
            farm_id=getattr(tree, "farm_id"),
            geohash_7=geohash,
            latitude=latitude,
            longitude=longitude,
            row_idx=getattr(tree, "row_idx", None),
            col_idx=getattr(tree, "col_idx", None),
            codebook_id=getattr(tree, "codebook_id", None),
            metadata=metadata,
            captured_at=getattr(tree, "captured_at", None),
            created_at=getattr(tree, "created_at"),
            updated_at=getattr(tree, "updated_at"),
        )


class TreeListQuery(BaseModel):
    """Query parameters for listing trees."""

    model_config = ConfigDict(extra="forbid")

    farm_id: Optional[str] = Field(default=None, description="Filter by farm identifier")
    region_code: Optional[str] = Field(default=None, description="Filter by region code")
    row_idx: Optional[int] = Field(default=None, ge=0, description="Filter by grid row index")
    col_idx: Optional[int] = Field(default=None, ge=0, description="Filter by grid column index")
    limit: int = Field(default=100, ge=1, description="Maximum number of trees to return")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class TreeListData(BaseModel):
    """List payload for tree table responses."""

    model_config = ConfigDict(extra="forbid")

    items: List[TreeResponse] = Field(default_factory=list, description="Tree records")
    limit: int = Field(description="Applied result limit")
    offset: int = Field(description="Applied result offset")


class TreeEvidenceResponse(BaseModel):
    """Single evidence record in a tree's evidence list."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="Evidence record UUID")
    tree_id: str = Field(description="Parent tree identifier")
    region_code: Optional[str] = Field(default=None, description="Region code")
    camera_heading: Optional[int] = Field(default=None, description="Camera heading (degrees)")
    camera_pitch: Optional[int] = Field(default=None, description="Camera pitch (degrees)")
    storage_cid: str = Field(description="MinIO / storage content identifier")
    evidence_hash: str = Field(description="Content hash of the evidence")
    is_c2pa_verified: bool = Field(default=False, description="C2PA provenance verification status")
    captured_at: Optional[int] = Field(default=None, description="Capture timestamp (Unix epoch ms)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Evidence metadata")

    @classmethod
    def from_record(cls, record: Any) -> "TreeEvidenceResponse":
        """Build from a TreeEvidenceRecord (SQLAlchemy ORM) or similar."""
        # raw_telemetry is stored as tree_metadata field; also expose tree_metadata separately
        raw_telemetry = getattr(record, "raw_telemetry", {}) or {}
        tree_metadata = getattr(record, "metadata", {}) or {}
        merged_metadata: Dict[str, Any] = dict(raw_telemetry)
        merged_metadata.update(tree_metadata)
        return cls(
            id=str(record.id),
            tree_id=record.tree_id,
            region_code=record.region_code,
            camera_heading=record.camera_heading,
            camera_pitch=record.camera_pitch,
            storage_cid=record.storage_cid,
            evidence_hash=record.evidence_hash,
            is_c2pa_verified=record.is_c2pa_verified,
            captured_at=int(record.captured_at.timestamp() * 1000) if record.captured_at else None,
            metadata=merged_metadata,
        )


class TreeEvidenceListData(BaseModel):
    """List payload for tree evidence responses."""

    model_config = ConfigDict(extra="forbid")

    items: List[TreeEvidenceResponse] = Field(default_factory=list, description="Evidence records")
    total: int = Field(description="Total number of evidences for this tree")
    tree_id: str = Field(description="Parent tree identifier")


__all__ = [
    "TreeCreateRequest",
    "TreeUpdateRequest",
    "TreePatchRequest",
    "TreeResponse",
    "TreeListQuery",
    "TreeListData",
    "TreeEvidenceResponse",
    "TreeEvidenceListData",
]
