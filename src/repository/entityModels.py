#!/usr/bin/env python3
"""
SQLAlchemy ORM models for SAM3.

This module maps the current PostgreSQL/PostGIS schema:
- farm_zones
- trees
- tree_evidences
"""

from __future__ import annotations

import uuid as uuid_stdlib
from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base, relationship, synonym

try:
    from geoalchemy2 import Geography

    HAS_GEOALCHEMY = True
except ImportError:
    HAS_GEOALCHEMY = False
    Geography = None  # type: ignore[assignment,misc]

# Try to import pgvector halfvec type
try:
    from pgvector.sqlalchemy import Vector

    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    Vector = None  # type: ignore[assignment,misc]


_CACHED_VECTOR_DIM: int | None = None
_FALLBACK_VECTOR_DIM = 384


def get_vector_dim() -> int:
    """Return configured vector dim for compatibility with existing startup checks."""
    global _CACHED_VECTOR_DIM
    if _CACHED_VECTOR_DIM is not None:
        return _CACHED_VECTOR_DIM
    try:
        from src.config.appConfig import get_config

        _CACHED_VECTOR_DIM = get_config().postgres_vector_dim
        return _CACHED_VECTOR_DIM
    except Exception:
        return _FALLBACK_VECTOR_DIM


def validate_vector_dim() -> int:
    """Compatibility shim retained for API lifespan startup."""
    return get_vector_dim()


def get_column_types() -> Dict[str, str]:
    """Return a compact summary of selected column types."""
    return {
        "geography": "Geography(POINT, 4326)" if HAS_GEOALCHEMY else "Text (WKT)",
        "evidence_pk": "UUID",
    }


Base = declarative_base()


class Tree(Base):
    """Tree model — mirrors ``trees`` table."""

    __tablename__ = "trees"
    __allow_unmapped__ = True

    # DB primary key: single varchar column "id"
    id: str = Column("id", String(50), primary_key=True)
    region_code: str = Column("region_code", String(10), nullable=False, index=True)
    farm_id: str = Column(
        "farm_id",
        String(50),
        ForeignKey("farm_zones.farm_id", ondelete="RESTRICT"),
        nullable=False,
    )
    geohash7: str = Column("geohash_7", String(7), nullable=False, index=True)

    location = Column(
        "location",
        Geography("POINT", srid=4326) if HAS_GEOALCHEMY else Text,
        nullable=True,
    )

    row_idx: int | None = Column("row_idx", SmallInteger, nullable=True)
    col_idx: int | None = Column("col_idx", SmallInteger, nullable=True)
    codebook_id: str | None = Column("codebook_id", String(50), nullable=True)
    # Named tree_metadata internally; 'metadata' cannot be a class attribute (reserved by ORM).
    # Use column_property() at module level to map it to the DB 'metadata' column.
    tree_metadata: Dict[str, Any] = Column("metadata", JSONB, nullable=True, default=dict)
    captured_at: datetime | None = Column("captured_at", DateTime(timezone=True), nullable=True)

    created_at: datetime = Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: datetime = Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    evidences: List[TreeEvidence] = relationship(
        "TreeEvidence",
        back_populates="tree",
        cascade="all, delete-orphan",
        lazy="select",
    )

    farm_zone = relationship(
        "FarmZone",
        back_populates="trees",
        foreign_keys="Tree.farm_id",
        primaryjoin="Tree.farm_id == FarmZone.farm_id",
        lazy="select",
        viewonly=True,
    )

    __table_args__ = (
        Index("idx_trees_grid", "farm_id", "row_idx", "col_idx"),
        Index("idx_trees_location", "location", postgresql_using="gist"),
        Index("idx_trees_geohash", "geohash_7"),
        Index("idx_trees_geohash_fm", "farm_id", "geohash_7"),
        Index("idx_trees_updated_at", "updated_at"),
    )

    def __repr__(self) -> str:
        return f"<Tree(id={self.id}, region_code={self.region_code}, farm_id={self.farm_id})>"


class TreeEvidence(Base):
    """Tree evidence model — mirrors ``tree_evidences`` table."""

    __tablename__ = "tree_evidences"
    __allow_unmapped__ = True

    # DB primary key: single varchar column "id" (UUID string)
    evidence_id: str = Column("id", String(50), primary_key=True)
    # Backward-compatible alias
    id = synonym("evidence_id")

    # FK to trees.id (NOT composite region_code+tree_id)
    tree_id: str = Column(
        "tree_id",
        String(50),
        ForeignKey("trees.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    region_code: str = Column("region_code", String(10), nullable=False, index=True)

    # DINO global descriptor vector — halfvec(384) in PostgreSQL
    # Nullable so evidence can be created before vector extraction
    global_vector: Any = Column(
        "global_vector",
        Vector(get_vector_dim()) if HAS_PGVECTOR else Text,
        nullable=True,
    )

    storage_cid: str = Column("storage_cid", String(128), nullable=False)
    evidence_hash: str = Column("evidence_hash", String(64), nullable=False)
    is_c2pa_verified: bool = Column(
        "is_c2pa_verified",
        Boolean,
        nullable=False,
        server_default=text("false"),
    )

    camera_heading: int | None = Column("camera_heading", SmallInteger, nullable=True)
    camera_pitch: int | None = Column("camera_pitch", SmallInteger, nullable=True)
    camera_roll: int | None = Column("camera_roll", SmallInteger, nullable=True)

    # Two JSONB columns: raw_telemetry (sensor data) and metadata (tree/image metadata)
    raw_telemetry: Dict[str, Any] = Column(
        "raw_telemetry",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    # Named tree_metadata internally; 'metadata' is reserved by SQLAlchemy ORM.
    # The DB column is "metadata". Access via .tree_metadata attribute.
    tree_metadata: Dict[str, Any] = Column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    # PostGIS point for capture location
    location = Column(
        "location",
        Geography("POINT", srid=4326) if HAS_GEOALCHEMY else Text,
        nullable=True,
    )

    captured_at: datetime | None = Column(
        "captured_at",
        DateTime(timezone=True),
        nullable=True,
        server_default=func.now(),
    )
    created_at: datetime = Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: datetime = Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    tree: Tree = relationship("Tree", back_populates="evidences")

    __table_args__ = (
        Index("idx_evidences_tree_id", "tree_id"),
        Index("idx_evidences_region", "region_code"),
        Index("idx_evidences_captured", "captured_at"),
        Index("idx_evidences_telemetry", "raw_telemetry", postgresql_using="gin"),
        Index("idx_evidences_camera", "tree_id", "camera_heading", "camera_pitch"),
        Index("idx_evidences_location", "location", postgresql_using="gist"),
    )

    def __repr__(self) -> str:
        return (
            f"<TreeEvidence(id={self.evidence_id}, tree_id={self.tree_id}, "
            f"captured_at={self.captured_at})>"
        )