#!/usr/bin/env python3
from __future__ import annotations

"""
Spatial ORM Models for SAM3 Tree Identification System.

Extends ``entityModels.py`` with ``FarmZone``.
Uses GeoAlchemy2 for PostGIS geometry and the same declarative ``Base`` as ``entityModels``.
"""

from typing import Any

from datetime import datetime

from sqlalchemy import Column, String, DateTime, Text, Float
from sqlalchemy import Index, func
from sqlalchemy.orm import relationship

from src.repository.entityModels import Base

# MUST import the same Base used by entityModels so that all models share a
# single MetaData registry.  If this import fails the module will hard-error,
# which is intentional — a broken import is far better than a silent orphan Base.

try:
    from geoalchemy2 import Geometry

    HAS_GEOALCHEMY = True
except ImportError:
    HAS_GEOALCHEMY = False
    Geometry = None  # type: ignore[assignment,misc]


class FarmZone(Base):
    """Farm polygon in SRID 4326; PK is ``farm_id`` (referenced by ``trees.farm_id``)."""

    __tablename__ = "farm_zones"
    __allow_unmapped__ = True  # Required: legacy Column() style is incompatible with SA 2.x strict Mapped[] typing

    farm_id: str = Column("farm_id", String(50), primary_key=True)
    owner_did: str = Column("owner_did", String(100), nullable=False)
    region_code: str = Column("region_code", String(10), nullable=False)
    farm_name: str | None = Column("farm_name", String(255), nullable=True)

    # PostGIS POLYGON — fixed at import time; migrate the DB when switching
    # between environments so the column type matches.
    boundary = Column(
        "boundary",
        Geometry("POLYGON", srid=4326, spatial_index=False) if HAS_GEOALCHEMY else Text,
        nullable=False,
    )

    # Grid topology (optional): origin + spacing for row/col index derivation.
    lon_origin: float | None = Column("lon_origin", Float, nullable=True)
    lat_origin: float | None = Column("lat_origin", Float, nullable=True)
    row_spacing: float | None = Column("row_spacing", Float, nullable=True)
    col_spacing: float | None = Column("col_spacing", Float, nullable=True)

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

    # viewonly=True mirrors Tree.farm_zone so writes are only possible via
    # Tree.farm_zone (Issue #5 fix).
    # The FK on Tree.farm_id uses ondelete="RESTRICT", so orphan trees are prevented.
    trees = relationship(
        "Tree",
        back_populates="farm_zone",
        foreign_keys="Tree.farm_id",
        primaryjoin="FarmZone.farm_id == Tree.farm_id",
        viewonly=True,
        lazy="select",
    )

    __table_args__ = (
        Index(
            "idx_farm_zones_boundary",
            "boundary",
            postgresql_using="gist",
        ),
        Index("idx_farm_zones_region_code", "region_code"),
        Index("idx_farm_zones_owner_did", "owner_did"),
    )

    def __repr__(self) -> str:
        return (
            f"<FarmZone(farm_id={self.farm_id}, "
            f"owner_did={self.owner_did}, region_code={self.region_code})>"
        )


__all__ = [
    "Base",
    "FarmZone",
]