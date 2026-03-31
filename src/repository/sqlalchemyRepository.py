#!/usr/bin/env python3
"""
SQLAlchemy ORM Repository for SAM3 Tree Identification System

Provides CRUD operations using SQLAlchemy ORM with:
- Vector similarity search using pgvector
- Geospatial queries using GeoAlchemy2/PostGIS

This is an alternative to the raw SQL postgreRepository.py,
using ORM patterns for better Flask/FastAPI integration.
"""

import json
import logging
import math
import uuid
from decimal import Decimal, ROUND_FLOOR
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from src.repository.entityModels import Tree
# Ensure FarmZone is registered for Tree.farm_zone relationship configuration.
from src.repository.spatialEntityModels import FarmZone  # noqa: F401
import numpy as np

logger = logging.getLogger(__name__)

# Namespace for deriving UUIDs from non-UUID evidence ids (e.g. image slugs like "tree-001")
_EVIDENCE_SLUG_NAMESPACE = uuid.NAMESPACE_URL


def _resolve_evidence_uuid(evidence_id: Optional[str]) -> uuid.UUID:
    """Return a UUID from ``evidence_id``, or a new random UUID if unset.

    If ``evidence_id`` is not valid hex UUID (e.g. ``tree-001``), use a stable
    UUID5 so re-ingest updates the same row.
    """
    if not evidence_id:
        return uuid.uuid4()
    s = str(evidence_id).strip()
    try:
        return uuid.UUID(s)
    except (ValueError, TypeError):
        return uuid.uuid5(_EVIDENCE_SLUG_NAMESPACE, f"sam3:evidence:{s}")


def _evidence_pk_str(evidence_uuid: uuid.UUID) -> str:
    """ORM primary key for ``tree_evidences.id`` (DB column is ``varchar``)."""
    return str(evidence_uuid)


def _log_pgvector_halfvec_dim_mismatch(exc: BaseException, configured_dim: int) -> None:
    """Log remediation when DB ``global_vector`` width disagrees with ``POSTGRES_VECTOR_DIM``."""
    msg = str(exc).lower()
    if "different halfvec dimensions" in msg:
        logger.error(
            "PostgreSQL tree_evidences.global_vector type does not match POSTGRES_VECTOR_DIM=%s "
            "(queries use halfvec(%s)). Run: python3 -m src.repository.databaseManager "
            "--migrate-vector-dim %s (clears vectors; re-ingest evidence afterward). "
            "Or set POSTGRES_VECTOR_DIM to match the existing column.",
            configured_dim,
            configured_dim,
            configured_dim,
        )
        return
    # INSERT/UPDATE: e.g. "expected 768 dimensions, not 384"
    if "expected" in msg and "dimensions" in msg and "not" in msg:
        logger.error(
            "PostgreSQL tree_evidences.global_vector column expects a different size than "
            "POSTGRES_VECTOR_DIM=%s (vector length from DINO/ingestion). Run: "
            "python3 -m src.repository.databaseManager --migrate-vector-dim %s "
            "(clears vectors; re-ingest). Or raise POSTGRES_VECTOR_DIM to match the column.",
            configured_dim,
            configured_dim,
        )


def _coerce_row_evidence_id(val: Any) -> uuid.UUID:
    """Build ``uuid.UUID`` for DTOs from DB (str or UUID)."""
    if isinstance(val, uuid.UUID):
        return val
    return uuid.UUID(str(val))


def _coerce_metadata_dict(val: Any) -> Dict[str, Any]:
    """Normalize ORM/SQL JSON metadata values into dictionaries."""
    if isinstance(val, dict):
        return val
    if val in (None, ""):
        return {}
    if isinstance(val, str):
        try:
            loaded = json.loads(val)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    try:
        return dict(val)
    except (TypeError, ValueError):
        return {}


# Try to import vector type
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False

# Try to import GeoAlchemy2
try:
    from geoalchemy2 import Geography, func
    HAS_GEOALCHEMY = True
except ImportError:
    HAS_GEOALCHEMY = False

# Import MilvusResult for adapter methods
try:
    from src.repository.milvusRepository import MilvusResult
except ImportError:
    # MilvusResult defined inline if milvusRepository not available
    MilvusResult = None

from sqlalchemy import select, and_, or_, not_, text
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.dialects.postgresql import insert


# Dataclasses for results (matching postgreRepository.py)
class TreeRecord:
    """Tree record from database."""
    def __init__(
        self,
        id: str,
        region_code: str,
        farm_id: str,
        geohash7: str,
        location: Optional[Any],
        row_idx: Optional[int],
        col_idx: Optional[int],
        codebook_id: str,
        captured_at: Optional[datetime],
        created_at: datetime,
        updated_at: datetime,
        metadata: Dict[str, Any],
    ):
        self.id = id
        self.region_code = region_code
        self.farm_id = farm_id
        self.geohash7 = geohash7
        self.location = location
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.codebook_id = codebook_id
        self.captured_at = captured_at
        self.created_at = created_at
        self.updated_at = updated_at
        self.metadata = metadata


class TreeEvidenceRecord:
    """Tree evidence record from database."""
    def __init__(
        self,
        id: uuid.UUID,
        tree_id: str,
        region_code: str,
        global_vector: Optional[List[float]],
        camera_heading: Optional[int],
        camera_pitch: Optional[int],
        storage_cid: str,
        evidence_hash: str,
        is_c2pa_verified: bool,
        raw_telemetry: Dict[str, Any],
        captured_at: datetime,
        metadata: Dict[str, Any],
    ):
        self.id = id
        self.tree_id = tree_id
        self.region_code = region_code
        self.global_vector = global_vector
        self.camera_heading = camera_heading
        self.camera_pitch = camera_pitch
        self.storage_cid = storage_cid
        self.evidence_hash = evidence_hash
        self.is_c2pa_verified = is_c2pa_verified
        self.raw_telemetry = raw_telemetry
        self.captured_at = captured_at
        self.metadata = metadata


class VectorSearchResult:
    """Result from vector similarity search."""
    def __init__(
        self,
        evidence_id: uuid.UUID,
        tree_id: str,
        similarity: float,
        captured_at: datetime,
        metadata: Dict[str, Any],
        error: Optional[str] = None,
    ):
        self.evidence_id = evidence_id
        self.tree_id = tree_id
        self.similarity = similarity
        self.captured_at = captured_at
        self.metadata = metadata
        self.error = error


class GeoSearchResult:
    """Result from geospatial search."""
    def __init__(
        self,
        tree_id: str,
        distance_meters: float,
        location: Optional[Any],
    ):
        self.tree_id = tree_id
        self.distance_meters = distance_meters
        self.location = location


class SQLAlchemyORMRepository:
    """
    SQLAlchemy ORM Repository for PostgreSQL with pgvector and PostGIS.

    Provides ORM-based CRUD operations with the same interface as the
    raw SQL postgreRepository.py for easy migration.
    """

    def __init__(self, session: Optional[Session] = None):
        """Initialize repository with optional session."""
        self._session = session

    @property
    def session(self) -> Session:
        """Get current session or create new one."""
        if self._session is None:
            from src.repository.databaseManager import get_session
            self._session = get_session()
        return self._session

    def set_session(self, session: Session) -> None:
        """Set session for operations."""
        self._session = session

    def close(self) -> None:
        """Close the underlying session and return the connection to the pool."""
        if self._session is not None:
            self._session.close()
            self._session = None

    # ==================== TREE OPERATIONS ====================

    def create_tree(
        self,
        tree_id: str,
        region_code: str,
        farm_id: str,
        geohash_7: str,
        row_idx: Optional[int],
        col_idx: Optional[int],
        longitude: Optional[float] = None,
        latitude: Optional[float] = None,
        codebook_id: str = "codebook_v1",
        metadata: Optional[Dict[str, Any]] = None,
        captured_at: Optional[datetime] = None,
    ) -> bool:
        """Create or update a tree record, allowing unknown grid coordinates."""
        try:
            # Build location from coordinates
            location = None
            if longitude is not None and latitude is not None:
                if HAS_GEOALCHEMY:
                    # Use WKT for GeoAlchemy2
                    location = f"POINT({longitude} {latitude})"
                else:
                    location = f"POINT({longitude} {latitude})"

            farm: FarmZone | None = None
            if longitude is not None and latitude is not None:
                farm = self.session.get(FarmZone, farm_id)

            if (
                row_idx is None
                and col_idx is None
                and longitude is not None
                and latitude is not None
                and farm is not None
            ):
                o_lon = farm.lon_origin
                o_lat = farm.lat_origin
                rs = farm.row_spacing
                cs = farm.col_spacing
                if o_lon is not None and o_lat is not None and rs is not None and cs is not None:
                    if rs == 0 or cs == 0:
                        raise ValueError(
                            "Farm row_spacing and col_spacing must be non-zero for grid auto-index"
                        )
                    # Decimal avoids binary float artifacts (e.g. (10.805-10.8)/0.001 < 5.0).
                    q_row = ((Decimal(str(latitude)) - Decimal(str(o_lat))) / Decimal(str(rs)))
                    q_col = ((Decimal(str(longitude)) - Decimal(str(o_lon))) / Decimal(str(cs)))
                    row_idx = int(q_row.to_integral_value(rounding=ROUND_FLOOR))
                    col_idx = int(q_col.to_integral_value(rounding=ROUND_FLOOR))

            if HAS_GEOALCHEMY and location:
                if farm is None:
                    raise ValueError(f"Unknown farm_id for tree location: {farm_id}")
                containment = self.session.execute(
                    text(
                        """
                        SELECT ST_Contains(
                            (SELECT boundary::geometry FROM farm_zones WHERE farm_id = :fid),
                            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
                        )
                        """
                    ),
                    {"fid": farm_id, "lon": longitude, "lat": latitude},
                ).scalar()
                if containment is not True:
                    raise ValueError(
                        f"Tree at ({longitude},{latitude}) is OUTSIDE farm boundary"
                    )

            # Check if tree exists
            existing = self.session.get(Tree, tree_id)

            if existing:
                # Update existing tree
                existing.region_code = region_code
                existing.farm_id = farm_id
                existing.geohash7 = geohash_7
                existing.row_idx = row_idx
                existing.col_idx = col_idx
                if location and HAS_GEOALCHEMY:
                    from geoalchemy2 import WKTElement
                    existing.location = WKTElement(location, srid=4326)
                existing.codebook_id = codebook_id
                existing.captured_at = captured_at
                existing.tree_metadata = metadata or {}
                existing.updated_at = datetime.now(timezone.utc)
            else:
                # Create new tree
                tree = Tree(
                    id=tree_id,
                    region_code=region_code,
                    farm_id=farm_id,
                    geohash7=geohash_7,
                    row_idx=row_idx,
                    col_idx=col_idx,
                    codebook_id=codebook_id,
                    captured_at=captured_at,
                    tree_metadata=metadata or {},
                )
                if location and HAS_GEOALCHEMY:
                    from geoalchemy2 import WKTElement
                    tree.location = WKTElement(location, srid=4326)
                self.session.add(tree)

            logger.info(f"Created/updated tree: {tree_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating tree {tree_id}: {e}")
            raise

    def get_tree(self, tree_id: str) -> Optional[TreeRecord]:
        """Get a tree by ID."""
        from src.repository.entityModels import Tree

        try:
            tree = self.session.get(Tree, tree_id)
            if tree is None:
                return None

            return TreeRecord(
                id=tree.id,
                region_code=tree.region_code,
                farm_id=tree.farm_id,
                geohash7=tree.geohash7,
                location=tree.location,
                row_idx=tree.row_idx,
                col_idx=tree.col_idx,
                codebook_id=tree.codebook_id,
                captured_at=tree.captured_at,
                created_at=tree.created_at,
                updated_at=tree.updated_at,
                metadata=_coerce_metadata_dict(tree.tree_metadata),
            )

        except Exception as e:
            logger.error(f"Error getting tree {tree_id}: {e}")
            return None

    def update_tree(self, tree_id: str, **updates) -> bool:
        """Update tree fields."""
        from src.repository.entityModels import Tree

        if not updates:
            return True

        try:
            tree = self.session.get(Tree, tree_id)
            if tree is None:
                return False

            allowed_fields = {
                "region_code", "farm_id", "geohash7", "row_idx", "col_idx",
                "codebook_id", "captured_at", "tree_metadata",
            }
            upd = dict(updates)
            if "metadata" in upd and "tree_metadata" not in upd:
                upd["tree_metadata"] = upd.pop("metadata")
            else:
                upd.pop("metadata", None)

            for field, value in upd.items():
                if field in allowed_fields:
                    setattr(tree, field, value)

            tree.updated_at = datetime.now(timezone.utc)
            logger.info(f"Updated tree: {tree_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating tree {tree_id}: {e}")
            raise

    def delete_tree(self, tree_id: str) -> bool:
        """Delete a tree (cascades to evidences)."""
        from src.repository.entityModels import Tree

        try:
            tree = self.session.get(Tree, tree_id)
            if tree:
                self.session.delete(tree)
                logger.info(f"Deleted tree: {tree_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting tree {tree_id}: {e}")
            raise

    def list_trees(
        self,
        farm_id: Optional[str] = None,
        region_code: Optional[str] = None,
        row_idx: Optional[int] = None,
        col_idx: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TreeRecord]:
        """List trees with optional filters."""
        from src.repository.entityModels import Tree

        try:
            query = select(Tree)

            if farm_id:
                query = query.where(Tree.farm_id == farm_id)
            if region_code:
                query = query.where(Tree.region_code == region_code)
            if row_idx is not None:
                query = query.where(Tree.row_idx == row_idx)
            if col_idx is not None:
                query = query.where(Tree.col_idx == col_idx)

            query = query.order_by(Tree.created_at.desc()).limit(limit).offset(offset)

            trees = self.session.execute(query).scalars().all()

            return [
                TreeRecord(
                    id=tree.id,
                    region_code=tree.region_code,
                    farm_id=tree.farm_id,
                    geohash7=tree.geohash7,
                    location=tree.location,
                    row_idx=tree.row_idx,
                    col_idx=tree.col_idx,
                    codebook_id=tree.codebook_id,
                    captured_at=tree.captured_at,
                    created_at=tree.created_at,
                    updated_at=tree.updated_at,
                    metadata=_coerce_metadata_dict(tree.tree_metadata),
                )
                for tree in trees
            ]

        except Exception as e:
            logger.error(f"Error listing trees: {e}")
            return []

    # ==================== TREE EVIDENCE OPERATIONS ====================

    def create_evidence(
        self,
        tree_id: str,
        region_code: str,
        global_vector: List[float],
        storage_cid: str,
        evidence_id: Optional[str] = None,
        evidence_hash: Optional[str] = None,
        camera_heading: Optional[int] = None,
        camera_pitch: Optional[int] = None,
        camera_roll: Optional[int] = None,
        is_c2pa_verified: bool = False,
        raw_telemetry: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        captured_at: Optional[int] = None,
    ) -> Optional[str]:
        """Create a new tree evidence record."""
        from src.repository.entityModels import TreeEvidence

        try:
            evidence_uuid = _resolve_evidence_uuid(evidence_id)
            evidence_pk = _evidence_pk_str(evidence_uuid)

            # Convert vector to appropriate format
            vector_value = None
            if global_vector:
                _pg_dim = self._get_vector_dimension()
                if len(global_vector) != _pg_dim:
                    logger.error(
                        f"[create_evidence] Vector dim mismatch: got {len(global_vector)}, "
                        f"expected {_pg_dim}. Check dino_model_type dim in "
                        f"SUPPORTED_MODELS and POSTGRES_VECTOR_DIM in .env."
                    )
                    raise ValueError(
                        f"global_vector dim={len(global_vector)} != expected={_pg_dim}"
                    )
                if HAS_PGVECTOR:
                    vector_value = global_vector  # pgvector handles conversion
                else:
                    vector_value = global_vector  # Store as JSONB

            # Build location
            location_value = None
            if latitude is not None and longitude is not None:
                if HAS_GEOALCHEMY:
                    from geoalchemy2 import WKTElement
                    location_value = WKTElement(f"POINT({longitude} {latitude})", srid=4326)
                else:
                    location_value = f"POINT({longitude} {latitude})"

            # Convert captured_at
            captured_datetime = None
            if captured_at:
                captured_datetime = datetime.fromtimestamp(captured_at, tz=timezone.utc)

            # Check if exists
            existing = self.session.get(TreeEvidence, evidence_pk)

            if existing:
                # Update existing
                existing.global_vector = vector_value
                existing.storage_cid = storage_cid
                existing.evidence_hash = evidence_hash
                existing.is_c2pa_verified = is_c2pa_verified
                existing.camera_heading = camera_heading
                existing.camera_pitch = camera_pitch
                existing.camera_roll = camera_roll
                existing.raw_telemetry = raw_telemetry or {}
                existing.tree_metadata = metadata or {}
                existing.captured_at = captured_datetime
                existing.location = location_value
            else:
                # Create new
                evidence = TreeEvidence(
                    id=evidence_pk,
                    tree_id=tree_id,
                    region_code=region_code,
                    global_vector=vector_value,
                    storage_cid=storage_cid,
                    evidence_hash=evidence_hash,
                    is_c2pa_verified=is_c2pa_verified,
                    camera_heading=camera_heading,
                    camera_pitch=camera_pitch,
                    camera_roll=camera_roll,
                    raw_telemetry=raw_telemetry or {},
                    tree_metadata=metadata or {},
                    captured_at=captured_datetime,
                    location=location_value,
                )
                self.session.add(evidence)

            logger.info(f"Created/updated evidence: {evidence_pk} for tree {tree_id}")
            return evidence_pk

        except Exception as e:
            _log_pgvector_halfvec_dim_mismatch(e, int(self._get_vector_dimension()))
            logger.error(f"Error creating evidence for tree {tree_id}: {e}")
            raise

    def get_evidence(self, evidence_id: uuid.UUID) -> Optional[TreeEvidenceRecord]:
        """Get evidence by ID."""
        from src.repository.entityModels import TreeEvidence

        try:
            evidence = self.session.get(TreeEvidence, str(evidence_id))
            if evidence is None:
                return None

            # Convert vector
            vector = evidence.global_vector
            if vector:
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()
                elif isinstance(vector, str):
                    vector = json.loads(vector)
            else:
                vector = []

            return TreeEvidenceRecord(
                id=_coerce_row_evidence_id(evidence.id),
                tree_id=evidence.tree_id,
                region_code=evidence.region_code,
                global_vector=vector,
                camera_heading=evidence.camera_heading,
                camera_pitch=evidence.camera_pitch,
                storage_cid=evidence.storage_cid,
                evidence_hash=evidence.evidence_hash,
                is_c2pa_verified=evidence.is_c2pa_verified,
                raw_telemetry=evidence.raw_telemetry if isinstance(evidence.raw_telemetry, dict) else json.loads(evidence.raw_telemetry) if evidence.raw_telemetry else {},
                captured_at=evidence.captured_at,
                metadata=_coerce_metadata_dict(evidence.tree_metadata),
            )

        except Exception as e:
            logger.error(f"Error getting evidence {evidence_id}: {e}")
            return None

    def get_evidence_by_image_id(self, image_id: str) -> Optional[TreeEvidenceRecord]:
        """Get evidence by image_id stored in metadata."""
        from src.repository.entityModels import TreeEvidence

        try:
            query = select(TreeEvidence).where(
                TreeEvidence.tree_metadata["image_id"].astext == image_id
            )
            evidence = self.session.execute(query).scalar_one_or_none()

            if evidence is None:
                return None

            vector = evidence.global_vector
            if vector:
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()
                elif isinstance(vector, str):
                    vector = json.loads(vector)
            else:
                vector = []

            return TreeEvidenceRecord(
                id=_coerce_row_evidence_id(evidence.id),
                tree_id=evidence.tree_id,
                region_code=evidence.region_code,
                global_vector=vector,
                camera_heading=evidence.camera_heading,
                camera_pitch=evidence.camera_pitch,
                storage_cid=evidence.storage_cid,
                evidence_hash=evidence.evidence_hash,
                is_c2pa_verified=evidence.is_c2pa_verified,
                raw_telemetry=evidence.raw_telemetry if isinstance(evidence.raw_telemetry, dict) else json.loads(evidence.raw_telemetry) if evidence.raw_telemetry else {},
                captured_at=evidence.captured_at,
                metadata=_coerce_metadata_dict(evidence.tree_metadata),
            )

        except Exception as e:
            logger.error(f"Error getting evidence by image_id {image_id}: {e}")
            return None

    def get_global_vector_by_id(self, evidence_id: str) -> Optional[np.ndarray]:
        """Get global vector by evidence primary key (UUID string).

        Preferred over ``get_global_vector_by_image_id`` when you already have the
        evidence UUID from a vector search result.
        """
        from src.repository.entityModels import TreeEvidence

        try:
            result = self.session.get(TreeEvidence, str(evidence_id))
            if result is None or result.global_vector is None:
                return None

            v = result.global_vector
            if hasattr(v, 'tolist'):
                return np.array(v.tolist())
            elif isinstance(v, str):
                return np.array(json.loads(v))
            return np.array(v)

        except Exception as e:
            logger.error(f"Error getting global vector by id {evidence_id}: {e}")
            return None

    def get_global_vector_by_image_id(self, image_id: str) -> Optional[np.ndarray]:
        """Get global vector by image_id stored in metadata.

        Use ``get_global_vector_by_id`` when you have the evidence UUID.
        This method is for legacy lookups by the original image slug.
        """
        from src.repository.entityModels import TreeEvidence

        try:
            query = select(TreeEvidence.global_vector).where(
                TreeEvidence.tree_metadata["image_id"].astext == image_id
            )
            result = self.session.execute(query).scalar_one_or_none()

            if result is None:
                return None

            if hasattr(result, 'tolist'):
                return np.array(result.tolist())
            elif isinstance(result, str):
                return np.array(json.loads(result))
            else:
                return np.array(result)

        except Exception as e:
            logger.error(f"Error getting global vector by image_id {image_id}: {e}")
            return None

    def get_evidences_by_tree(
        self,
        tree_id: str,
        limit: int = 100,
    ) -> List[TreeEvidenceRecord]:
        """Get all evidences for a tree."""
        from src.repository.entityModels import TreeEvidence

        try:
            query = (
                select(TreeEvidence)
                .where(TreeEvidence.tree_id == tree_id)
                .order_by(TreeEvidence.captured_at.desc())
                .limit(limit)
            )
            evidences = self.session.execute(query).scalars().all()

            records = []
            for evidence in evidences:
                vector = evidence.global_vector
                if vector:
                    if hasattr(vector, 'tolist'):
                        vector = vector.tolist()
                    elif isinstance(vector, str):
                        vector = json.loads(vector)
                else:
                    vector = []

                records.append(TreeEvidenceRecord(
                    id=_coerce_row_evidence_id(evidence.id),
                    tree_id=evidence.tree_id,
                    region_code=evidence.region_code,
                    global_vector=vector,
                    camera_heading=evidence.camera_heading,
                    camera_pitch=evidence.camera_pitch,
                    storage_cid=evidence.storage_cid,
                    evidence_hash=evidence.evidence_hash,
                    is_c2pa_verified=evidence.is_c2pa_verified,
                    raw_telemetry=evidence.raw_telemetry if isinstance(evidence.raw_telemetry, dict) else json.loads(evidence.raw_telemetry) if evidence.raw_telemetry else {},
                    captured_at=evidence.captured_at,
                    metadata=_coerce_metadata_dict(evidence.tree_metadata),
                ))

            return records

        except Exception as e:
            logger.error(f"Error getting evidences for tree {tree_id}: {e}")
            return []

    def delete_evidence(self, evidence_id: uuid.UUID) -> bool:
        """Delete an evidence record."""
        from src.repository.entityModels import TreeEvidence

        try:
            evidence = self.session.get(TreeEvidence, str(evidence_id))
            if evidence:
                self.session.delete(evidence)
                logger.info(f"Deleted evidence: {evidence_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting evidence {evidence_id}: {e}")
            raise

    # ==================== VECTOR SIMILARITY SEARCH ====================

    def _get_vector_dimension(self) -> int:
        """Get PostgreSQL vector dimension for halfvec queries.

        The halfvec column is ``halfvec(384)`` to match dinov3-vitb16-pretrain-lvd1689m (384 dims).
        This must match ``postgres_vector_dim`` / ``POSTGRES_VECTOR_DIM``.
        """
        try:
            from src.config.appConfig import get_config
            return get_config().postgres_vector_dim
        except Exception:
            return 384

    def search_similar_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        tree_id_filter: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors using pgvector cosine similarity."""
        from src.repository.entityModels import TreeEvidence

        vector_dim = self._get_vector_dimension()
        if len(query_vector) != vector_dim:
            logger.error(f"Vector dimension mismatch: expected {vector_dim}, got {len(query_vector)}")
            return [VectorSearchResult(
                evidence_id=uuid.uuid4(),
                tree_id="",
                similarity=0.0,
                captured_at=datetime.now(),
                metadata={},
                error=f"Vector dimension mismatch: expected {vector_dim}, got {len(query_vector)}"
            )]

        try:
            if not HAS_PGVECTOR:
                logger.error("pgvector not available")
                return []

            # Build query using pgvector cosine distance
            # Note: Use raw SQL for vector operations as ORM support is limited
            from sqlalchemy import text

            vector_str = f"[{','.join(map(str, query_vector))}]"

            # NOTE: dimension MUST be interpolated as a literal — halfvec(N) requires
            # a compile-time integer constant, NOT a bound parameter (:dim placeholder).
            _dim = int(vector_dim)
            if not (1 <= _dim <= 65535):
                raise ValueError(
                    f"vector_dim must be between 1 and 65535, got {_dim}"
                )
            sql = f"""
                SELECT
                    id, tree_id,
                    1 - (global_vector <=> CAST(:query_vector AS halfvec({_dim}))) AS similarity,
                    captured_at, metadata
                FROM tree_evidences
                WHERE global_vector IS NOT NULL
            """

            params = {"query_vector": vector_str}

            if tree_id_filter:
                sql += " AND tree_id = :tree_id"
                params["tree_id"] = tree_id_filter

            sql += f"""
                ORDER BY global_vector <=> CAST(:query_vector AS halfvec({_dim}))
                LIMIT :top_k
            """
            params["top_k"] = top_k

            result = self.session.execute(text(sql), params)
            rows = result.fetchall()

            return [
                VectorSearchResult(
                    evidence_id=_coerce_row_evidence_id(row[0]),
                    tree_id=row[1],
                    similarity=float(row[2]),
                    captured_at=row[3],
                    metadata=row[4] if isinstance(row[4], dict) else json.loads(row[4]) if row[4] else {},
                )
                for row in rows
                if row[2] >= min_similarity
            ]

        except Exception as e:
            _log_pgvector_halfvec_dim_mismatch(e, int(vector_dim))
            logger.error(f"Error searching vectors: {e}")
            return [VectorSearchResult(
                evidence_id=uuid.uuid4(),
                tree_id="",
                similarity=0.0,
                captured_at=datetime.now(),
                metadata={},
                error=str(e)
            )]

    def search_grid_first_spatial_fallback(
        self,
        query_vector: List[float],
        farm_id: str,
        top_k: int = 10,
        row_idx: Optional[int] = None,
        col_idx: Optional[int] = None,
        longitude: Optional[float] = None,
        latitude: Optional[float] = None,
        radius_meters: float = 25.0,
        tree_id_filter: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[VectorSearchResult]:
        """Grid-first retrieval with spatial fallback, then exact cosine (<=>) ranking.

        Flow:
        1) Try trees in the same farm/grid cell (farm_id + row_idx + col_idx)
        2) If grid has no candidates, fallback to trees within ``radius_meters``
        3) Rank candidate evidences by exact pgvector cosine distance via ``<=>``
        """
        from sqlalchemy import text

        vector_dim = self._get_vector_dimension()
        if len(query_vector) != vector_dim:
            logger.error(f"Vector dimension mismatch: expected {vector_dim}, got {len(query_vector)}")
            return [VectorSearchResult(
                evidence_id=uuid.uuid4(),
                tree_id="",
                similarity=0.0,
                captured_at=datetime.now(),
                metadata={},
                error=f"Vector dimension mismatch: expected {vector_dim}, got {len(query_vector)}"
            )]

        use_grid = row_idx is not None and col_idx is not None
        use_spatial = longitude is not None and latitude is not None and radius_meters > 0

        if not use_grid and not use_spatial:
            logger.error(
                "Grid-first search requires either grid coordinates "
                "(row_idx + col_idx) or spatial fallback inputs (longitude + latitude + radius)."
            )
            return [VectorSearchResult(
                evidence_id=uuid.uuid4(),
                tree_id="",
                similarity=0.0,
                captured_at=datetime.now(),
                metadata={},
                error=(
                    "Missing filters: provide (row_idx, col_idx) "
                    "or (longitude, latitude, radius_meters)"
                ),
            )]

        try:
            vector_str = f"[{','.join(map(str, query_vector))}]"
            _dim = int(vector_dim)
            if not (1 <= _dim <= 65535):
                raise ValueError(
                    f"vector_dim must be between 1 and 65535, got {_dim}"
                )

            sql = f"""
                WITH grid_candidates AS (
                    SELECT t.id
                    FROM trees t
                    WHERE t.farm_id = :farm_id
                      AND :use_grid
                      AND t.row_idx = :row_idx
                      AND t.col_idx = :col_idx
                ),
                fallback_candidates AS (
                    SELECT t.id
                    FROM trees t
                    WHERE t.farm_id = :farm_id
                      AND :use_spatial
                      AND t.location IS NOT NULL
                      AND ST_DWithin(
                            t.location,
                            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                            :radius_meters
                      )
                ),
                candidate_trees AS (
                    SELECT id FROM grid_candidates
                    UNION ALL
                    SELECT id
                    FROM fallback_candidates
                    WHERE NOT EXISTS (SELECT 1 FROM grid_candidates)
                )
                SELECT
                    e.id,
                    e.tree_id,
                    1 - (e.global_vector <=> CAST(:query_vector AS halfvec({_dim}))) AS similarity,
                    e.captured_at,
                    e.metadata
                FROM tree_evidences e
                JOIN candidate_trees c ON c.id = e.tree_id
                WHERE e.global_vector IS NOT NULL
                  AND (:tree_id_filter IS NULL OR e.tree_id = :tree_id_filter)
                ORDER BY e.global_vector <=> CAST(:query_vector AS halfvec({_dim}))
                LIMIT :top_k
            """

            params = {
                "farm_id": farm_id,
                "use_grid": use_grid,
                "row_idx": row_idx,
                "col_idx": col_idx,
                "use_spatial": use_spatial,
                "lon": longitude,
                "lat": latitude,
                "radius_meters": radius_meters,
                "tree_id_filter": tree_id_filter,
                "query_vector": vector_str,
                "top_k": top_k,
            }

            result = self.session.execute(text(sql), params)
            rows = result.fetchall()

            return [
                VectorSearchResult(
                    evidence_id=_coerce_row_evidence_id(row[0]),
                    tree_id=row[1],
                    similarity=float(row[2]),
                    captured_at=row[3],
                    metadata=row[4] if isinstance(row[4], dict) else json.loads(row[4]) if row[4] else {},
                )
                for row in rows
                if row[2] >= min_similarity
            ]

        except Exception as e:
            _log_pgvector_halfvec_dim_mismatch(e, int(vector_dim))
            logger.error(f"Error in grid-first spatial-fallback search: {e}")
            return [VectorSearchResult(
                evidence_id=uuid.uuid4(),
                tree_id="",
                similarity=0.0,
                captured_at=datetime.now(),
                metadata={},
                error=str(e),
            )]

    # ==================== MILVUS ADAPTER METHODS ====================
    # These methods provide a Milvus-compatible interface over PostgreSQL/pgvector
    # so HierarchicalMatchingService can use PostgreSQL as the vector store.

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 50,
        tree_id_filter: Optional[str] = None,
    ):
        """Adapter: Milvus-compatible search returning MilvusResult.

        Mirrors MilvusRepository.search() interface so HierarchicalMatchingService
        works transparently with PostgreSQL.
        """
        from src.repository.milvusRepository import MilvusResult

        vec_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
        results = self.search_similar_vectors(
            query_vector=vec_list,
            top_k=top_k,
            tree_id_filter=tree_id_filter,
            min_similarity=0.0,
        )

        ids = [str(r.evidence_id) for r in results]
        similarities = [r.similarity for r in results]
        tree_ids = [r.tree_id for r in results]
        metadatas = [r.metadata for r in results]

        return MilvusResult(
            ids=ids,
            distances=similarities,  # pgvector cosine distance = similarity here
            tree_ids=tree_ids,
            metadatas=metadatas,
        )

    def search_with_bounding_box(
        self,
        query_vector: np.ndarray,
        top_k: int = 50,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        hor_angle_min: Optional[float] = None,
        hor_angle_max: Optional[float] = None,
        ver_angle_min: Optional[float] = None,
        ver_angle_max: Optional[float] = None,
        pitch_min: Optional[float] = None,
        pitch_max: Optional[float] = None,
        captured_at_min: Optional[int] = None,
        captured_at_max: Optional[int] = None,
        tree_id_filter: Optional[str] = None,
    ):
        """Adapter: Milvus-compatible search with bounding-box filters returning MilvusResult.

        Mirrors MilvusRepository.search_with_bounding_box() interface so
        HierarchicalMatchingService works transparently with PostgreSQL.
        """
        from src.repository.milvusRepository import MilvusResult

        # Derive centroid lat/lon from bounding box
        latitude = (lat_min + lat_max) / 2 if lat_min is not None and lat_max is not None else None
        longitude = (lon_min + lon_max) / 2 if lon_min is not None and lon_max is not None else None
        # Approximate radius from bounding box diagonal
        import math
        radius = None
        if lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None:
            lat_span = abs(lat_max - lat_min)
            lon_span = abs(lon_max - lon_min)
            # Approximate: use the larger span in degrees * 111km/degree
            max_span_deg = max(lat_span, lon_span * math.cos(math.radians(latitude or 0)))
            radius = max_span_deg * 111320  # metres

        # Use camera heading as hor_angle, pitch as ver_angle
        vec_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
        results = self.search_similar_vectors_with_filters(
            query_vector=vec_list,
            top_k=top_k,
            longitude=longitude,
            latitude=latitude,
            radius_meters=radius,
            camera_heading=int((hor_angle_min + hor_angle_max) / 2) if hor_angle_min is not None and hor_angle_max is not None else None,
            camera_pitch=int((ver_angle_min + ver_angle_max) / 2) if ver_angle_min is not None and ver_angle_max is not None else None,
            heading_tolerance=int((hor_angle_max - hor_angle_min) / 2) if hor_angle_min is not None and hor_angle_max is not None else 30,
            pitch_tolerance=int((ver_angle_max - ver_angle_min) / 2) if ver_angle_min is not None and ver_angle_max is not None else 15,
            tree_id_filter=tree_id_filter,
        )

        ids = [str(r.evidence_id) for r in results]
        similarities = [r.similarity for r in results]
        tree_ids = [r.tree_id for r in results]
        metadatas = [r.metadata for r in results]

        return MilvusResult(
            ids=ids,
            distances=similarities,
            tree_ids=tree_ids,
            metadatas=metadatas,
        )

    # ==================== GEOSPATIAL QUERIES ====================

    def find_trees_nearby(
        self,
        longitude: float,
        latitude: float,
        radius_meters: float = 1000,
        limit: int = 50,
    ) -> List[GeoSearchResult]:
        """Find trees within a radius using PostGIS."""
        from src.repository.entityModels import Tree
        from sqlalchemy import text

        try:
            if not HAS_GEOALCHEMY:
                logger.error("GeoAlchemy2 not available")
                return []

            sql = """
                SELECT
                    id,
                    ST_Distance(
                        location,
                        ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
                    ) AS distance_meters,
                    location
                FROM trees
                WHERE location IS NOT NULL
                  AND ST_DWithin(
                      location,
                      ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                      :radius
                  )
                ORDER BY distance_meters
                LIMIT :limit
            """

            result = self.session.execute(text(sql), {
                "lon": longitude,
                "lat": latitude,
                "radius": radius_meters,
                "limit": limit
            })
            rows = result.fetchall()

            return [
                GeoSearchResult(
                    tree_id=row[0],
                    distance_meters=float(row[1]),
                    location=row[2],
                )
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Error finding nearby trees: {e}")
            return []

    def get_tree_location(self, tree_id: str) -> Optional[Tuple[float, float]]:
        """Get GPS coordinates for a tree."""
        from src.repository.entityModels import Tree
        from sqlalchemy import text

        try:
            sql = """
                SELECT ST_X(location::geometry), ST_Y(location::geometry)
                FROM trees
                WHERE id = :tree_id AND location IS NOT NULL
            """

            result = self.session.execute(text(sql), {"tree_id": tree_id})
            row = result.fetchone()

            if row:
                return (float(row[0]), float(row[1]))
            return None

        except Exception as e:
            logger.error(f"Error getting tree location: {e}")
            return None

    # ==================== STATISTICS ====================

    def get_tree_count(self, farm_id: Optional[str] = None) -> int:
        """Get total tree count."""
        from src.repository.entityModels import Tree
        from sqlalchemy import func, select

        try:
            query = select(func.count(Tree.id))
            if farm_id:
                query = query.where(Tree.farm_id == farm_id)
            return self.session.execute(query).scalar() or 0

        except Exception as e:
            logger.error(f"Error getting tree count: {e}")
            return 0

    def get_evidence_count(self, tree_id: Optional[str] = None) -> int:
        """Get total evidence count."""
        from src.repository.entityModels import TreeEvidence
        from sqlalchemy import func, select

        try:
            query = select(func.count(TreeEvidence.id))
            if tree_id:
                query = query.where(TreeEvidence.tree_id == tree_id)
            return self.session.execute(query).scalar() or 0

        except Exception as e:
            logger.error(f"Error getting evidence count: {e}")
            return 0

    # ==================== VECTOR SEARCH WITH FILTERS ====================

    def search_similar_vectors_with_filters(
        self,
        query_vector: List[float],
        top_k: int = 10,
        longitude: Optional[float] = None,
        latitude: Optional[float] = None,
        radius_meters: Optional[float] = None,
        camera_heading: Optional[int] = None,
        camera_pitch: Optional[int] = None,
        heading_tolerance: int = 30,
        pitch_tolerance: int = 15,
        tree_id_filter: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Search with multiple filters combining vector similarity with geo/angle constraints."""
        from sqlalchemy import text

        vector_dim = self._get_vector_dimension()
        if len(query_vector) != vector_dim:
            logger.error(f"Vector dimension mismatch: expected {vector_dim}, got {len(query_vector)}")
            return [VectorSearchResult(
                evidence_id=uuid.uuid4(),
                tree_id="",
                similarity=0.0,
                captured_at=datetime.now(),
                metadata={},
                error=f"Vector dimension mismatch: expected {vector_dim}, got {len(query_vector)}"
            )]

        try:
            vector_str = f"[{','.join(map(str, query_vector))}]"

            conditions = ["global_vector IS NOT NULL"]
            params: Dict[str, Any] = {"top_k": top_k, "query_vector": vector_str}

            # Geo filter
            if longitude is not None and latitude is not None and radius_meters is not None:
                conditions.append("""
                    tree_id IN (
                        SELECT id FROM trees
                        WHERE location IS NOT NULL
                        AND ST_DWithin(
                            location,
                            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                            :radius
                        )
                    )
                """)
                params["lon"] = longitude
                params["lat"] = latitude
                params["radius"] = radius_meters

            # Angle filters
            if camera_heading is not None:
                conditions.append("""
                    camera_heading IS NOT NULL
                    AND ABS(camera_heading - :heading) <= :heading_tol
                """)
                params["heading"] = camera_heading
                params["heading_tol"] = heading_tolerance

            if camera_pitch is not None:
                conditions.append("""
                    camera_pitch IS NOT NULL
                    AND ABS(camera_pitch - :pitch) <= :pitch_tol
                """)
                params["pitch"] = camera_pitch
                params["pitch_tol"] = pitch_tolerance

            # Tree ID filter
            if tree_id_filter:
                conditions.append("tree_id = :tree_id")
                params["tree_id"] = tree_id_filter

            where_clause = "WHERE " + " AND ".join(conditions)

            # NOTE: dimension MUST be interpolated as a literal — halfvec(N) requires
            # a compile-time integer constant, NOT a bound parameter (:dim placeholder).
            _dim = int(vector_dim)
            if not (1 <= _dim <= 65535):
                raise ValueError(
                    f"vector_dim must be between 1 and 65535, got {_dim}"
                )
            sql = f"""
                SELECT
                    id, tree_id,
                    1 - (global_vector <=> CAST(:query_vector AS halfvec({_dim}))) AS similarity,
                    captured_at, metadata
                FROM tree_evidences
                {where_clause}
                ORDER BY global_vector <=> CAST(:query_vector AS halfvec({_dim}))
                LIMIT :top_k
            """

            result = self.session.execute(text(sql), params)
            rows = result.fetchall()

            return [
                VectorSearchResult(
                    evidence_id=_coerce_row_evidence_id(row[0]),
                    tree_id=row[1],
                    similarity=float(row[2]),
                    captured_at=row[3],
                    metadata=row[4] if isinstance(row[4], dict) else json.loads(row[4]) if row[4] else {},
                )
                for row in rows
            ]

        except Exception as e:
            _log_pgvector_halfvec_dim_mismatch(e, int(vector_dim))
            logger.error(f"Error in filtered vector search: {e}")
            return []

    def search_with_filters(
        self,
        query_vector: List[float],
        top_k: int = 10,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        geo_radius_meters: Optional[float] = None,
        geo_latitude: Optional[float] = None,
        geo_longitude: Optional[float] = None,
        hor_angle_min: Optional[float] = None,
        hor_angle_max: Optional[float] = None,
        ver_angle_min: Optional[float] = None,
        ver_angle_max: Optional[float] = None,
        pitch_min: Optional[float] = None,
        pitch_max: Optional[float] = None,
        captured_at_min: Optional[int] = None,
        captured_at_max: Optional[int] = None,
        tree_id_filter: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors with geo, angle, and time filters."""
        from sqlalchemy import text

        vector_dim = self._get_vector_dimension()
        if len(query_vector) != vector_dim:
            return [VectorSearchResult(
                evidence_id=uuid.uuid4(),
                tree_id="",
                similarity=0.0,
                captured_at=datetime.now(),
                metadata={},
                error=f"Vector dimension mismatch"
            )]

        try:
            vector_str = f"[{','.join(map(str, query_vector))}]"

            conditions = ["global_vector IS NOT NULL"]
            params: Dict[str, Any] = {"top_k": top_k, "query_vector": vector_str}

            # Geo filter - Support both bounding box AND radius-based search
            has_bbox_geo = (lon_min is not None and lon_max is not None and
                          lat_min is not None and lat_max is not None)
            has_radius_geo = (geo_radius_meters is not None and
                            geo_latitude is not None and geo_longitude is not None)

            if has_radius_geo:
                conditions.append("""
                    location IS NOT NULL
                    AND ST_DWithin(
                        location::geography,
                        ST_SetSRID(ST_MakePoint(:geo_longitude, :geo_latitude), 4326)::geography,
                        :geo_radius_meters
                    )
                """)
                params["geo_radius_meters"] = geo_radius_meters
                params["geo_latitude"] = geo_latitude
                params["geo_longitude"] = geo_longitude
            elif has_bbox_geo:
                conditions.append("""
                    location IS NOT NULL
                    AND ST_X(location::geometry) BETWEEN :lon_min AND :lon_max
                    AND ST_Y(location::geometry) BETWEEN :lat_min AND :lat_max
                """)
                params["lon_min"] = lon_min
                params["lon_max"] = lon_max
                params["lat_min"] = lat_min
                params["lat_max"] = lat_max

            # Horizontal angle (heading) filter
            if hor_angle_min is not None and hor_angle_max is not None:
                conditions.append("""
                    camera_heading IS NOT NULL
                    AND camera_heading BETWEEN :hor_angle_min AND :hor_angle_max
                """)
                params["hor_angle_min"] = hor_angle_min
                params["hor_angle_max"] = hor_angle_max

            # Vertical angle filter
            if ver_angle_min is not None and ver_angle_max is not None:
                conditions.append("""
                    raw_telemetry ? 'ver_angle'
                    AND (raw_telemetry->>'ver_angle')::float BETWEEN :ver_angle_min AND :ver_angle_max
                """)
                params["ver_angle_min"] = ver_angle_min
                params["ver_angle_max"] = ver_angle_max

            # Pitch filter
            if pitch_min is not None and pitch_max is not None:
                conditions.append("""
                    camera_pitch IS NOT NULL
                    AND camera_pitch BETWEEN :pitch_min AND :pitch_max
                """)
                params["pitch_min"] = pitch_min
                params["pitch_max"] = pitch_max

            # Time filter
            if captured_at_min is not None and captured_at_max is not None:
                conditions.append("""
                    captured_at IS NOT NULL
                    AND captured_at BETWEEN :captured_at_min AND :captured_at_max
                """)
                params["captured_at_min"] = datetime.fromtimestamp(captured_at_min, tz=timezone.utc)
                params["captured_at_max"] = datetime.fromtimestamp(captured_at_max, tz=timezone.utc)

            # Tree ID filter
            if tree_id_filter:
                conditions.append("tree_id = :tree_id_filter")
                params["tree_id_filter"] = tree_id_filter

            where_clause = "WHERE " + " AND ".join(conditions)

            # NOTE: dimension MUST be interpolated as a literal — halfvec(N) requires
            # a compile-time integer constant, NOT a bound parameter (:dim placeholder).
            _dim = int(vector_dim)
            if not (1 <= _dim <= 65535):
                raise ValueError(
                    f"vector_dim must be between 1 and 65535, got {_dim}"
                )
            sql = f"""
                SELECT
                    id, tree_id,
                    1 - (global_vector <=> CAST(:query_vector AS halfvec({_dim}))) AS similarity,
                    captured_at, metadata,
                    ST_X(location::geometry) as longitude,
                    ST_Y(location::geometry) as latitude,
                    camera_heading, camera_pitch
                FROM tree_evidences
                {where_clause}
                ORDER BY global_vector <=> CAST(:query_vector AS halfvec({_dim}))
                LIMIT :top_k
            """

            result = self.session.execute(text(sql), params)
            rows = result.fetchall()

            return [
                VectorSearchResult(
                    evidence_id=_coerce_row_evidence_id(row[0]),
                    tree_id=row[1],
                    similarity=float(row[2]),
                    captured_at=row[3],
                    metadata=row[4] if isinstance(row[4], dict) else json.loads(row[4]) if row[4] else {},
                )
                for row in rows
            ]

        except Exception as e:
            _log_pgvector_halfvec_dim_mismatch(e, int(vector_dim))
            logger.error(f"Error in search_with_filters: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [VectorSearchResult(
                evidence_id=uuid.uuid4(),
                tree_id="",
                similarity=0.0,
                captured_at=datetime.now(),
                metadata={},
                error=str(e)
            )]

    def search_with_geo_radius(
        self,
        query_vector: List[float],
        radius_meters: float,
        latitude: float,
        longitude: float,
        top_k: int = 10,
        hor_angle_min: Optional[float] = None,
        hor_angle_max: Optional[float] = None,
        pitch_min: Optional[float] = None,
        pitch_max: Optional[float] = None,
        ver_angle_min: Optional[float] = None,
        ver_angle_max: Optional[float] = None,
        captured_at_min: Optional[int] = None,
        captured_at_max: Optional[int] = None,
        tree_id_filter: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors using PostGIS ST_DWithin for accurate radius-based geo filtering."""
        return self.search_with_filters(
            query_vector=query_vector,
            top_k=top_k,
            geo_radius_meters=radius_meters,
            geo_latitude=latitude,
            geo_longitude=longitude,
            hor_angle_min=hor_angle_min,
            hor_angle_max=hor_angle_max,
            ver_angle_min=ver_angle_min,
            ver_angle_max=ver_angle_max,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            captured_at_min=captured_at_min,
            captured_at_max=captured_at_max,
            tree_id_filter=tree_id_filter,
        )

    def search_with_combined_filters(
        self,
        query_vector: List[float],
        top_k: int = 10,
        radius_meters: Optional[float] = None,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        hor_angle_min: Optional[float] = None,
        hor_angle_max: Optional[float] = None,
        ver_angle_min: Optional[float] = None,
        ver_angle_max: Optional[float] = None,
        pitch_min: Optional[float] = None,
        pitch_max: Optional[float] = None,
        captured_at_min: Optional[int] = None,
        captured_at_max: Optional[int] = None,
        tree_id_filter: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Unified search method supporting all filter types."""
        has_radius = radius_meters is not None and center_lat is not None and center_lon is not None
        has_bbox = all(v is not None for v in [lat_min, lat_max, lon_min, lon_max])

        if has_radius:
            return self.search_with_geo_radius(
                query_vector=query_vector,
                radius_meters=radius_meters,
                latitude=center_lat,
                longitude=center_lon,
                top_k=top_k,
                hor_angle_min=hor_angle_min,
                hor_angle_max=hor_angle_max,
                ver_angle_min=ver_angle_min,
                ver_angle_max=ver_angle_max,
                pitch_min=pitch_min,
                pitch_max=pitch_max,
                captured_at_min=captured_at_min,
                captured_at_max=captured_at_max,
                tree_id_filter=tree_id_filter,
            )
        elif has_bbox:
            return self.search_with_filters(
                query_vector=query_vector,
                top_k=top_k,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                hor_angle_min=hor_angle_min,
                hor_angle_max=hor_angle_max,
                ver_angle_min=ver_angle_min,
                ver_angle_max=ver_angle_max,
                pitch_min=pitch_min,
                pitch_max=pitch_max,
                captured_at_min=captured_at_min,
                captured_at_max=captured_at_max,
                tree_id_filter=tree_id_filter,
            )
        else:
            return self.search_with_filters(
                query_vector=query_vector,
                top_k=top_k,
                hor_angle_min=hor_angle_min,
                hor_angle_max=hor_angle_max,
                ver_angle_min=ver_angle_min,
                ver_angle_max=ver_angle_max,
                pitch_min=pitch_min,
                pitch_max=pitch_max,
                captured_at_min=captured_at_min,
                captured_at_max=captured_at_max,
                tree_id_filter=tree_id_filter,
            )

    def find_evidences_nearby(
        self,
        longitude: float,
        latitude: float,
        radius_meters: float = 1000,
        limit: int = 50,
    ) -> List[VectorSearchResult]:
        """Find evidences from trees within a radius."""
        from sqlalchemy import text

        try:
            sql = """
                SELECT
                    e.id, e.tree_id,
                    1.0 AS similarity,
                    e.captured_at, e.metadata
                FROM tree_evidences e
                INNER JOIN trees t ON e.tree_id = t.id
                WHERE t.location IS NOT NULL
                  AND ST_DWithin(
                      t.location,
                      ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                      :radius
                  )
                ORDER BY e.captured_at DESC
                LIMIT :limit
            """

            result = self.session.execute(text(sql), {
                "lon": longitude,
                "lat": latitude,
                "radius": radius_meters,
                "limit": limit
            })
            rows = result.fetchall()

            return [
                VectorSearchResult(
                    evidence_id=_coerce_row_evidence_id(row[0]),
                    tree_id=row[1],
                    similarity=float(row[2]),
                    captured_at=row[3],
                    metadata=row[4] if isinstance(row[4], dict) else json.loads(row[4]) if row[4] else {},
                )
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Error finding nearby evidences: {e}")
            return []


# Factory functions
def create_sqlalchemy_repository(session: Optional[Session] = None) -> "SQLAlchemyORMRepository":
    """Create SQLAlchemy ORM repository."""
    return SQLAlchemyORMRepository(session)


def get_sqlalchemy_repository() -> SQLAlchemyORMRepository:
    """Get SQLAlchemy repository with a managed session.

    IMPORTANT: Use as a context manager or call repo.session.close() when done:
        repo = get_sqlalchemy_repository()
        try:
            repo.do_something()
            repo.session.commit()
        except Exception:
            repo.session.rollback()
            raise
        finally:
            repo.close()
    """
    from src.repository.databaseManager import get_session
    session = get_session()
    return SQLAlchemyORMRepository(session)