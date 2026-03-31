#!/usr/bin/env python3
"""
Trees API router for SAM3.

Implements a full REST resource for the trees table with CRUD operations
and an evidences sub-resource. All responses are wrapped in ApiEnvelope.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query

from src.api.helpers import _envelope_json_response, convert_numpy_types
from src.dto.tree import (
    TreeCreateRequest,
    TreeUpdateRequest,
    TreePatchRequest,
    TreeResponse,
    TreeListData,
    TreeEvidenceListData,
    TreeEvidenceResponse,
)
from src.repository.sqlalchemyRepository import (
    get_sqlalchemy_repository,
    SQLAlchemyORMRepository,
)

router = APIRouter(prefix="/trees", tags=["Trees"])


# =============================================================================
# Dependencies
# =============================================================================


def get_repo() -> SQLAlchemyORMRepository:
    """Dependency that yields a SQLAlchemy repository instance."""
    return get_sqlalchemy_repository()


# =============================================================================
# Helpers
# =============================================================================


def _build_envelope(data: Any, message: str = "OK", status_code: int = 200):
    """Wrap response data in an ApiEnvelope JSONResponse."""
    if hasattr(data, "model_dump"):
        data = convert_numpy_types(data.model_dump(mode="json"))
    return _envelope_json_response(status_code, message, error=None, data=data)


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "",
    summary="List trees",
    description="List trees with optional filters.",
)
def list_trees(
    farm_id: Optional[str] = Query(None, description="Filter by farm identifier"),
    region_code: Optional[str] = Query(None, description="Filter by region code"),
    row_idx: Optional[int] = Query(None, ge=0, description="Filter by grid row index"),
    col_idx: Optional[int] = Query(None, ge=0, description="Filter by grid column index"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of trees to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    repo: SQLAlchemyORMRepository = Depends(get_repo),
):
    """List trees with optional filters."""
    records = repo.list_trees(
        farm_id=farm_id,
        region_code=region_code,
        row_idx=row_idx,
        col_idx=col_idx,
        limit=limit,
        offset=offset,
    )
    items = [TreeResponse.from_record(r) for r in records]
    list_data = TreeListData(items=items, limit=limit, offset=offset)
    return _build_envelope(list_data, message="Trees retrieved", status_code=200)


@router.post(
    "",
    summary="Create a tree",
    description="Create a new tree record.",
    status_code=201,
)
def create_tree(
    body: TreeCreateRequest,
    repo: SQLAlchemyORMRepository = Depends(get_repo),
):
    """Create a new tree."""
    # Conflict if tree already exists
    existing = repo.get_tree(body.id)
    if existing is not None:
        return _envelope_json_response(
            409,
            f"Tree '{body.id}' already exists",
            error=f"Tree '{body.id}' already exists",
            data=None,
        )

    repo.create_tree(
        tree_id=body.id,
        region_code=body.region_code,
        farm_id=body.farm_id,
        geohash_7=body.geohash_7,
        row_idx=body.row_idx,
        col_idx=body.col_idx,
        longitude=body.longitude,
        latitude=body.latitude,
        codebook_id=body.codebook_id,
        metadata=body.metadata,
        captured_at=body.captured_at,
    )
    repo.session.commit()

    record = repo.get_tree(body.id)
    response = TreeResponse.from_record(record)
    return _build_envelope(response, message="Tree created", status_code=201)


@router.get(
    "/{tree_id}",
    summary="Get a tree",
    description="Get a tree by its identifier.",
)
def get_tree(
    tree_id: str,
    repo: SQLAlchemyORMRepository = Depends(get_repo),
):
    """Get a tree by ID."""
    record = repo.get_tree(tree_id)
    if record is None:
        return _envelope_json_response(
            404,
            f"Tree '{tree_id}' not found",
            error=f"Tree '{tree_id}' not found",
            data=None,
        )
    response = TreeResponse.from_record(record)
    return _build_envelope(response, message="Tree retrieved", status_code=200)


@router.put(
    "/{tree_id}",
    summary="Full replace a tree",
    description="Replace all fields of an existing tree.",
)
def replace_tree(
    tree_id: str,
    body: TreeUpdateRequest,
    repo: SQLAlchemyORMRepository = Depends(get_repo),
):
    """Full replace of a tree record."""
    # 404 if not found
    existing = repo.get_tree(tree_id)
    if existing is None:
        return _envelope_json_response(
            404,
            f"Tree '{tree_id}' not found",
            error=f"Tree '{tree_id}' not found",
            data=None,
        )

    repo.create_tree(
        tree_id=tree_id,
        region_code=body.region_code,
        farm_id=body.farm_id,
        geohash_7=body.geohash_7,
        row_idx=body.row_idx,
        col_idx=body.col_idx,
        longitude=body.longitude,
        latitude=body.latitude,
        codebook_id=body.codebook_id,
        metadata=body.metadata,
        captured_at=body.captured_at,
    )
    repo.session.commit()

    record = repo.get_tree(tree_id)
    response = TreeResponse.from_record(record)
    return _build_envelope(response, message="Tree replaced", status_code=200)


@router.patch(
    "/{tree_id}",
    summary="Partial update a tree",
    description="Update only the provided fields of a tree.",
)
def patch_tree(
    tree_id: str,
    body: TreePatchRequest,
    repo: SQLAlchemyORMRepository = Depends(get_repo),
):
    """Partial update of a tree record."""
    # Load current state
    current = repo.get_tree(tree_id)
    if current is None:
        return _envelope_json_response(
            404,
            f"Tree '{tree_id}' not found",
            error=f"Tree '{tree_id}' not found",
            data=None,
        )

    # Merge: only non-None fields from body override current values
    updates: Dict[str, Any] = {}

    for field_name in (
        "region_code",
        "farm_id",
        "geohash_7",
        "row_idx",
        "col_idx",
        "codebook_id",
        "captured_at",
        "metadata",
    ):
        body_val = getattr(body, field_name, None)
        if body_val is not None:
            updates[field_name] = body_val

    # Handle latitude/longitude pair — only include if both are set
    lat = body.latitude
    lon = body.longitude
    if lat is not None and lon is not None:
        updates["latitude"] = lat
        updates["longitude"] = lon

    if not updates:
        # No meaningful changes — return current state
        response = TreeResponse.from_record(current)
        return _build_envelope(response, message="Tree unchanged", status_code=200)

    repo.update_tree(tree_id, **updates)
    repo.session.commit()

    record = repo.get_tree(tree_id)
    response = TreeResponse.from_record(record)
    return _build_envelope(response, message="Tree updated", status_code=200)


@router.delete(
    "/{tree_id}",
    summary="Delete a tree",
    description="Delete a tree and all its evidences.",
)
def delete_tree(
    tree_id: str,
    repo: SQLAlchemyORMRepository = Depends(get_repo),
):
    """Delete a tree."""
    existing = repo.get_tree(tree_id)
    if existing is None:
        return _envelope_json_response(
            404,
            f"Tree '{tree_id}' not found",
            error=f"Tree '{tree_id}' not found",
            data=None,
        )

    repo.delete_tree(tree_id)
    repo.session.commit()
    return _envelope_json_response(
        202,
        f"Tree '{tree_id}' deleted",
        error=None,
        data={"tree_id": tree_id, "deleted": True},
    )


@router.get(
    "/{tree_id}/evidences",
    summary="List evidences for a tree",
    description="Get all evidence records associated with a tree.",
)
def list_tree_evidences(
    tree_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of evidences to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    repo: SQLAlchemyORMRepository = Depends(get_repo),
):
    """List evidences for a tree."""
    # 404 if tree not found
    existing = repo.get_tree(tree_id)
    if existing is None:
        return _envelope_json_response(
            404,
            f"Tree '{tree_id}' not found",
            error=f"Tree '{tree_id}' not found",
            data=None,
        )

    records = repo.get_evidences_by_tree(tree_id, limit=limit, offset=offset)
    total = repo.count_evidences_by_tree(tree_id)
    items = [TreeEvidenceResponse.from_record(r) for r in records]
    list_data = TreeEvidenceListData(items=items, total=total, tree_id=tree_id)
    return _build_envelope(list_data, message="Evidences retrieved", status_code=200)