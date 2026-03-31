#!/usr/bin/env python3
"""
Unit tests for ORM schema alignment (entityModels + spatialEntityModels).

These tests verify the mapped columns/indexes match the current DB schema for:
- farm_zones
- trees
- tree_evidences

DB references:
- farm_zones:   PK = farm_id
- trees:        PK = id (single column, NOT composite)
- tree_evidences: PK = id, FK = tree_id -> trees.id
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy import inspect

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Ensure shared Base is initialized before importing spatial models.
from src.repository import entityModels  # noqa: F401


def _tables(mapper) -> dict:
    return {t.name: t for t in mapper.tables}


def _col_info(mapper, table_name: str, col_name: str):
    return _tables(mapper)[table_name].c[col_name]


def _idx_info(mapper, table_name: str, idx_name: str):
    table = _tables(mapper)[table_name]
    for idx in table.indexes:
        if idx.name == idx_name:
            return idx
    return None


# ---------------------------------------------------------------------------
# Import / export tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_base_is_shared(self):
        from src.repository.entityModels import Base as base_entity
        from src.repository.spatialEntityModels import Base as base_spatial

        assert base_entity is base_spatial

    def test_core_models_import(self):
        from src.repository.entityModels import Tree, TreeEvidence
        from src.repository.spatialEntityModels import FarmZone

        assert Tree.__name__ == "Tree"
        assert TreeEvidence.__name__ == "TreeEvidence"
        assert FarmZone.__name__ == "FarmZone"

    def test_spatial_exports(self):
        from src.repository import spatialEntityModels

        assert spatialEntityModels.__all__ == ["Base", "FarmZone"]


# ---------------------------------------------------------------------------
# FarmZone model tests
# ---------------------------------------------------------------------------


class TestFarmZoneModel:
    @pytest.fixture
    def mapper(self):
        from src.repository.spatialEntityModels import FarmZone

        return inspect(FarmZone)

    def test_pk_is_farm_id(self, mapper):
        table = _tables(mapper)["farm_zones"]
        assert [c.name for c in table.primary_key.columns] == ["farm_id"]

    def test_core_columns(self, mapper):
        assert _col_info(mapper, "farm_zones", "owner_did").nullable is False
        assert _col_info(mapper, "farm_zones", "region_code").nullable is False
        assert _col_info(mapper, "farm_zones", "farm_name").nullable is True
        assert _col_info(mapper, "farm_zones", "boundary").nullable is False

    def test_row_col_removed(self, mapper):
        """farm_zones must not expose tree-only grid coordinates."""
        table = _tables(mapper)["farm_zones"]
        assert "row_idx" not in table.c
        assert "col_idx" not in table.c

    def test_timestamps(self, mapper):
        created = _col_info(mapper, "farm_zones", "created_at")
        updated = _col_info(mapper, "farm_zones", "updated_at")
        assert created.server_default is not None
        assert updated.server_default is not None
        assert updated.onupdate is not None or updated.info.get("onupdate") is not None

    def test_indexes(self, mapper):
        idx_boundary = _idx_info(mapper, "farm_zones", "idx_farm_zones_boundary")
        assert idx_boundary is not None
        assert idx_boundary.dialect_options.get("postgresql", {}).get("using") == "gist"

        idx_region = _idx_info(mapper, "farm_zones", "idx_farm_zones_region_code")
        assert idx_region is not None
        assert list(idx_region.columns.keys()) == ["region_code"]

        idx_owner = _idx_info(mapper, "farm_zones", "idx_farm_zones_owner_did")
        assert idx_owner is not None
        assert list(idx_owner.columns.keys()) == ["owner_did"]


# ---------------------------------------------------------------------------
# Tree model tests — mirrors actual DB: PK = id (single varchar), NOT composite
# ---------------------------------------------------------------------------


class TestTreeModel:
    @pytest.fixture
    def mapper(self):
        from src.repository.entityModels import Tree

        return inspect(Tree)

    def test_pk_is_id(self, mapper):
        """DB PK is 'id' (single varchar), NOT composite (region_code, tree_id)."""
        table = _tables(mapper)["trees"]
        assert [c.name for c in table.primary_key.columns] == ["id"]

    def test_core_columns(self, mapper):
        assert _col_info(mapper, "trees", "farm_id").nullable is False
        assert _col_info(mapper, "trees", "geohash_7").nullable is False
        # location is nullable in the real DB
        assert _col_info(mapper, "trees", "location").nullable is True
        assert _col_info(mapper, "trees", "row_idx").nullable is True
        assert _col_info(mapper, "trees", "col_idx").nullable is True
        assert _col_info(mapper, "trees", "row_idx").server_default is None
        assert _col_info(mapper, "trees", "col_idx").server_default is None
        # binary_code / pq_code do NOT exist in the real DB
        assert "binary_code" not in _tables(mapper)["trees"].c
        assert "pq_code" not in _tables(mapper)["trees"].c
        # codebook_id IS nullable in the real DB
        assert _col_info(mapper, "trees", "codebook_id").nullable is True

    def test_geohash_len(self, mapper):
        col = _col_info(mapper, "trees", "geohash_7")
        assert getattr(col.type, "length", None) == 7

    def test_id_alias_exists(self):
        """Tree.id is the primary key column (mirrors DB 'id' column)."""
        from src.repository.entityModels import Tree

        assert hasattr(Tree, "id")
        # __mapper__.primary_key is a tuple of Column objects in SQLAlchemy 2.x
        pk_col_names = [c.name for c in Tree.__mapper__.primary_key]
        assert pk_col_names == ["id"]

    def test_tree_farm_fk_restrict(self, mapper):
        col = _col_info(mapper, "trees", "farm_id")
        fk = next(iter(col.foreign_keys))
        assert fk.target_fullname == "farm_zones.farm_id"
        assert fk.ondelete == "RESTRICT"

    def test_tree_indexes(self, mapper):
        idx_grid = _idx_info(mapper, "trees", "idx_trees_grid")
        assert idx_grid is not None
        assert list(idx_grid.columns.keys()) == ["farm_id", "row_idx", "col_idx"]

        idx_loc = _idx_info(mapper, "trees", "idx_trees_location")
        assert idx_loc is not None
        assert list(idx_loc.columns.keys()) == ["location"]
        assert idx_loc.dialect_options.get("postgresql", {}).get("using") == "gist"

        idx_geo = _idx_info(mapper, "trees", "idx_trees_geohash")
        assert idx_geo is not None
        assert list(idx_geo.columns.keys()) == ["geohash_7"]

        idx_geo_farm = _idx_info(mapper, "trees", "idx_trees_geohash_fm")
        assert idx_geo_farm is not None
        assert list(idx_geo_farm.columns.keys()) == ["farm_id", "geohash_7"]

        idx_updated = _idx_info(mapper, "trees", "idx_trees_updated_at")
        assert idx_updated is not None


# ---------------------------------------------------------------------------
# Tree <-> FarmZone relationship tests
# ---------------------------------------------------------------------------


class TestTreeFarmZoneRelationship:
    def test_relationships_exist(self):
        from src.repository.entityModels import Tree
        from src.repository.spatialEntityModels import FarmZone

        assert hasattr(Tree, "farm_zone")
        assert hasattr(FarmZone, "trees")

    def test_back_populates(self):
        from src.repository.entityModels import Tree
        from src.repository.spatialEntityModels import FarmZone

        farm_mapper = inspect(FarmZone)
        tree_mapper = inspect(Tree)

        farm_rel = next(r for r in farm_mapper.relationships if r.key == "trees")
        tree_rel = next(r for r in tree_mapper.relationships if r.key == "farm_zone")

        assert farm_rel.back_populates == "farm_zone"
        assert tree_rel.back_populates == "trees"


# ---------------------------------------------------------------------------
# TreeEvidence model tests — mirrors actual DB schema
# ---------------------------------------------------------------------------


class TestTreeEvidenceModel:
    @pytest.fixture
    def mapper(self):
        from src.repository.entityModels import TreeEvidence

        return inspect(TreeEvidence)

    def test_pk_is_id_string(self, mapper):
        """DB PK column name is 'id' (varchar 50, exposed as evidence_id via synonym)."""
        table = _tables(mapper)["tree_evidences"]
        # The actual PK column name in DB/ORM is "id"
        pk_col_names = [c.name for c in table.primary_key.columns]
        assert pk_col_names == ["id"], (
            f"Expected PK column name 'id', got {pk_col_names}. "
            "Note: evidence_id is a synonym for the 'id' column."
        )

        # The PK column "id" is String(50)
        pk_col = table.c["id"]
        assert pk_col.type.python_type is str

    def test_core_columns(self, mapper):
        assert _col_info(mapper, "tree_evidences", "tree_id").nullable is False
        assert _col_info(mapper, "tree_evidences", "region_code").nullable is False
        assert _col_info(mapper, "tree_evidences", "camera_heading").nullable is True
        assert _col_info(mapper, "tree_evidences", "camera_pitch").nullable is True
        assert _col_info(mapper, "tree_evidences", "storage_cid").nullable is False
        assert _col_info(mapper, "tree_evidences", "evidence_hash").nullable is False
        assert _col_info(mapper, "tree_evidences", "raw_telemetry").nullable is False

    def test_global_vector_column_exists(self, mapper):
        """global_vector (halfvec) must exist and be nullable."""
        table = _tables(mapper)["tree_evidences"]
        assert "global_vector" in table.c
        assert table.c["global_vector"].nullable is True

    def test_extra_columns_exist(self, mapper):
        """camera_roll, metadata, location, created_at, updated_at must exist."""
        table = _tables(mapper)["tree_evidences"]
        assert "camera_roll" in table.c
        assert "metadata" in table.c
        assert "location" in table.c
        assert "created_at" in table.c
        assert "updated_at" in table.c

    def test_defaults(self, mapper):
        captured = _col_info(mapper, "tree_evidences", "captured_at")
        verified = _col_info(mapper, "tree_evidences", "is_c2pa_verified")
        assert captured.server_default is not None
        assert verified.server_default is not None

    def test_indexes(self, mapper):
        idx_tree = _idx_info(mapper, "tree_evidences", "idx_evidences_tree_id")
        assert idx_tree is not None
        assert list(idx_tree.columns.keys()) == ["tree_id"]

        idx_region = _idx_info(mapper, "tree_evidences", "idx_evidences_region")
        assert idx_region is not None
        assert list(idx_region.columns.keys()) == ["region_code"]

        idx_captured = _idx_info(mapper, "tree_evidences", "idx_evidences_captured")
        assert idx_captured is not None
        assert list(idx_captured.columns.keys()) == ["captured_at"]

        idx_telemetry = _idx_info(mapper, "tree_evidences", "idx_evidences_telemetry")
        assert idx_telemetry is not None
        assert list(idx_telemetry.columns.keys()) == ["raw_telemetry"]
        assert idx_telemetry.dialect_options.get("postgresql", {}).get("using") == "gin"

        idx_camera = _idx_info(mapper, "tree_evidences", "idx_evidences_camera")
        assert idx_camera is not None
        assert list(idx_camera.columns.keys()) == ["tree_id", "camera_heading", "camera_pitch"]

        idx_location = _idx_info(mapper, "tree_evidences", "idx_evidences_location")
        assert idx_location is not None
        assert list(idx_location.columns.keys()) == ["location"]
        assert idx_location.dialect_options.get("postgresql", {}).get("using") == "gist"

    def test_simple_fk_to_trees_id(self, mapper):
        """FK is simple: tree_evidences.tree_id -> trees.id (NOT composite)."""
        col = _col_info(mapper, "tree_evidences", "tree_id")
        fk = next(iter(col.foreign_keys))
        assert fk.target_fullname == "trees.id"
        assert fk.ondelete == "CASCADE"


# ---------------------------------------------------------------------------
# Metadata registry tests
# ---------------------------------------------------------------------------


class TestMetadataRegistry:
    def test_registry_has_expected_tables(self):
        from src.repository.entityModels import Base

        table_names = {t.name for t in Base.metadata.tables.values()}
        assert {"farm_zones", "trees", "tree_evidences"}.issubset(table_names)
