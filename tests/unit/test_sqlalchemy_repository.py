#!/usr/bin/env python3
"""Unit tests for SQLAlchemyORMRepository tree/evidence metadata handling."""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.repository import spatialEntityModels  # noqa: F401
from src.repository.entityModels import Tree, TreeEvidence
from src.repository.sqlalchemyRepository import SQLAlchemyORMRepository


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_tree(**overrides) -> Tree:
    tree = Tree(
        id="tree-001",
        region_code="REG01",
        farm_id="farm-001",
        geohash7="w3gvb7h",
        row_idx=None,
        col_idx=None,
        tree_metadata={"source": "seed"},
    )
    tree.created_at = _now()
    tree.updated_at = _now()
    tree.captured_at = None
    for key, value in overrides.items():
        setattr(tree, key, value)
    return tree


def _make_evidence(**overrides) -> TreeEvidence:
    evidence = TreeEvidence(
        id=str(uuid.uuid4()),
        tree_id="tree-001",
        region_code="REG01",
        storage_cid="cid-001",
        evidence_hash="hash-001",
        raw_telemetry={"sensor": "imu"},
        tree_metadata={"image_id": "img-001"},
    )
    evidence.created_at = _now()
    evidence.updated_at = _now()
    evidence.captured_at = _now()
    for key, value in overrides.items():
        setattr(evidence, key, value)
    return evidence


class TestCreateTreeNullableGrid:
    def test_create_tree_accepts_null_grid_coordinates(self):
        session = MagicMock()
        session.get.return_value = None
        repo = SQLAlchemyORMRepository(session=session)

        ok = repo.create_tree(
            tree_id="tree-001",
            region_code="REG01",
            farm_id="farm-001",
            geohash_7="w3gvb7h",
            row_idx=None,
            col_idx=None,
            metadata=None,
        )

        assert ok is True
        session.get.assert_called_once_with(Tree, "tree-001")
        session.add.assert_called_once()

        tree = session.add.call_args.args[0]
        assert isinstance(tree, Tree)
        assert tree.row_idx is None
        assert tree.col_idx is None
        assert tree.tree_metadata == {}

    def test_create_tree_updates_existing_tree_with_null_grid_coordinates(self):
        session = MagicMock()
        existing = _make_tree(row_idx=4, col_idx=9)
        session.get.return_value = existing
        repo = SQLAlchemyORMRepository(session=session)

        ok = repo.create_tree(
            tree_id="tree-002",
            region_code="REG02",
            farm_id="farm-002",
            geohash_7="w3gvb7j",
            row_idx=None,
            col_idx=None,
            metadata=None,
        )

        assert ok is True
        assert existing.region_code == "REG02"
        assert existing.farm_id == "farm-002"
        assert existing.geohash7 == "w3gvb7j"
        assert existing.row_idx is None
        assert existing.col_idx is None
        assert existing.tree_metadata == {}
        session.add.assert_not_called()

    def test_create_tree_persists_tree_metadata(self):
        session = MagicMock()
        session.get.return_value = None
        repo = SQLAlchemyORMRepository(session=session)

        ok = repo.create_tree(
            tree_id="tree-003",
            region_code="REG03",
            farm_id="farm-003",
            geohash_7="w3gvb7k",
            row_idx=3,
            col_idx=7,
            metadata={"source": "ingest", "grid": True},
        )

        assert ok is True
        tree = session.add.call_args.args[0]
        assert tree.tree_metadata == {"source": "ingest", "grid": True}


class TestTreeMetadataReads:
    def test_get_tree_reads_tree_metadata(self):
        session = MagicMock()
        session.get.return_value = _make_tree(tree_metadata={"source": "ingest", "tree": "T-001"})
        repo = SQLAlchemyORMRepository(session=session)

        record = repo.get_tree("tree-001")

        assert record is not None
        assert record.metadata == {"source": "ingest", "tree": "T-001"}

    def test_list_trees_reads_tree_metadata(self):
        session = MagicMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [
            _make_tree(id="tree-010", tree_metadata={"batch": "A"}),
            _make_tree(id="tree-011", tree_metadata={"batch": "B"}),
        ]
        session.execute.return_value = result
        repo = SQLAlchemyORMRepository(session=session)

        records = repo.list_trees(farm_id="farm-001")

        assert [record.id for record in records] == ["tree-010", "tree-011"]
        assert [record.metadata for record in records] == [{"batch": "A"}, {"batch": "B"}]

    def test_update_tree_accepts_metadata_alias_and_sets_tree_metadata(self):
        session = MagicMock()
        existing = _make_tree(tree_metadata={"old": True})
        session.get.return_value = existing
        repo = SQLAlchemyORMRepository(session=session)

        ok = repo.update_tree("tree-001", metadata={"source": "updated"})

        assert ok is True
        assert existing.tree_metadata == {"source": "updated"}


class TestEvidenceMetadataMapping:
    def test_create_evidence_persists_tree_metadata(self):
        session = MagicMock()
        session.get.return_value = None
        repo = SQLAlchemyORMRepository(session=session)

        evidence_id = repo.create_evidence(
            tree_id="tree-001",
            region_code="REG01",
            global_vector=[0.1] * repo._get_vector_dimension(),
            storage_cid="cid-002",
            evidence_hash="hash-002",
            raw_telemetry={"sensor": "gps"},
            metadata={"image_id": "img-002", "minio_key": "features/img-002.npz.gz"},
        )

        assert evidence_id is not None
        session.add.assert_called_once()
        evidence = session.add.call_args.args[0]
        assert isinstance(evidence, TreeEvidence)
        assert evidence.tree_metadata == {
            "image_id": "img-002",
            "minio_key": "features/img-002.npz.gz",
        }

    def test_get_evidence_reads_tree_metadata(self):
        session = MagicMock()
        evidence = _make_evidence(tree_metadata={"image_id": "img-003", "camera": "front"})
        session.get.return_value = evidence
        repo = SQLAlchemyORMRepository(session=session)

        record = repo.get_evidence(uuid.UUID(str(evidence.id)))

        assert record is not None
        assert record.metadata == {"image_id": "img-003", "camera": "front"}

    def test_get_evidence_by_image_id_filters_on_tree_metadata(self):
        session = MagicMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = _make_evidence(
            tree_metadata={"image_id": "img-004", "camera": "rear"}
        )
        session.execute.return_value = result
        repo = SQLAlchemyORMRepository(session=session)

        record = repo.get_evidence_by_image_id("img-004")

        assert record is not None
        assert record.metadata == {"image_id": "img-004", "camera": "rear"}
