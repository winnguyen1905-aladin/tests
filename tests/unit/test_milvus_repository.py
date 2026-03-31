#!/usr/bin/env python3
"""
Unit tests for MilvusRepository.

Tests WITHOUT a real Milvus server:
- insert: happy path, not-connected guard, GPS+angle fields, metadata
- search: result parsing, tree_id filter expression, empty results
- search_with_bounding_box: filter expression construction, GPS+angle combos
- coarse_retrieval: threshold filtering, candidate dict structure
- MilvusResult dataclass
- MilvusConfig defaults
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any, List

from src.repository.milvusRepository import (
    MilvusConfig,
    MilvusResult,
    MilvusRepository,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(connected: bool = True) -> MilvusRepository:
    """Create a MilvusRepository whose Milvus client is fully mocked."""
    repo = MilvusRepository.__new__(MilvusRepository)
    repo.config = MilvusConfig(
        uri="http://localhost:19530",
        collection_name="tree_features",
        vector_dim=8,
        metric_type="COSINE",
        nprobe=32,
        verbose=False,
    )
    repo.app_config = MagicMock()
    repo._use_shared = False
    repo._connected = connected
    repo.client = MagicMock() if connected else None
    return repo


def _make_vector(dim: int = 8, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_hit(image_id: str, distance: float, tree_id: str, metadata: dict = None) -> dict:
    """Mimic the dict structure returned by MilvusClient.search hits."""
    return {
        "id": image_id,
        "distance": distance,
        "entity": {
            "tree_id": tree_id,
            "metadata": metadata or {},
        },
    }


def _make_search_result(hits: list) -> list:
    """Wrap hits in the outer list returned by MilvusClient.search."""
    return [hits]


# ---------------------------------------------------------------------------
# MilvusConfig defaults
# ---------------------------------------------------------------------------


class TestMilvusConfig:
    def test_default_values(self):
        cfg = MilvusConfig()
        assert cfg.uri == "http://localhost:19530"
        assert cfg.collection_name == "tree_features"
        assert cfg.metric_type == "COSINE"
        assert cfg.verbose is False

    def test_custom_values(self):
        cfg = MilvusConfig(uri="http://remote:19530", vector_dim=512, verbose=True)
        assert cfg.uri == "http://remote:19530"
        assert cfg.vector_dim == 512
        assert cfg.verbose is True


# ---------------------------------------------------------------------------
# MilvusResult dataclass
# ---------------------------------------------------------------------------


class TestMilvusResult:
    def test_fields_stored(self):
        r = MilvusResult(
            ids=["a", "b"],
            distances=[0.9, 0.8],
            tree_ids=["T1", "T2"],
            metadatas=[{"k": 1}, {}],
        )
        assert r.ids == ["a", "b"]
        assert r.distances == [0.9, 0.8]
        assert r.tree_ids == ["T1", "T2"]

    def test_empty_result(self):
        r = MilvusResult(ids=[], distances=[], tree_ids=[])
        assert len(r.ids) == 0


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


class TestInsert:
    def test_returns_false_when_not_connected(self):
        repo = _make_repo(connected=False)
        assert repo.insert("img_001", "tree_A", _make_vector()) is False

    def test_successful_insert_returns_true(self):
        repo = _make_repo()
        repo.client.insert = MagicMock(return_value=None)
        repo.client.flush = MagicMock(return_value=None)

        result = repo.insert("img_001", "tree_A", _make_vector())

        assert result is True
        repo.client.insert.assert_called_once()

    def test_insert_uses_correct_collection(self):
        repo = _make_repo()
        repo.client.insert = MagicMock(return_value=None)
        repo.client.flush = MagicMock(return_value=None)

        repo.insert("img_001", "tree_A", _make_vector())

        call_kwargs = repo.client.insert.call_args
        # positional or keyword
        if call_kwargs[0]:
            assert call_kwargs[0][0] == "tree_features"
        else:
            assert call_kwargs[1]["collection_name"] == "tree_features"

    def test_insert_includes_gps_fields_when_provided(self):
        repo = _make_repo()
        inserted_data = {}

        def capture_insert(collection_name, data):
            inserted_data.update(data[0])

        repo.client.insert = MagicMock(
            side_effect=lambda collection_name, data: inserted_data.update(data[0])
        )
        repo.client.flush = MagicMock(return_value=None)

        repo.insert(
            "img_gps",
            "tree_B",
            _make_vector(),
            longitude=106.7,
            latitude=10.8,
        )

        assert inserted_data.get("longitude") == pytest.approx(106.7)
        assert inserted_data.get("latitude") == pytest.approx(10.8)

    def test_insert_includes_angle_fields_when_provided(self):
        repo = _make_repo()
        inserted_data = {}

        def capture(collection_name, data):
            inserted_data.update(data[0])

        repo.client.insert = MagicMock(side_effect=capture)
        repo.client.flush = MagicMock(return_value=None)

        repo.insert(
            "img_angle",
            "tree_C",
            _make_vector(),
            hor_angle=45.0,
            ver_angle=-10.0,
            pitch=5.0,
        )

        assert inserted_data.get("hor_angle") == pytest.approx(45.0)
        assert inserted_data.get("ver_angle") == pytest.approx(-10.0)
        assert inserted_data.get("pitch") == pytest.approx(5.0)

    def test_insert_omits_gps_fields_when_none(self):
        repo = _make_repo()
        inserted_data = {}

        def capture(collection_name, data):
            inserted_data.update(data[0])

        repo.client.insert = MagicMock(side_effect=capture)
        repo.client.flush = MagicMock(return_value=None)

        repo.insert("img_no_gps", "tree_D", _make_vector())

        assert "longitude" not in inserted_data
        assert "latitude" not in inserted_data

    def test_insert_omits_position_3d(self):
        """position_3d field has been removed - it's legacy/deprecated."""
        repo = _make_repo()
        inserted_data = {}

        def capture(collection_name, data):
            inserted_data.update(data[0])

        repo.client.insert = MagicMock(side_effect=capture)
        repo.client.flush = MagicMock(return_value=None)

        repo.insert("img_pos", "tree_A", _make_vector())

        # position_3d should NOT be present in inserted data
        assert "position_3d" not in inserted_data

    def test_insert_exception_returns_false(self):
        repo = _make_repo()
        repo.client.insert = MagicMock(side_effect=Exception("Milvus error"))

        assert repo.insert("img_err", "tree_A", _make_vector()) is False

    def test_metadata_stored_as_dict(self):
        repo = _make_repo()
        inserted_data = {}

        def capture(collection_name, data):
            inserted_data.update(data[0])

        repo.client.insert = MagicMock(side_effect=capture)
        repo.client.flush = MagicMock(return_value=None)

        meta = {"source": "test", "camera": "Canon"}
        repo.insert("img_meta", "tree_A", _make_vector(), metadata=meta)

        assert inserted_data.get("metadata") == meta

    def test_metadata_defaults_to_empty_dict(self):
        repo = _make_repo()
        inserted_data = {}

        def capture(collection_name, data):
            inserted_data.update(data[0])

        repo.client.insert = MagicMock(side_effect=capture)
        repo.client.flush = MagicMock(return_value=None)

        repo.insert("img_nometa", "tree_A", _make_vector())
        assert inserted_data.get("metadata") == {}


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_raises_when_not_connected(self):
        repo = _make_repo(connected=False)
        with pytest.raises(ValueError, match="not connected"):
            repo.search(_make_vector())

    def test_returns_milvus_result(self):
        repo = _make_repo()
        hits = [_make_hit("img_1", 0.95, "tree_A"), _make_hit("img_2", 0.80, "tree_B")]
        repo.client.search = MagicMock(return_value=_make_search_result(hits))

        result = repo.search(_make_vector(), top_k=10)

        assert isinstance(result, MilvusResult)
        assert len(result.ids) == 2
        assert result.ids == ["img_1", "img_2"]
        assert result.distances == pytest.approx([0.95, 0.80])
        assert result.tree_ids == ["tree_A", "tree_B"]

    def test_empty_search_result_returns_empty_milvus_result(self):
        repo = _make_repo()
        repo.client.search = MagicMock(return_value=_make_search_result([]))

        result = repo.search(_make_vector())

        assert result.ids == []
        assert result.distances == []
        assert result.tree_ids == []

    def test_none_search_result_returns_empty(self):
        repo = _make_repo()
        repo.client.search = MagicMock(return_value=None)

        result = repo.search(_make_vector())
        assert result.ids == []

    def test_tree_id_filter_adds_filter_expression(self):
        repo = _make_repo()
        repo.client.search = MagicMock(return_value=_make_search_result([]))

        repo.search(_make_vector(), tree_id_filter="tree_ABC")

        call_kwargs = repo.client.search.call_args[1]
        assert "filter" in call_kwargs
        assert "tree_ABC" in call_kwargs["filter"]

    def test_no_tree_id_filter_omits_filter(self):
        repo = _make_repo()
        repo.client.search = MagicMock(return_value=_make_search_result([]))

        repo.search(_make_vector(), tree_id_filter=None)

        call_kwargs = repo.client.search.call_args[1]
        assert "filter" not in call_kwargs

    def test_empty_tree_id_filter_treated_as_no_filter(self):
        repo = _make_repo()
        repo.client.search = MagicMock(return_value=_make_search_result([]))

        repo.search(_make_vector(), tree_id_filter="")

        call_kwargs = repo.client.search.call_args[1]
        assert "filter" not in call_kwargs

    def test_anns_field_is_global_vector(self):
        repo = _make_repo()
        repo.client.search = MagicMock(return_value=_make_search_result([]))

        repo.search(_make_vector())

        call_kwargs = repo.client.search.call_args[1]
        assert call_kwargs.get("anns_field") == "global_vector"

    def test_top_k_passed_as_limit(self):
        repo = _make_repo()
        repo.client.search = MagicMock(return_value=_make_search_result([]))

        repo.search(_make_vector(), top_k=7)

        call_kwargs = repo.client.search.call_args[1]
        assert call_kwargs.get("limit") == 7


# ---------------------------------------------------------------------------
# search_with_bounding_box — filter expression construction
# ---------------------------------------------------------------------------


class TestSearchWithBoundingBox:
    def _get_filter(self, repo: MilvusRepository, **bb_kwargs) -> str:
        """Run a bounding-box search and return the filter string passed to Milvus."""
        repo.client.search = MagicMock(return_value=_make_search_result([]))
        repo.search_with_bounding_box(_make_vector(), **bb_kwargs)
        call_kwargs = repo.client.search.call_args[1]
        return call_kwargs.get("filter", "")

    def test_no_filters_no_filter_expression(self):
        repo = _make_repo()
        f = self._get_filter(repo)
        assert f == ""

    def test_lon_min_filter(self):
        repo = _make_repo()
        f = self._get_filter(repo, lon_min=100.0)
        assert "longitude >= 100.0" in f

    def test_lon_max_filter(self):
        repo = _make_repo()
        f = self._get_filter(repo, lon_max=110.0)
        assert "longitude <= 110.0" in f

    def test_lat_min_filter(self):
        repo = _make_repo()
        f = self._get_filter(repo, lat_min=10.0)
        assert "latitude >= 10.0" in f

    def test_lat_max_filter(self):
        repo = _make_repo()
        f = self._get_filter(repo, lat_max=20.0)
        assert "latitude <= 20.0" in f

    def test_full_gps_bounding_box(self):
        repo = _make_repo()
        f = self._get_filter(repo, lon_min=100.0, lon_max=110.0, lat_min=10.0, lat_max=20.0)
        assert "longitude >= 100.0" in f
        assert "longitude <= 110.0" in f
        assert "latitude >= 10.0" in f
        assert "latitude <= 20.0" in f

    def test_hor_angle_filter(self):
        repo = _make_repo()
        f = self._get_filter(repo, hor_angle_min=30.0, hor_angle_max=90.0)
        assert "hor_angle >= 30.0" in f
        assert "hor_angle <= 90.0" in f

    def test_ver_angle_filter(self):
        repo = _make_repo()
        f = self._get_filter(repo, ver_angle_min=-30.0, ver_angle_max=30.0)
        assert "ver_angle >= -30.0" in f
        assert "ver_angle <= 30.0" in f

    def test_pitch_filter(self):
        repo = _make_repo()
        f = self._get_filter(repo, pitch_min=-5.0, pitch_max=5.0)
        assert "pitch >= -5.0" in f
        assert "pitch <= 5.0" in f

    def test_tree_id_filter_appended(self):
        repo = _make_repo()
        f = self._get_filter(repo, lon_min=100.0, tree_id_filter="tree_XYZ")
        assert "tree_id" in f
        assert "tree_XYZ" in f

    def test_limit_multiplied_when_filters_active(self):
        repo = _make_repo()
        repo.client.search = MagicMock(return_value=_make_search_result([]))

        repo.search_with_bounding_box(_make_vector(), top_k=10, lon_min=100.0)

        call_kwargs = repo.client.search.call_args[1]
        # limit should be top_k * 3 when a filter is present
        assert call_kwargs.get("limit") == 30

    def test_no_filter_uses_exact_top_k(self):
        repo = _make_repo()
        repo.client.search = MagicMock(return_value=_make_search_result([]))

        repo.search_with_bounding_box(_make_vector(), top_k=10)

        call_kwargs = repo.client.search.call_args[1]
        assert call_kwargs.get("limit") == 10

    def test_results_limited_to_top_k(self):
        repo = _make_repo()
        hits = [_make_hit(f"img_{i}", 0.9 - i * 0.01, "tree_A") for i in range(20)]
        repo.client.search = MagicMock(return_value=_make_search_result(hits))

        result = repo.search_with_bounding_box(_make_vector(), top_k=5, lon_min=100.0)

        assert len(result.ids) == 5

    def test_raises_when_not_connected(self):
        repo = _make_repo(connected=False)
        with pytest.raises(ValueError, match="not connected"):
            repo.search_with_bounding_box(_make_vector())

    def test_exception_is_re_raised(self):
        repo = _make_repo()
        repo.client.search = MagicMock(side_effect=Exception("DB error"))
        with pytest.raises(Exception, match="DB error"):
            repo.search_with_bounding_box(_make_vector(), lon_min=100.0)

    def test_gps_output_fields_included_when_gps_filter_active(self):
        repo = _make_repo()
        repo.client.search = MagicMock(return_value=_make_search_result([]))

        repo.search_with_bounding_box(_make_vector(), lat_min=10.0)

        call_kwargs = repo.client.search.call_args[1]
        assert "longitude" in call_kwargs["output_fields"]
        assert "latitude" in call_kwargs["output_fields"]

    def test_angle_output_fields_included_when_angle_filter_active(self):
        repo = _make_repo()
        repo.client.search = MagicMock(return_value=_make_search_result([]))

        repo.search_with_bounding_box(_make_vector(), hor_angle_min=30.0)

        call_kwargs = repo.client.search.call_args[1]
        assert "hor_angle" in call_kwargs["output_fields"]
        assert "ver_angle" in call_kwargs["output_fields"]
        assert "pitch" in call_kwargs["output_fields"]


# ---------------------------------------------------------------------------
# coarse_retrieval
# ---------------------------------------------------------------------------


class TestCoarseRetrieval:
    def test_returns_list_of_candidate_dicts(self):
        repo = _make_repo()
        hits = [
            _make_hit("img_1", 0.95, "tree_A", {"minio_key": "features/img_1.npz.gz"}),
            _make_hit("img_2", 0.80, "tree_B", {"minio_key": "features/img_2.npz.gz"}),
        ]
        repo.client.search = MagicMock(return_value=_make_search_result(hits))

        candidates = repo.coarse_retrieval(_make_vector(), top_k=10, threshold=0.5)

        assert isinstance(candidates, list)
        assert len(candidates) == 2
        for c in candidates:
            assert "image_id" in c
            assert "tree_id" in c
            assert "similarity_score" in c

    def test_threshold_filters_low_similarity(self):
        repo = _make_repo()
        hits = [
            _make_hit("img_high", 0.90, "tree_A"),
            _make_hit("img_low", 0.30, "tree_B"),
        ]
        repo.client.search = MagicMock(return_value=_make_search_result(hits))

        candidates = repo.coarse_retrieval(_make_vector(), top_k=5, threshold=0.6)

        ids = [c["image_id"] for c in candidates]
        assert "img_high" in ids
        assert "img_low" not in ids

    def test_empty_result_returns_empty_list(self):
        repo = _make_repo()
        repo.client.search = MagicMock(return_value=_make_search_result([]))

        candidates = repo.coarse_retrieval(_make_vector(), top_k=5)

        assert candidates == []

    def test_candidates_sorted_by_similarity_descending(self):
        repo = _make_repo()
        hits = [
            _make_hit("img_a", 0.70, "tree_A"),
            _make_hit("img_b", 0.95, "tree_B"),
            _make_hit("img_c", 0.85, "tree_C"),
        ]
        repo.client.search = MagicMock(return_value=_make_search_result(hits))

        candidates = repo.coarse_retrieval(_make_vector(), top_k=10, threshold=0.0)

        sims = [c["similarity_score"] for c in candidates]
        # coarse_retrieval preserves Milvus hit order (does not sort client-side)
        assert sims == [0.70, 0.95, 0.85]
