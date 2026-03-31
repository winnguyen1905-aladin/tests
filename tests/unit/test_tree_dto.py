#!/usr/bin/env python3
"""Unit tests for tree DTO validation and exports."""

import struct
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from geoalchemy2.elements import WKBElement
from pydantic import ValidationError

from src.dto import (
    TreeCreateRequest,
    TreeListData,
    TreeListQuery,
    TreePatchRequest,
    TreeUpdateRequest,
    TreeResponse,
)
from src.dto.tree import _extract_lat_lon


def _tree_create_payload(**overrides):
    payload = {
        "id": "tree-001",
        "region_code": "REG01",
        "farm_id": "farm-001",
        "geohash_7": "w3gvb7h",
    }
    payload.update(overrides)
    return payload


class TestTreeCreateRequest:
    def test_requires_tree_id(self):
        payload = _tree_create_payload()
        payload.pop("id")

        with pytest.raises(ValidationError, match="id"):
            TreeCreateRequest.model_validate(payload)

    def test_validates_fixed_length_geohash(self):
        with pytest.raises(ValidationError, match="at least 7 characters"):
            TreeCreateRequest.model_validate(_tree_create_payload(geohash_7="short"))

        with pytest.raises(ValidationError, match="at most 7 characters"):
            TreeCreateRequest.model_validate(_tree_create_payload(geohash_7="toolong1"))

    def test_requires_coordinate_pair(self):
        with pytest.raises(
            ValidationError, match="latitude and longitude must be provided together"
        ):
            TreeCreateRequest.model_validate(_tree_create_payload(latitude=10.0))

        model = TreeCreateRequest.model_validate(
            _tree_create_payload(latitude=10.805, longitude=106.705)
        )
        assert model.latitude == pytest.approx(10.805)
        assert model.longitude == pytest.approx(106.705)

    def test_parses_datetime_and_keeps_default_codebook(self):
        model = TreeCreateRequest.model_validate(
            _tree_create_payload(captured_at="2026-03-29T12:34:56Z")
        )

        assert isinstance(model.captured_at, datetime)
        assert model.codebook_id == "codebook_v1"


class TestTreePatchRequest:
    def test_rejects_empty_payload(self):
        with pytest.raises(ValidationError, match="at least one field must be provided"):
            TreePatchRequest.model_validate({})

    def test_requires_coordinate_pair_when_patching_location(self):
        with pytest.raises(
            ValidationError, match="latitude and longitude must be provided together"
        ):
            TreePatchRequest.model_validate({"latitude": 10.0})

        model = TreePatchRequest.model_validate({"latitude": 10.805, "longitude": 106.705})
        assert model.latitude == pytest.approx(10.805)
        assert model.longitude == pytest.approx(106.705)


class TestTreeUpdateRequest:
    def test_requires_core_fields(self):
        with pytest.raises(ValidationError, match="region_code"):
            TreeUpdateRequest.model_validate({"farm_id": "farm-001", "geohash_7": "w3gvb7h"})

    def test_requires_coordinate_pair(self):
        with pytest.raises(
            ValidationError, match="latitude and longitude must be provided together"
        ):
            TreeUpdateRequest.model_validate(
                {
                    "region_code": "REG01",
                    "farm_id": "farm-001",
                    "geohash_7": "w3gvb7h",
                    "longitude": 106.705,
                }
            )


class TestTreeResponse:
    def test_serializes_datetimes_as_iso_strings(self):
        timestamp = datetime(2026, 3, 29, 12, 34, 56, tzinfo=timezone.utc)

        model = TreeResponse(
            id="tree-001",
            region_code="REG01",
            farm_id="farm-001",
            geohash_7="w3gvb7h",
            latitude=10.805,
            longitude=106.705,
            row_idx=3,
            col_idx=7,
            codebook_id="codebook_v1",
            metadata={"source": "unit"},
            captured_at=timestamp,
            created_at=timestamp,
            updated_at=timestamp,
        )

        dumped = model.model_dump(mode="json")

        assert dumped["captured_at"] == "2026-03-29T12:34:56Z"
        assert dumped["created_at"] == "2026-03-29T12:34:56Z"
        assert dumped["updated_at"] == "2026-03-29T12:34:56Z"

    def test_from_record_extracts_lat_lon_from_point_wkt(self):
        timestamp = datetime(2026, 3, 29, 12, 34, 56, tzinfo=timezone.utc)
        record = SimpleNamespace(
            id="tree-001",
            region_code="REG01",
            farm_id="farm-001",
            geohash7="w3gvb7h",
            location="POINT(106.705 10.805)",
            row_idx=3,
            col_idx=7,
            codebook_id="codebook_v1",
            metadata={"source": "unit"},
            captured_at=timestamp,
            created_at=timestamp,
            updated_at=timestamp,
        )

        model = TreeResponse.from_record(record)

        assert model.latitude == pytest.approx(10.805)
        assert model.longitude == pytest.approx(106.705)

    def test_from_record_extracts_lat_lon_from_wkb(self):
        timestamp = datetime(2026, 3, 29, 12, 34, 56, tzinfo=timezone.utc)
        raw = struct.pack("<bIdd", 1, 1, 106.705, 10.805)
        location = WKBElement(raw, srid=4326, extended=False)
        record = SimpleNamespace(
            id="tree-001",
            region_code="REG01",
            farm_id="farm-001",
            geohash7="w3gvb7h",
            location=location,
            row_idx=3,
            col_idx=7,
            codebook_id="codebook_v1",
            metadata={},
            captured_at=timestamp,
            created_at=timestamp,
            updated_at=timestamp,
        )

        model = TreeResponse.from_record(record)

        assert model.latitude == pytest.approx(10.805)
        assert model.longitude == pytest.approx(106.705)


class TestTreeListQuery:
    def test_accepts_grid_filters(self):
        model = TreeListQuery.model_validate({"row_idx": 3, "col_idx": 7, "limit": 25, "offset": 5})

        assert model.row_idx == 3
        assert model.col_idx == 7
        assert model.limit == 25
        assert model.offset == 5


class TestLocationExtraction:
    def test_extract_lat_lon_from_ewkb_hex_string(self):
        raw = struct.pack("<bIIdd", 1, 0x20000001, 4326, 106.705, 10.805)
        latitude, longitude = _extract_lat_lon(raw.hex())

        assert latitude == pytest.approx(10.805)
        assert longitude == pytest.approx(106.705)


class TestTreeDtoExports:
    def test_tree_dtos_are_exported(self):
        assert TreeCreateRequest.__name__ == "TreeCreateRequest"
        assert TreeUpdateRequest.__name__ == "TreeUpdateRequest"
        assert TreePatchRequest.__name__ == "TreePatchRequest"
        assert TreeResponse.__name__ == "TreeResponse"
        assert TreeListQuery.__name__ == "TreeListQuery"
        assert TreeListData.__name__ == "TreeListData"
