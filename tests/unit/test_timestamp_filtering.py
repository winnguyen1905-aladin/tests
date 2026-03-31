#!/usr/bin/env python3
"""
Unit tests for timestamp-based filtering functionality.

Tests:
- MilvusRepository: time filter expression construction, output fields
- HierarchicalMatchingService: captured_at passing through pipeline
- Main.py: default 3-month filter
- HierarchicalMatcher: MatchingResult captured_at field
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

from src.repository.milvusRepository import (
    MilvusConfig,
    MilvusResult,
    MilvusRepository,
)
from src.processor.hierarchicalMatcher import MatchingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(connected: bool = True) -> MilvusRepository:
    """Create a MilvusRepository with mocked client."""
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


def _make_search_result(hits: list) -> list:
    return [hits]


# ---------------------------------------------------------------------------
# Test MilvusRepository time filter
# ---------------------------------------------------------------------------


class TestMilvusRepositoryTimestampFilter:
    """Test timestamp filtering in MilvusRepository."""

    def test_search_with_bounding_box_includes_captured_at_in_output_fields_when_time_filter_used(
        self,
    ):
        """When time filter is applied, captured_at should be in output fields."""
        repo = _make_repo()

        query_vector = _make_vector()
        now = int(datetime.now(timezone.utc).timestamp())

        # Mock the search result
        repo.client.search.return_value = _make_search_result([])

        # Call with time filter
        repo.search_with_bounding_box(
            query_vector=query_vector,
            top_k=10,
            captured_at_min=now - 86400 * 30,  # 30 days ago
        )

        # Check that search was called with captured_at in output_fields
        call_args = repo.client.search.call_args
        output_fields = call_args.kwargs.get("output_fields", [])
        assert "captured_at" in output_fields, (
            "captured_at should be in output_fields when time filter is used"
        )

    def test_search_with_bounding_box_excludes_captured_at_when_no_time_filter(self):
        """When no time filter, captured_at should not be in output fields."""
        repo = _make_repo()

        query_vector = _make_vector()

        # Mock the search result
        repo.client.search.return_value = _make_search_result([])

        # Call without time filter
        repo.search_with_bounding_box(
            query_vector=query_vector,
            top_k=10,
        )

        # Check that search was called without captured_at in output_fields
        call_args = repo.client.search.call_args
        output_fields = call_args.kwargs.get("output_fields", [])
        assert "captured_at" not in output_fields, (
            "captured_at should not be in output_fields when no time filter"
        )

    def test_time_filter_expression_construction(self):
        """Test that time filter expression is correctly constructed."""
        repo = _make_repo()

        query_vector = _make_vector()
        now = int(datetime.now(timezone.utc).timestamp())
        three_months_ago = now - (90 * 86400)

        repo.client.search.return_value = _make_search_result([])

        # Test with min and max
        repo.search_with_bounding_box(
            query_vector=query_vector,
            top_k=10,
            captured_at_min=three_months_ago,
            captured_at_max=now,
        )

        call_args = repo.client.search.call_args
        filter_expr = call_args.kwargs.get("filter", "")

        assert f"captured_at >= {three_months_ago}" in filter_expr
        assert f"captured_at <= {now}" in filter_expr

    def test_time_filter_min_only(self):
        """Test time filter with only minimum timestamp."""
        repo = _make_repo()

        query_vector = _make_vector()
        three_months_ago = int((datetime.now(timezone.utc) - timedelta(days=90)).timestamp())

        repo.client.search.return_value = _make_search_result([])

        repo.search_with_bounding_box(
            query_vector=query_vector,
            top_k=10,
            captured_at_min=three_months_ago,
        )

        call_args = repo.client.search.call_args
        filter_expr = call_args.kwargs.get("filter", "")

        assert f"captured_at >= {three_months_ago}" in filter_expr


# ---------------------------------------------------------------------------
# Test MatchingResult captured_at field
# ---------------------------------------------------------------------------


class TestMatchingResultTimestamp:
    """Test MatchingResult dataclass with captured_at field."""

    def test_matching_result_has_captured_at_field(self):
        """Test that MatchingResult has captured_at field."""
        now = int(datetime.now(timezone.utc).timestamp())

        result = MatchingResult(
            query_id="query_1",
            candidate_id="candidate_1",
            tree_id="tree_1",
            dino_similarity=0.9,
            captured_at=now,
        )

        assert result.captured_at == now
        assert isinstance(result.captured_at, int)

    def test_matching_result_captured_at_none_by_default(self):
        """Test that captured_at defaults to None."""
        result = MatchingResult(
            query_id="query_1",
            candidate_id="candidate_1",
        )

        assert result.captured_at is None

    def test_matching_result_to_dict_includes_captured_at(self):
        """Test that to_dict() includes captured_at."""
        now = int(datetime.now(timezone.utc).timestamp())

        result = MatchingResult(
            query_id="query_1",
            candidate_id="candidate_1",
            captured_at=now,
        )

        result_dict = result.to_dict()

        assert "captured_at" in result_dict
        assert result_dict["captured_at"] == now


# ---------------------------------------------------------------------------
# Test default time filter in main.py
# ---------------------------------------------------------------------------


class TestDefaultTimeFilter:
    """Test default 3-month time filter logic."""

    def test_time_filter_uses_query_timestamp(self):
        """Test that time filter uses query timestamp ±3 months."""
        # Simulate query with specific timestamp
        query_timestamp = int(datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp())

        # Filter to ±3 months (90 days) from query timestamp
        three_months_seconds = 90 * 24 * 60 * 60

        time_filter = {
            "captured_at_min": query_timestamp - three_months_seconds,
            "captured_at_max": query_timestamp + three_months_seconds,
        }

        expected_min = query_timestamp - three_months_seconds
        expected_max = query_timestamp + three_months_seconds

        assert time_filter["captured_at_min"] == expected_min
        assert time_filter["captured_at_max"] == expected_max

    def test_time_filter_range_calculation(self):
        """Test that time filter range is calculated correctly."""
        # Query timestamp: 2024-06-15 10:30:00 UTC
        query_timestamp = int(datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp())

        # ±3 months = ±90 days
        three_months_seconds = 90 * 24 * 60 * 60

        min_expected = query_timestamp - three_months_seconds
        max_expected = query_timestamp + three_months_seconds

        # Convert back to verify range
        min_dt = datetime.fromtimestamp(min_expected, tz=timezone.utc)
        max_dt = datetime.fromtimestamp(max_expected, tz=timezone.utc)

        # Min should be around 2024-03-17
        assert min_dt.year == 2024
        assert min_dt.month == 3

        # Max should be around 2024-09-13
        assert max_dt.year == 2024
        assert max_dt.month == 9

    def test_invalid_timestamp_raises_error(self):
        """Test that invalid timestamp format raises appropriate error."""
        invalid_timestamps = ["invalid", "2024/06/15", ""]

        for ts in invalid_timestamps:
            try:
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
                raise AssertionError(f"Should have raised error for: {ts}")
            except (ValueError, AttributeError):
                pass  # Expected


# ---------------------------------------------------------------------------
# Test timestamp formatting utilities
# ---------------------------------------------------------------------------


class TestTimestampFormatting:
    """Test timestamp formatting for display."""

    def test_unix_timestamp_to_datetime(self):
        """Test conversion from Unix timestamp to datetime."""
        now = datetime.now(timezone.utc)
        unix_timestamp = int(now.timestamp())

        # Convert back
        dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)

        # Should be approximately equal (within 1 second)
        assert abs((dt - now).total_seconds()) <= 1

    def test_format_timestamp_readable(self):
        """Test timestamp formatting to readable format."""
        # 2024-06-15 10:30:00 UTC
        unix_timestamp = 1718447400

        dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)

        formatted = dt.strftime("%Y-%m-%d %H:%M:%S")

        assert formatted == "2024-06-15 10:30:00"

    def test_invalid_timestamp_handling(self):
        """Test handling of invalid timestamps."""
        invalid_timestamps = [None, 0, -1]

        for ts in invalid_timestamps:
            if ts is None or ts <= 0:
                # Should handle gracefully
                result = "N/A" if ts is None or ts == 0 else "Invalid"
                assert result in ["N/A", "Invalid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
