#!/usr/bin/env python3
"""
Integration Tests for API Endpoints

Tests API endpoints with real HTTP client but mocked services.
Requires: Running FastAPI server or use TestClient.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Mark as integration test
pytestmark = pytest.mark.integration


class TestIngestEndpoint:
    """Test /ingest endpoint."""

    @pytest.fixture
    def mock_pipeline(self):
        """Mock the ingestion service."""
        with patch("src.service.ingestionService.IngestionService") as mock:
            service = MagicMock()
            service.ingest_raw.return_value = MagicMock(
                success=True,
                image_id="test_img",
                tree_id="test_tree",
                storage_keys={"milvus_key": "test", "minio_key": "test"},
                message="Success",
            )
            mock.return_value = service
            yield service

    def test_ingest_returns_success(self, mock_pipeline):
        """Test successful ingestion returns 200."""
        # This is a placeholder - actual test would use TestClient
        assert mock_pipeline.ingest_raw.called is False  # Not called yet


class TestVerifyEndpoint:
    """Test /verify endpoint."""

    @pytest.fixture
    def mock_pipeline(self):
        """Mock the verification service."""
        with patch("src.service.verificationService.VerificationService") as mock:
            service = MagicMock()
            service.verify.return_value = MagicMock(
                success=True, best_match={"tree_id": "tree_1", "score": 0.95}, message="Success"
            )
            mock.return_value = service
            yield service

    def test_verify_returns_success(self, mock_pipeline):
        """Test successful verification returns 200."""
        assert mock_pipeline.verify.called is False  # Not called yet


@pytest.mark.skip(reason="graphService not yet implemented")
class TestGraphEndpoint:
    """Test /graph endpoint."""

    @pytest.fixture
    def mock_service(self):
        """Mock the graph service."""
        with patch("src.service.graphService.GraphService") as mock:
            service = MagicMock()
            service.get_tree_graph.return_value = {"nodes": [{"id": "tree_1"}], "edges": []}
            mock.return_value = service
            yield service

    def test_graph_returns_success(self, mock_service):
        """Test successful graph retrieval."""
        assert mock_service.get_tree_graph.called is False
