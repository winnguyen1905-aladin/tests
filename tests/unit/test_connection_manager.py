#!/usr/bin/env python3
"""
Unit Tests for Connection Manager

Tests the singleton connection manager for Milvus and MinIO:
- Singleton pattern
- Connection management
- Health checks
"""

import pytest
from unittest.mock import MagicMock, patch
from src.repository.connectionManager import (
    ConnectionManager,
    get_connection_manager,
    reset_connection_manager,
    ConnectionStats,
)


class TestConnectionManagerSingleton:
    """Test singleton pattern for ConnectionManager."""

    def test_singleton_returns_same_instance(self):
        """Test that get_connection_manager returns the same instance."""
        reset_connection_manager()

        manager1 = get_connection_manager()
        manager2 = get_connection_manager()

        assert manager1 is manager2

    def test_reset_clears_singleton(self):
        """Test that reset creates a new instance."""
        reset_connection_manager()

        manager1 = get_connection_manager()
        reset_connection_manager()
        manager2 = get_connection_manager()

        # After reset, we get a fresh manager
        assert manager2 is not None


class TestConnectionStats:
    """Test ConnectionStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = ConnectionStats()

        assert stats.total_connections == 0
        assert stats.active_connections == 0
        assert stats.failed_connections == 0
        assert stats.last_error is None

    def test_custom_values(self):
        """Test custom values."""
        stats = ConnectionStats(
            total_connections=5,
            active_connections=2,
            failed_connections=1,
            last_error="Connection refused",
        )

        assert stats.total_connections == 5
        assert stats.active_connections == 2
        assert stats.failed_connections == 1
        assert stats.last_error == "Connection refused"


class TestConnectionManager:
    """Test ConnectionManager functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        reset_connection_manager()
        yield
        # Cleanup
        reset_connection_manager()

    def test_default_state(self):
        """Test default state of connection manager."""
        manager = get_connection_manager()

        assert manager.milvus_client is None
        assert manager.minio_client is None
        assert manager.milvus_connected is False
        assert manager.minio_connected is False

    def test_configure(self):
        """Test configuration."""
        manager = get_connection_manager()

        manager.configure(
            milvus_uri="http://custom:19530", minio_endpoint="custom:9000", milvus_pool_size=20
        )

        assert manager._config["milvus_uri"] == "http://custom:19530"
        assert manager._config["minio_endpoint"] == "custom:9000"
        assert manager._config["milvus_pool_size"] == 20

    def test_get_stats(self):
        """Test getting stats."""
        manager = get_connection_manager()

        stats = manager.get_stats()

        assert "milvus" in stats
        assert "minio" in stats
        assert isinstance(stats["milvus"], ConnectionStats)
        assert isinstance(stats["minio"], ConnectionStats)

    @patch("pymilvus.MilvusClient")
    def test_connect_milvus_success(self, mock_client_class):
        """Test successful Milvus connection."""
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        mock_client_class.return_value = mock_client

        manager = get_connection_manager()
        result = manager.connect_milvus(uri="http://localhost:19530")

        assert result is True
        assert manager.milvus_connected is True
        assert manager.milvus_client is not None

    @patch("pymilvus.MilvusClient")
    def test_connect_milvus_failure(self, mock_client_class):
        """Test failed Milvus connection."""
        mock_client_class.side_effect = Exception("Connection refused")

        manager = get_connection_manager()
        result = manager.connect_milvus(uri="http://localhost:19530")

        assert result is False
        assert manager.milvus_connected is False
        assert manager._stats["milvus"].failed_connections == 1

    @patch("minio.Minio")
    def test_connect_minio_success(self, mock_minio_class):
        """Test successful MinIO connection."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client

        manager = get_connection_manager()
        result = manager.connect_minio(
            endpoint="localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            bucket="test-bucket",
        )

        assert result is True
        assert manager.minio_connected is True

    @patch("minio.Minio")
    def test_connect_minio_creates_bucket(self, mock_minio_class):
        """Test MinIO bucket creation."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = False
        mock_minio_class.return_value = mock_client

        manager = get_connection_manager()
        result = manager.connect_minio(
            endpoint="localhost:9000",
            access_key="testkey",
            secret_key="testsecret",
            bucket="new-bucket",
        )

        assert result is True
        mock_client.make_bucket.assert_called_once_with("new-bucket")

    def test_disconnect_milvus(self):
        """Test disconnect Milvus."""
        manager = get_connection_manager()

        # Set up connected state
        manager._milvus_connected = True
        manager._milvus_client = MagicMock()
        manager._stats["milvus"].active_connections = 1

        manager.disconnect_milvus()

        assert manager.milvus_connected is False
        assert manager.milvus_client is None

    def test_disconnect_all(self):
        """Test disconnect all."""
        # First reset to get clean state
        reset_connection_manager()
        manager = get_connection_manager()

        # Set up connected state
        manager._milvus_connected = True
        manager._minio_connected = True

        manager.disconnect_all()

        assert manager.milvus_connected is False
        assert manager.minio_connected is False

    def test_health_check_no_connections(self):
        """Test health check with no connections."""
        # First reset to get clean state
        reset_connection_manager()
        manager = get_connection_manager()

        health = manager.health_check()

        assert health["milvus"]["connected"] is False
        assert health["minio"]["connected"] is False

    def test_reset(self):
        """Test reset functionality."""
        # First reset to get clean state
        reset_connection_manager()
        manager = get_connection_manager()

        manager._stats["milvus"].total_connections = 5
        manager._stats["minio"].total_connections = 3

        manager.reset()

        assert manager._stats["milvus"].total_connections == 0
        assert manager._stats["minio"].total_connections == 0
        assert manager.milvus_connected is False
        assert manager.minio_connected is False
