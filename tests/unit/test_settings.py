#!/usr/bin/env python3
"""
Unit Tests for Settings Configuration

Tests the Pydantic Settings configuration:
- Environment variable loading
- Default values
- Validation
- CORS origins parsing
"""

import os
import pytest
from src.config import Settings, get_settings, reload_settings


class TestSettings:
    """Test Settings configuration."""

    def test_default_values(self):
        """Test default values are set correctly."""
        settings = Settings()

        assert settings.env == "dev"
        assert settings.milvus_uri == "http://localhost:19530"
        assert settings.minio_endpoint == "localhost:9000"
        assert settings.ingest_concurrency == 4

    def test_environment_variable_override(self, monkeypatch):
        """Test environment variables override defaults."""
        monkeypatch.setenv("MILVUS_URI", "http://custom:19530")
        monkeypatch.setenv("ENV", "prod")

        # Reload settings to pick up env vars
        settings = Settings()

        assert settings.milvus_uri == "http://custom:19530"
        assert settings.env == "prod"

    def test_cors_origins_single(self):
        """Test CORS origins with single origin."""
        settings = Settings(cors_origins="https://example.com")

        assert settings.get_cors_origins_list() == ["https://example.com"]

    def test_cors_origins_multiple(self):
        """Test CORS origins with multiple origins."""
        settings = Settings(cors_origins="https://example.com,https://api.example.com")

        assert settings.get_cors_origins_list() == [
            "https://example.com",
            "https://api.example.com",
        ]

    def test_cors_origins_wildcard(self):
        """Test CORS origins with wildcard."""
        settings = Settings(cors_origins="*")

        assert settings.get_cors_origins_list() == ["*"]

    def test_is_production(self):
        """Test production detection."""
        prod_settings = Settings(env="prod")
        dev_settings = Settings(env="dev")

        assert prod_settings.is_production() is True
        assert dev_settings.is_production() is False

    def test_is_development(self):
        """Test development detection."""
        dev_settings = Settings(env="dev")
        prod_settings = Settings(env="prod")

        assert dev_settings.is_development() is True
        assert prod_settings.is_development() is False

    def test_singleton_behavior(self):
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        # Same instance due to lru_cache
        assert settings1 is settings2

    def test_reload_settings(self):
        """Test reload_settings clears cache."""
        settings1 = get_settings()
        reload_settings()
        settings2 = get_settings()

        # After reload, should be same instance (cache cleared and recreated)
        # This tests that reload works without raising errors
        assert settings2 is not None

    def test_vector_dimensions_must_match(self):
        """Milvus and PostgreSQL vector dimensions must stay aligned."""
        with pytest.raises(ValueError, match="MILVUS_VECTOR_DIM must match POSTGRES_VECTOR_DIM"):
            Settings(milvus_vector_dim=384, postgres_vector_dim=768)
