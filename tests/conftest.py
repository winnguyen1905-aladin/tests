#!/usr/bin/env python3
"""
Pytest Configuration and Shared Fixtures

This module provides shared fixtures for all tests:
- Configuration fixtures
- Mock services and processors
- Sample data fixtures
- Database fixtures
"""

import sys
import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from typing import Generator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.appConfig import AppConfig
from src.config import Settings, reload_settings


# ─── Configuration Fixtures ───────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Create test settings instance."""
    reload_settings()  # Clear cache
    settings = Settings(
        env="dev",
        milvus_uri="http://localhost:19530",
        minio_endpoint="localhost:9000",
        log_level="DEBUG",
    )
    return settings


@pytest.fixture(scope="function")
def test_config() -> AppConfig:
    """Create test AppConfig instance."""
    return AppConfig(
        verbose=True,
        log_level="DEBUG",
        milvus_collection="test_tree_features",
    )


# ─── Image Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample RGB image for testing."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask() -> np.ndarray:
    """Create a sample binary mask for testing."""
    return np.ones((480, 640), dtype=np.uint8) * 255


@pytest.fixture
def sample_grayscale() -> np.ndarray:
    """Create a sample grayscale image for testing."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (480, 640), dtype=np.uint8)


# ─── Mock Processor Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_dino_processor():
    """Create a mock DINO processor."""
    processor = MagicMock()
    processor.extract = MagicMock(
        return_value=MagicMock(
            global_descriptor=np.random.rand(1280).astype(np.float32),
            image_size=(480, 640),
            model_name="mock_dino",
        )
    )
    processor.config = MagicMock()
    processor.config.device = "cpu"
    return processor


@pytest.fixture
def mock_superpoint_processor():
    """Create a mock SuperPoint processor."""
    processor = MagicMock()
    processor.extract = MagicMock(
        return_value=MagicMock(
            keypoints=np.random.rand(100, 2).astype(np.float32),
            descriptors=np.random.rand(100, 256).astype(np.float32),
            scores=np.random.rand(100).astype(np.float32),
            image_size=(480, 640),
        )
    )
    processor.config = MagicMock()
    processor.config.device = "cpu"
    return processor


@pytest.fixture
def mock_sam3_processor():
    """Create a mock SAM3 processor."""
    processor = MagicMock()
    processor.segment = MagicMock(
        return_value=(
            np.ones((480, 640), dtype=np.uint8) * 255,  # mask
            [{"segmentation": np.ones((480, 640), dtype=np.uint8), "bbox": [0, 0, 640, 480]}],
        )
    )
    processor.config = MagicMock()
    return processor


# ─── Repository Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def mock_milvus_repo():
    """Create a mock Milvus repository."""
    repo = MagicMock()
    repo.insert = MagicMock(return_value="test_milvus_key")
    repo.search = MagicMock(
        return_value=MagicMock(
            ids=["img_1", "img_2"],
            distances=[0.95, 0.90],
            tree_ids=["tree_1", "tree_1"],
        )
    )
    repo.get_by_id = MagicMock(
        return_value={
            "id": "img_1",
            "tree_id": "tree_1",
            "global_vector": np.random.rand(1280).astype(np.float32),
        }
    )
    return repo


@pytest.fixture
def mock_minio_repo():
    """Create a mock MinIO repository."""
    repo = MagicMock()
    mock_result = MagicMock()
    mock_result.storage_key = "test_minio_key"
    mock_result.success = True
    repo.save_features = MagicMock(return_value=mock_result)
    repo.get_features = MagicMock(
        return_value={
            "keypoints": np.random.rand(100, 2).astype(np.float32),
            "descriptors": np.random.rand(100, 256).astype(np.float32),
            "scores": np.random.rand(100).astype(np.float32),
        }
    )
    return repo


# ─── Service Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def mock_preprocessor():
    """Create a mock preprocessor."""
    preprocessor = MagicMock()
    preprocessor.apply_mask = MagicMock(side_effect=lambda img, m: img)
    preprocessor.get_bounding_box = MagicMock(return_value=(0, 0, 640, 480))
    preprocessor.crop_to_bounding_box = MagicMock(
        side_effect=lambda img, bbox, mask=None: (img, mask, bbox)
    )
    preprocessor.prepare_for_dino = MagicMock(side_effect=lambda img: img)
    preprocessor.prepare_for_superpoint = MagicMock(side_effect=lambda img: (img, 1.0))
    preprocessor.to_grayscale = MagicMock(side_effect=lambda img: img)
    preprocessor.segment_with_sam3 = MagicMock(
        return_value=(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.ones((480, 640), dtype=np.uint8) * 255,
        )
    )
    preprocessor.segment_with_sam3_box = MagicMock(
        return_value=(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.ones((480, 640), dtype=np.uint8) * 255,
        )
    )
    return preprocessor


# ─── Result Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def dino_result():
    """Create a sample DINO result."""
    from src.processor.dinoProcessor import DinoResult

    return DinoResult(
        global_descriptor=np.random.rand(1280).astype(np.float32),
        image_size=(480, 640),
        model_name="test_dino",
    )


@pytest.fixture
def superpoint_result():
    """Create a sample SuperPoint result."""
    from src.processor.superPointProcessor import SuperPointResult

    return SuperPointResult(
        keypoints=np.random.rand(100, 2).astype(np.float32),
        descriptors=np.random.rand(100, 256).astype(np.float32),
        scores=np.random.rand(100).astype(np.float32),
        image_size=(480, 640),
    )


# ─── Test Data Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def sample_tree_ids() -> list:
    """Sample tree IDs for testing."""
    return ["tree_1", "tree_2", "tree_3"]


@pytest.fixture
def sample_image_ids() -> list:
    """Sample image IDs for testing."""
    return ["img_001", "img_002", "img_003"]


# ─── Pytest Configuration ─────────────────────────────────────────────────────


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests (require Milvus/MinIO)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (full flow testing)")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add unit marker to tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add integration marker to tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        # Add e2e marker to tests in e2e/ directory
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
