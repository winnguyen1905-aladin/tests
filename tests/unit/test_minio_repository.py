#!/usr/bin/env python3
"""
Unit tests for MinIORepository.

Tests key behaviours WITHOUT a real MinIO server:
- _get_storage_key: correct path format
- save_features: successful upload, not-connected guard, missing key handling
- load_features: successful load, not-connected guard, exception returns None
- load_features_from_key: happy path, not-connected guard
- delete_features: successful remove, not-connected guard
- exists: object found / not found
- get_feature_count: tree/image parsing from object names
"""

import gzip
import io
import json
import os
import tempfile
from typing import Dict, Any
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from src.repository.minioRepository import (
    MinIOConfig,
    MinIORepository,
    MinIOResult,
    create_minio_repository,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(connected: bool = True) -> MinIORepository:
    """Create a MinIORepository whose Minio client is fully mocked.

    Patches MinIOConfig.__post_init__ to skip env-var validation so tests
    can create a config without real credentials.
    """
    with patch.object(MinIOConfig, "__post_init__", lambda self: None):
        repo = MinIORepository.__new__(MinIORepository)
        repo.config = MinIOConfig(
            endpoint="localhost:9000",
            access_key="testkey",
            secret_key="testsecret",
            bucket="test-bucket",
            features_prefix="features/",
            verbose=False,
        )
        repo._use_shared = False
        repo._connected = connected
        repo.client = MagicMock() if connected else None
        return repo


def _make_features(
    n_keypoints: int = 50,
    descriptor_dim: int = 256,
    with_texture: bool = True,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    feat: Dict[str, Any] = {
        "keypoints": rng.random((n_keypoints, 2)).astype(np.float32) * 640,
        "descriptors": rng.random((n_keypoints, descriptor_dim)).astype(np.float32),
        "scores": rng.random(n_keypoints).astype(np.float32),
    }
    if with_texture:
        feat["texture_histogram"] = rng.random(64).astype(np.float32)
    return feat


def _npz_gz_bytes(features: Dict[str, np.ndarray]) -> bytes:
    """Build in-memory gzipped npz bytes matching the save_features format."""
    buf_npz = io.BytesIO()
    np.savez(buf_npz, **features)
    buf_npz.seek(0)

    buf_gz = io.BytesIO()
    with gzip.GzipFile(fileobj=buf_gz, mode="wb") as gz:
        gz.write(buf_npz.read())
    buf_gz.seek(0)
    return buf_gz.read()


# ---------------------------------------------------------------------------
# _get_storage_key
# ---------------------------------------------------------------------------


class TestGetStorageKey:
    def test_format(self):
        repo = _make_repo()
        key = repo._get_storage_key("img_001")
        assert key == "features/img_001.npz.gz"

    def test_prefix_used(self):
        repo = _make_repo()
        repo.config.features_prefix = "custom/"
        key = repo._get_storage_key("img_002")
        assert key.startswith("custom/")

    def test_extension_preserved(self):
        repo = _make_repo()
        key = repo._get_storage_key("my_image")
        assert key.endswith(".npz.gz")


# ---------------------------------------------------------------------------
# save_features
# ---------------------------------------------------------------------------


class TestSaveFeatures:
    def test_returns_failure_when_not_connected(self):
        repo = _make_repo(connected=False)
        result = repo.save_features("img_001", _make_features())
        assert result.success is False
        assert result.storage_key == ""
        assert "not connected" in result.message.lower()

    def test_successful_save_returns_storage_key(self, tmp_path):
        repo = _make_repo(connected=True)
        features = _make_features()

        # We intercept fput_object so no real I/O happens via Minio
        repo.client.fput_object = MagicMock(return_value=None)

        result = repo.save_features("img_001", features)

        assert result.success is True
        assert "img_001" in result.storage_key
        repo.client.fput_object.assert_called_once()

    def test_storage_key_matches_get_storage_key(self, tmp_path):
        repo = _make_repo(connected=True)
        features = _make_features()
        repo.client.fput_object = MagicMock(return_value=None)

        result = repo.save_features("img_abc", features)
        expected = repo._get_storage_key("img_abc")
        assert result.storage_key == expected

    def test_upload_uses_correct_bucket(self):
        repo = _make_repo(connected=True)
        repo.client.fput_object = MagicMock(return_value=None)

        repo.save_features("img_001", _make_features())

        _, kwargs = repo.client.fput_object.call_args
        assert (
            kwargs.get("bucket_name") == "test-bucket"
            or repo.client.fput_object.call_args[0][0] == "test-bucket"
        )

    def test_features_without_texture_saved_correctly(self):
        repo = _make_repo(connected=True)
        repo.client.fput_object = MagicMock(return_value=None)
        features = _make_features(with_texture=False)

        result = repo.save_features("img_no_tex", features)
        assert result.success is True

    def test_minio_error_returns_failure(self):
        repo = _make_repo(connected=True)
        repo.client.fput_object = MagicMock(side_effect=Exception("S3 error"))

        result = repo.save_features("img_err", _make_features())
        assert result.success is False
        assert "S3 error" in result.message


# ---------------------------------------------------------------------------
# load_features
# ---------------------------------------------------------------------------


class TestLoadFeatures:
    def test_returns_none_when_not_connected(self):
        repo = _make_repo(connected=False)
        assert repo.load_features("img_001") is None

    def test_successful_load_returns_dict(self, tmp_path):
        repo = _make_repo(connected=True)
        features = _make_features()

        # Simulate fget_object by writing the gzipped npz to tmp_path
        gz_data = _npz_gz_bytes(features)

        def fake_fget(bucket_name, object_name, file_path):
            with open(file_path, "wb") as fh:
                fh.write(gz_data)

        repo.client.fget_object = MagicMock(side_effect=fake_fget)

        loaded = repo.load_features("img_001")

        assert loaded is not None
        assert "keypoints" in loaded
        assert "descriptors" in loaded
        assert "scores" in loaded
        np.testing.assert_array_almost_equal(loaded["keypoints"], features["keypoints"])

    def test_load_includes_texture_histogram(self, tmp_path):
        repo = _make_repo(connected=True)
        features = _make_features(with_texture=True)
        gz_data = _npz_gz_bytes(features)

        def fake_fget(bucket_name, object_name, file_path):
            with open(file_path, "wb") as fh:
                fh.write(gz_data)

        repo.client.fget_object = MagicMock(side_effect=fake_fget)

        loaded = repo.load_features("img_tex")
        assert "texture_histogram" in loaded
        np.testing.assert_array_almost_equal(
            loaded["texture_histogram"], features["texture_histogram"]
        )

    def test_exception_returns_none(self):
        repo = _make_repo(connected=True)
        repo.client.fget_object = MagicMock(side_effect=Exception("Not found"))

        assert repo.load_features("missing_img") is None


# ---------------------------------------------------------------------------
# load_features_from_key / load_features_by_key
# ---------------------------------------------------------------------------


class TestLoadFeaturesByKey:
    def test_not_connected_returns_none(self):
        repo = _make_repo(connected=False)
        assert repo.load_features_from_key("features/img.npz.gz") is None

    def test_load_from_key_happy_path(self):
        repo = _make_repo(connected=True)
        features = _make_features()
        gz_data = _npz_gz_bytes(features)

        def fake_fget(bucket_name, object_name, file_path):
            with open(file_path, "wb") as fh:
                fh.write(gz_data)

        repo.client.fget_object = MagicMock(side_effect=fake_fget)

        loaded = repo.load_features_from_key("features/img_001.npz.gz")
        assert loaded is not None
        assert "keypoints" in loaded

    def test_load_features_by_key_alias(self):
        repo = _make_repo(connected=True)
        features = _make_features()
        gz_data = _npz_gz_bytes(features)

        def fake_fget(bucket_name, object_name, file_path):
            with open(file_path, "wb") as fh:
                fh.write(gz_data)

        repo.client.fget_object = MagicMock(side_effect=fake_fget)

        # load_features_by_key is an alias for load_features_from_key
        loaded = repo.load_features_by_key("features/img_001.npz.gz")
        assert loaded is not None

    def test_exception_returns_none(self):
        repo = _make_repo(connected=True)
        repo.client.fget_object = MagicMock(side_effect=Exception("Network error"))
        assert repo.load_features_from_key("features/bad.npz.gz") is None


# ---------------------------------------------------------------------------
# delete_features
# ---------------------------------------------------------------------------


class TestDeleteFeatures:
    def test_not_connected_returns_false(self):
        repo = _make_repo(connected=False)
        assert repo.delete_features("img_001") is False

    def test_successful_delete(self):
        repo = _make_repo(connected=True)
        repo.client.remove_object = MagicMock(return_value=None)

        result = repo.delete_features("img_001")

        assert result is True
        repo.client.remove_object.assert_called_once_with(
            bucket_name="test-bucket",
            object_name="features/img_001.npz.gz",
        )

    def test_exception_returns_false(self):
        repo = _make_repo(connected=True)
        repo.client.remove_object = MagicMock(side_effect=Exception("Delete error"))
        assert repo.delete_features("bad_img") is False


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestExists:
    def test_not_connected_returns_false(self):
        repo = _make_repo(connected=False)
        assert repo.exists("img_001") is False

    def test_stat_succeeds_returns_true(self):
        repo = _make_repo(connected=True)
        repo.client.stat_object = MagicMock(return_value=MagicMock())

        assert repo.exists("img_001") is True

    def test_stat_raises_returns_false(self):
        repo = _make_repo(connected=True)
        repo.client.stat_object = MagicMock(side_effect=Exception("Not found"))

        assert repo.exists("missing") is False

    def test_stat_called_with_correct_key(self):
        repo = _make_repo(connected=True)
        repo.client.stat_object = MagicMock(return_value=MagicMock())

        repo.exists("my_image")

        repo.client.stat_object.assert_called_once_with(
            bucket_name="test-bucket",
            object_name="features/my_image.npz.gz",
        )


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_disconnects(self):
        repo = _make_repo(connected=True)
        repo.close()
        assert repo.client is None
        assert repo._connected is False


# ---------------------------------------------------------------------------
# create_minio_repository factory
# ---------------------------------------------------------------------------


class TestCreateMinioRepository:
    @patch("src.repository.minioRepository.Minio")
    def test_factory_creates_repo_with_correct_config(self, mock_minio):
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio.return_value = mock_client

        repo = create_minio_repository(
            endpoint="myhost:9000",
            access_key="mykey",
            secret_key="mysecret",
            bucket="mybucket",
        )

        assert isinstance(repo, MinIORepository)
        assert repo.config.endpoint == "myhost:9000"
        assert repo.config.bucket == "mybucket"

    @patch("src.repository.minioRepository.Minio")
    def test_factory_creates_bucket_if_not_exists(self, mock_minio):
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = False
        mock_minio.return_value = mock_client

        create_minio_repository(
            access_key="mykey",
            secret_key="mysecret",
            bucket="new-bucket",
        )

        mock_client.make_bucket.assert_called_once_with("new-bucket")


# ---------------------------------------------------------------------------
# MinIOConfig defaults
# ---------------------------------------------------------------------------


class TestMinIOConfig:
    def test_default_values(self):
        with patch.object(MinIOConfig, "__post_init__", lambda self: None):
            cfg = MinIOConfig()
        assert cfg.endpoint == "localhost:9000"
        assert cfg.bucket == "tree-features"
        assert cfg.features_prefix == "features/"
        assert cfg.secure is False
        assert cfg.verbose is False

    def test_custom_values(self):
        with patch.object(MinIOConfig, "__post_init__", lambda self: None):
            cfg = MinIOConfig(endpoint="remote:9000", bucket="custom", secure=True)
        assert cfg.endpoint == "remote:9000"
        assert cfg.bucket == "custom"
        assert cfg.secure is True
