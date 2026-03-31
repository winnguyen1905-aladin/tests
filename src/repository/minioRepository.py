#!/usr/bin/env python3
"""
MinIO Repository - Object Storage for Local Features

Stores and retrieves SuperPoint local features (keypoints + descriptors) as compressed .npz files.
"""

import gzip
import json
import logging
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from minio import Minio

logger = logging.getLogger(__name__)

# Suppress urllib3 connection retry warnings (not an error, just noisy)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


@dataclass
class MinIOConfig:
    """Configuration for MinIO repository.

    Credentials MUST be provided via environment variables:
    - MINIO_ENDPOINT: Server endpoint (host:port)
    - MINIO_ACCESS_KEY: Access key
    - MINIO_SECRET_KEY: Secret key
    - MINIO_BUCKET: Bucket name (default: tree-features)
    - MINIO_SECURE: Use HTTPS (default: false)
    """
    # Get from environment - no hardcoded defaults for security
    endpoint: str = field(default_factory=lambda: os.getenv("MINIO_ENDPOINT", "localhost:9000"))
    access_key: str = field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY", ""))
    bucket: str = field(default_factory=lambda: os.getenv("MINIO_BUCKET", "tree-features"))
    secure: bool = field(default_factory=lambda: os.getenv("MINIO_SECURE", "false").lower() == "true")
    features_prefix: str = "features/"  # Prefix for feature files
    verbose: bool = False

    def __post_init__(self):
        """Validate required credentials after initialization."""
        if not self.access_key:
            raise ValueError("MINIO_ACCESS_KEY environment variable is required")
        if not self.secret_key:
            raise ValueError("MINIO_SECRET_KEY environment variable is required")


@dataclass
class MinIOResult:
    """Result from MinIO operation."""
    success: bool
    storage_key: str
    message: str


class MinIORepository:
    """Repository for storing and retrieving local features from MinIO/S3.

    Supports two connection modes:
    - Standalone: Creates its own connection (default, backward compatible)
    - Shared: Uses ConnectionManager singleton (recommended for production)
    """

    # Class-level reference to shared connection manager
    _use_shared_connection: bool = False

    def __init__(
        self,
        config: Optional[MinIOConfig] = None,
        use_shared_connection: bool = False
    ) -> None:
        """Initialize MinIO repository.

        Args:
            config: Optional configuration. Uses defaults if not provided.
            use_shared_connection: If True, use ConnectionManager singleton
        """
        self.config = config or MinIOConfig()
        self.client = None
        self._connected = False

        # Connection mode
        self._use_shared = use_shared_connection or MinIORepository._use_shared_connection

        if self._use_shared:
            self._init_shared_client()
        else:
            self._init_client()

    def _init_shared_client(self) -> None:
        """Initialize MinIO client using shared ConnectionManager."""
        from src.repository.connectionManager import get_connection_manager

        if self.config.verbose:
            print("Using shared ConnectionManager for MinIO...")

        manager = get_connection_manager()
        manager.configure(
            minio_endpoint=self.config.endpoint,
            minio_access_key=self.config.access_key,
            minio_secret_key=self.config.secret_key,
            minio_bucket=self.config.bucket
        )

        if manager.connect_minio(
            endpoint=self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            bucket=self.config.bucket,
            secure=self.config.secure
        ):
            self.client = manager.minio_client
            self._connected = True

            if self.config.verbose:
                print("✓ MinIO client connected via shared manager!")
        else:
            self._connected = False
            if self.config.verbose:
                print("✗ Failed to connect to MinIO via shared manager")

    # ─── Class Methods for Shared Connection Mode ─────────────────────────────────

    @classmethod
    def use_shared_connections(cls, enabled: bool = True) -> None:
        """Enable or disable shared connection mode for all MinIORepository instances.

        When enabled, all instances will use the ConnectionManager singleton
        instead of creating their own connections.

        Args:
            enabled: True to use shared connections, False for standalone
        """
        cls._use_shared_connection = enabled
        logger.info(f"MinIORepository shared connections: {enabled}")

    @classmethod
    def get_connection_manager(cls):
        """Get the shared ConnectionManager instance.

        Returns:
            The ConnectionManager singleton
        """
        from src.repository.connectionManager import get_connection_manager
        return get_connection_manager()

    def _init_client(self) -> None:
        """Initialize MinIO client."""
        if self.config.verbose:
            print(f"Initializing MinIO client...")
            print(f"Endpoint: {self.config.endpoint}")
            print(f"Bucket: {self.config.bucket}")

        try:

            self.client = Minio(
                self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure
            )

            # Check if bucket exists, create if not
            if not self.client.bucket_exists(self.config.bucket):
                self.client.make_bucket(self.config.bucket)
                if self.config.verbose:
                    print(f"Created bucket: {self.config.bucket}")
            else:
                if self.config.verbose:
                    print(f"Bucket exists: {self.config.bucket}")

            self._connected = True
            if self.config.verbose:
                print("MinIO client connected successfully!")

        except ImportError as e:
            raise ImportError(
                "minio package is required. "
                "Install with: pip install minio"
            ) from e
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not connect to MinIO: {e}")
            self._connected = False

    def _get_storage_key(self, image_id: str) -> str:
        """Generate storage key for an image ID.

        Args:
            image_id: Unique image identifier

        Returns:
            Storage key path
        """
        return f"{self.config.features_prefix}{image_id}.npz.gz"

    def save_features(
        self,
        image_id: str,
        features: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> MinIOResult:
        """Save SuperPoint features to MinIO.

        Args:
            image_id: Unique identifier for the image
            features: Dict with 'keypoints', 'descriptors', 'scores'
            metadata: Optional metadata dict

        Returns:
            MinIOResult with storage key
        """
        if not self._connected:
            return MinIOResult(
                success=False,
                storage_key="",
                message="MinIO not connected"
            )

        try:
            storage_key = self._get_storage_key(image_id)

            # Prepare data for saving
            save_data = {
                'keypoints': features['keypoints'],
                'descriptors': features['descriptors'],
                'scores': features['scores'],
            }

            # Include texture histogram if available
            if 'texture_histogram' in features and features['texture_histogram'] is not None:
                save_data['texture_histogram'] = features['texture_histogram']

            if metadata:
                save_data['metadata'] = json.dumps(metadata)

            # Create temporary directory for safe file handling
            temp_dir = tempfile.mkdtemp(prefix="sam3_save_")
            tmp_path = None
            gzip_path = None

            try:
                # Create temp file path
                tmp_path = os.path.join(temp_dir, 'features.npz')
                gzip_path = os.path.join(temp_dir, 'features.npz.gz')

                # Save to npz
                np.savez(tmp_path, **save_data)

                # Compress with gzip
                with open(tmp_path, 'rb') as f_in:
                    with gzip.open(gzip_path, 'wb') as f_out:
                        f_out.write(f_in.read())

                # Upload to MinIO
                self.client.fput_object(
                    bucket_name=self.config.bucket,
                    object_name=storage_key,
                    file_path=gzip_path,
                    content_type='application/gzip'
                )

                if self.config.verbose:
                    print(f"Saved features for {image_id}: {storage_key}")

                return MinIOResult(
                    success=True,
                    storage_key=storage_key,
                    message="Features saved successfully"
                )

            finally:
                # Always clean up temp directory
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        # Remove both files if they exist
                        if tmp_path and os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        if gzip_path and os.path.exists(gzip_path):
                            os.unlink(gzip_path)
                        # Remove temp directory and any remaining files
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except OSError:
                        pass  # Ignore cleanup errors

        except Exception as e:
            if self.config.verbose:
                print(f"Error saving features to MinIO: {e}")
            return MinIOResult(
                success=False,
                storage_key="",
                message=str(e)
            )

    def load_features(
        self,
        image_id: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load SuperPoint features from MinIO.

        Args:
            image_id: Unique identifier for the image

        Returns:
            Dict with 'keypoints', 'descriptors', 'scores' or None if not found
        """
        if not self._connected:
            if self.config.verbose:
                print("Warning: MinIO not connected")
            return None
        try:
            # Validate and sanitize image_id to prevent path traversal
            if not image_id or not isinstance(image_id, str):
                raise ValueError("Invalid image_id: must be a non-empty string")
            if '..' in image_id or '/' in image_id or '\\' in image_id:
                raise ValueError("Invalid image_id: path traversal not allowed")

            storage_key = self._get_storage_key(image_id)

            # Use secure temp directory with unique filename
            temp_dir = tempfile.mkdtemp(prefix="sam3_minio_")
            tmp_path = os.path.join(temp_dir, f'features.npz.gz')

            try:
                self.client.fget_object(
                    bucket_name=self.config.bucket,
                    object_name=storage_key,
                    file_path=tmp_path
                )

                # Decompress and load
                with gzip.open(tmp_path, 'rb') as f:
                    try:
                        data = np.load(f, allow_pickle=False)
                    except ValueError as e:
                        logger.error(
                            f"Refused to load {storage_key}: file may contain pickled objects "
                            f"(allow_pickle is disabled for security). Error: {e}"
                        )
                        return None
                    features = {
                        'keypoints': data['keypoints'],
                        'descriptors': data['descriptors'],
                        'scores': data['scores'],
                        'texture_histogram': data['texture_histogram']
                    }
                return features
            finally:
                # Always clean up temp files
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
        except ValueError:
            raise  # Re-raise validation errors
        except Exception as e:
            if self.config.verbose:
                print(f"Error loading features from MinIO: {e}")
            return None

    def load_features_by_key(
        self,
        storage_key: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load features using storage key directly (alias for load_features_from_key).

        Args:
            storage_key: Full storage key path (e.g., 'features/IMG_12345.npz.gz')

        Returns:
            Dict with features or None if not found
        """
        return self.load_features_from_key(storage_key)

    def load_features_from_key(
        self,
        storage_key: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load features using storage key directly.

        Args:
            storage_key: Full storage key path

        Returns:
            Dict with features or None if not found
        """
        if not self._connected:
            return None

        try:
            # Validate storage_key to prevent path traversal
            if not storage_key or not isinstance(storage_key, str):
                raise ValueError("Invalid storage_key: must be a non-empty string")

            # Sanitize: only allow safe characters, use only basename
            # storage_key should be like "features/image_id.npz.gz"
            if '..' in storage_key or storage_key.startswith('/'):
                raise ValueError("Invalid storage_key: path traversal not allowed")

            # Extract only the filename, ignore any directory components
            filename = os.path.basename(storage_key)
            if not filename.endswith('.npz.gz'):
                raise ValueError("Invalid storage_key: must end with .npz.gz")

            # Use secure temp directory
            temp_dir = tempfile.mkdtemp(prefix="sam3_minio_")
            tmp_path = os.path.join(temp_dir, filename)

            try:
                self.client.fget_object(
                    bucket_name=self.config.bucket,
                    object_name=storage_key,
                    file_path=tmp_path
                )

                # Decompress and load
                with gzip.open(tmp_path, 'rb') as f:
                    try:
                        data = np.load(f, allow_pickle=False)
                    except ValueError as e:
                        logger.error(
                            f"Refused to load {storage_key}: file may contain pickled objects "
                            f"(allow_pickle is disabled for security). Error: {e}"
                        )
                        return None

                    features = {
                        'keypoints': data['keypoints'],
                        'descriptors': data['descriptors'],
                        'scores': data['scores'],
                    }

                    # Load texture histogram if available
                    if 'texture_histogram' in data:
                        features['texture_histogram'] = data['texture_histogram']

                if self.config.verbose:
                    print(f"Loaded features from key {storage_key}: {len(features['keypoints'])} keypoints")

                return features
            finally:
                # Always clean up temp files
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)

        except ValueError:
            raise  # Re-raise validation errors
        except Exception as e:
            if self.config.verbose:
                print(f"Error loading features from key {storage_key}: {e}")
            return None

    def delete_features(self, image_id: str) -> bool:
        """Delete features from MinIO.

        Args:
            image_id: ID of image to delete

        Returns:
            True if deletion successful
        """
        if not self._connected:
            return False

        try:
            storage_key = self._get_storage_key(image_id)
            self.client.remove_object(
                bucket_name=self.config.bucket,
                object_name=storage_key
            )
            return True
        except Exception as e:
            if self.config.verbose:
                print(f"Error deleting features from MinIO: {e}")
            return False

    def exists(self, image_id: str) -> bool:
        """Check if features exist for an image.

        Args:
            image_id: Image ID to check

        Returns:
            True if features exist
        """
        if not self._connected:
            return False

        try:
            storage_key = self._get_storage_key(image_id)
            self.client.stat_object(
                bucket_name=self.config.bucket,
                object_name=storage_key
            )
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Clean up resources."""
        self.client = None
        self._connected = False

    def get_feature_count(self) -> Dict[str, Any]:
        """Get feature count information from MinIO."""
        try:
            if not self._connected:
                self._init_client()

            # List all feature objects
            objects = self.client.list_objects(self.config.bucket, prefix=self.config.features_prefix)

            total_features = 0
            features_by_tree = {}

            for obj in objects:
                # Extract tree_id from path: features/tree_id/image_id.npz
                if obj.object_name.startswith(self.config.features_prefix):
                    path_parts = obj.object_name[len(self.config.features_prefix):].split('/')
                    if len(path_parts) >= 2:
                        tree_id = path_parts[0]
                        image_id = path_parts[1].replace('.npz', '')

                        total_features += 1
                        if tree_id not in features_by_tree:
                            features_by_tree[tree_id] = 0
                        features_by_tree[tree_id] += 1

            return {
                "total_features": total_features,
                "features_by_tree": features_by_tree,
                "bucket": self.config.bucket,
                "prefix": self.config.features_prefix
            }
        except Exception as e:
            return {
                "error": str(e),
                "total_features": 0,
                "features_by_tree": {}
            }


def create_minio_repository(
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    bucket: Optional[str] = None,
    verbose: bool = False
) -> MinIORepository:
    """Factory function to create MinIO repository.

    Credentials are read from environment variables (MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY) when not explicitly provided.

    Args:
        endpoint: MinIO server endpoint
        access_key: Access key
        secret_key: Secret key
        bucket: Bucket name
        verbose: Enable verbose output

    Returns:
        Configured MinIORepository instance
    """
    config = MinIOConfig(
        endpoint=endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=access_key or os.getenv("MINIO_ACCESS_KEY", ""),
        secret_key=secret_key or os.getenv("MINIO_SECRET_KEY", ""),
        bucket=bucket or os.getenv("MINIO_BUCKET", "tree-features"),
        verbose=verbose,
    )
    return MinIORepository(config)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="MinIO Repository Demo")
    parser.add_argument("--endpoint", default="localhost:9000")
    parser.add_argument("--bucket", default="tree-features")
    parser.add_argument("--image-id", default="test_image_001")
    args = parser.parse_args()

    # Create repository
    repo = create_minio_repository(
        endpoint=args.endpoint,
        bucket=args.bucket,
        verbose=True
    )

    # Create test features
    test_features = {
        'keypoints': np.random.rand(100, 2).astype(np.float32) * 640,
        'descriptors': np.random.rand(100, 256).astype(np.float32),
        'scores': np.random.rand(100).astype(np.float32),
    }

    # Save features
    result = repo.save_features(
        image_id=args.image_id,
        features=test_features,
        metadata={"source": "demo"}
    )

    print(f"Save result: {result.success}, key: {result.storage_key}")

    # Load features back
    loaded = repo.load_features(args.image_id)

    if loaded:
        print(f"\nLoaded features:")
        print(f"  Keypoints: {loaded['keypoints'].shape}")
        print(f"  Descriptors: {loaded['descriptors'].shape}")
        print(f"  Scores: {loaded['scores'].shape}")

    repo.close()
