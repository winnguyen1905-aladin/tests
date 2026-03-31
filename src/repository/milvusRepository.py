#!/usr/bin/env python3
"""
Milvus Repository - Vector Database for Global Features

Stores and retrieves DINO global feature vectors for coarse similarity search.
"""

import numpy as np
from pymilvus import DataType, Collection  # Collection used in get_collection_info
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import json
import logging

from src.config.appConfig import get_config, AppConfig

logger = logging.getLogger(__name__)

# =============================================================================
# Constants for Milvus index and search parameters
# =============================================================================
# IVF_FLAT index cluster count - balances search speed vs recall
# Range: [1, 65536], typical values: 4-1024 for millions of vectors
DEFAULT_NLIST: int = 128

# Number of clusters to search during query - balances speed vs recall
# Higher = more accurate but slower. Range: [1, nlist]
DEFAULT_NPROBE: int = 32

# Maximum retries for connection attempts
MAX_CONNECTION_RETRIES: int = 3

# Delay between connection retry attempts (seconds)
CONNECTION_RETRY_DELAY: float = 2.0


@dataclass
class MilvusConfig:
    """Configuration for Milvus repository (DEPRECATED: Use AppConfig instead).

    Note: This class is kept for backward compatibility only.
    All new code should use appConfig.milvus_* values directly.
    """
    uri: str = "http://localhost:19530"
    collection_name: str = "tree_features"
    vector_dim: int = 4096  # Default from appConfig (matches DINO output dimension)
    metric_type: str = "COSINE"  # COSINE, L2, IP
    index_type: str = "IVF_FLAT"  # Index type
    nlist: int = DEFAULT_NLIST  # Number of clusters (must be between 1-65536)
    nprobe: int = DEFAULT_NPROBE  # Number of probes (increased for better recall)
    timeout: float = 30.0  # Connection timeout in seconds (default 30s for slower networks)
    verbose: bool = False


@dataclass
class MilvusResult:
    """Result from Milvus search."""
    ids: List[str]  # Image IDs
    distances: List[float]  # Similarity scores (1 = perfect match for cosine)
    tree_ids: List[str]  # Tree IDs corresponding to images
    metadatas: List[Dict[str, Any]] = None  # Metadata for each result (includes minio_key)


class MilvusRepository:
    """Repository for storing and searching global feature vectors.

    Supports two connection modes:
    - Standalone: Creates its own connection (default, backward compatible)
    - Shared: Uses ConnectionManager singleton (recommended for production)
    """

    # Class-level reference to shared connection manager
    _use_shared_connection: bool = False

    def __init__(
        self,
        config: Optional[MilvusConfig] = None,
        app_config: Optional[AppConfig] = None,
        use_shared_connection: bool = False
    ) -> None:
        """Initialize Milvus repository.

        Args:
            config: Optional MilvusConfig (DEPRECATED). Uses appConfig if not provided.
            app_config: Optional AppConfig instance. Uses global config if not provided.
            use_shared_connection: If True, use ConnectionManager singleton
        """
        # Use AppConfig as primary source
        if app_config is None:
            app_config = get_config()

        self.app_config = app_config

        # Create MilvusConfig from AppConfig for backward compatibility
        if config is None:
            config = MilvusConfig(
                uri=app_config.milvus_uri,
                collection_name=app_config.milvus_collection,
                vector_dim=app_config.milvus_vector_dim,
                verbose=app_config.verbose
            )

        self.config = config
        self.client = None
        self._connected = False

        # Connection mode
        self._use_shared = use_shared_connection or MilvusRepository._use_shared_connection

        if self._use_shared:
            self._init_shared_client()
        else:
            self._init_client()

    def _init_client(self) -> None:
        """Initialize Milvus client with retry logic."""
        logger.info(f"Initializing Milvus client: URI={self.config.uri}, Collection={self.config.collection_name}")

        try:
            from pymilvus import MilvusClient, DataType
            import time

            # Retry logic - wait for Milvus to be ready
            max_retries = MAX_CONNECTION_RETRIES
            retry_delay = CONNECTION_RETRY_DELAY

            self.client = None
            last_error = None

            for attempt in range(max_retries):
                try:
                    # Create client with configurable timeout
                    self.client = MilvusClient(uri=self.config.uri, timeout=self.config.timeout)

                    # Test connection by listing collections
                    collections = self.client.list_collections()

                    # Check if collection exists, create if not
                    if self.config.collection_name not in collections:
                        self._create_collection()

                    self._connected = True
                    logger.info(f"Milvus client connected successfully: {self.config.uri}")
                    return

                except Exception as e:
                    last_error = e
                    logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Could not connect to Milvus after {max_retries} attempts: {e}")
                        self._connected = False

        except ImportError as e:
            raise ImportError(
                "pymilvus package is required. "
                "Install with: pip install pymilvus"
            ) from e

    def _init_shared_client(self) -> None:
        """Initialize Milvus client using shared ConnectionManager."""
        from src.repository.connectionManager import get_connection_manager

        logger.info("Using shared ConnectionManager for Milvus")

        manager = get_connection_manager()
        manager.configure(milvus_uri=self.config.uri)

        if manager.connect_milvus(uri=self.config.uri):
            self.client = manager.milvus_client
            self._connected = True

            # Ensure collection exists
            collections = self.client.list_collections()
            if self.config.collection_name not in collections:
                self._create_collection()

            logger.info("Milvus client connected via shared manager")
        else:
            self._connected = False
            logger.error("Failed to connect to Milvus via shared manager")

    # ─── Class Methods for Shared Connection Mode ─────────────────────────────────

    @classmethod
    def use_shared_connections(cls, enabled: bool = True) -> None:
        """Enable or disable shared connection mode for all MilvusRepository instances.

        When enabled, all instances will use the ConnectionManager singleton
        instead of creating their own connections.

        Args:
            enabled: True to use shared connections, False for standalone
        """
        cls._use_shared_connection = enabled
        logger.info(f"MilvusRepository shared connections: {enabled}")

    @classmethod
    def get_connection_manager(cls):
        """Get the shared ConnectionManager instance.

        Returns:
            The ConnectionManager singleton
        """
        from src.repository.connectionManager import get_connection_manager
        return get_connection_manager()

    def _create_collection(self) -> None:
        """Create Milvus collection with schema."""
        logger.info(f"Creating collection: {self.config.collection_name}")

        try:
            from pymilvus import MilvusClient, DataType

            # Must use schema approach for varchar primary key
            schema = self.client.create_schema()
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=64
            )
            schema.add_field(
                field_name="tree_id",
                datatype=DataType.VARCHAR,
                max_length=64
            )
            schema.add_field(
                field_name="global_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.config.vector_dim
            )
            # === NEW: Geospatial and angle fields ===
            # GPS coordinates
            schema.add_field(
                field_name="longitude",
                datatype=DataType.FLOAT,
                nullable=True
            )
            schema.add_field(
                field_name="latitude",
                datatype=DataType.FLOAT,
                nullable=True
            )
            # Viewing angles (for camera position/orientation)
            schema.add_field(
                field_name="hor_angle",
                datatype=DataType.FLOAT,
                nullable=True,
                description="Horizontal viewing angle (degrees, 0-360)"
            )
            schema.add_field(
                field_name="ver_angle",
                datatype=DataType.FLOAT,
                nullable=True,
                description="Vertical viewing angle (degrees, -90 to 90)"
            )
            schema.add_field(
                field_name="pitch",
                datatype=DataType.FLOAT,
                nullable=True,
                description="Camera pitch angle (degrees)"
            )
            # Capture timestamp (Unix epoch seconds)
            schema.add_field(
                field_name="captured_at",
                datatype=DataType.INT64,
                nullable=True,
                description="Capture timestamp (Unix epoch seconds)"
            )
            # Add metadata field (JSON type - always provide value)
            schema.add_field(
                field_name="metadata",
                datatype=DataType.JSON
            )

            # Create index for global_vector (main search field)
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="global_vector",
                index_type="IVF_FLAT",
                metric_type=self.config.metric_type,
                params={"nlist": DEFAULT_NLIST}  # Must be in range [1, 65536]
            )

            # === NEW: Add indexes for scalar fields (required for expr filtering) ===
            # GPS coordinates indexes
            index_params.add_index(
                field_name="longitude",
                index_type="STL_SORT"
            )
            index_params.add_index(
                field_name="latitude",
                index_type="STL_SORT"
            )
            # Angle indexes
            index_params.add_index(
                field_name="hor_angle",
                index_type="STL_SORT"
            )
            index_params.add_index(
                field_name="ver_angle",
                index_type="STL_SORT"
            )
            index_params.add_index(
                field_name="pitch",
                index_type="STL_SORT"
            )
            index_params.add_index(
                field_name="captured_at",
                index_type="STL_SORT"
            )

            # Without this index, filtered searches return empty results
            index_params.add_index(
                field_name="tree_id",
                index_type="INVERTED",
            )

            # Create collection with schema
            self.client.create_collection(
                collection_name=self.config.collection_name,
                schema=schema,
                index_params=index_params,
            )

            logger.info(f"Collection '{self.config.collection_name}' created with geospatial and angle fields")

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def insert(
        self,
        image_id: str,
        tree_id: str,
        global_vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        # === NEW: Geospatial and angle fields ===
        longitude: Optional[float] = None,
        latitude: Optional[float] = None,
        hor_angle: Optional[float] = None,
        ver_angle: Optional[float] = None,
        pitch: Optional[float] = None,
        captured_at: Optional[int] = None,
    ) -> bool:
        """Insert a global feature vector with geospatial, angle, and timestamp fields.

        Args:
            image_id: Unique identifier for the image
            tree_id: Tree identifier (can have multiple images per tree)
            global_vector: DINO feature vector (vector_dim,)
            metadata: Optional metadata dict
            longitude: Optional GPS longitude (-180 to 180)
            latitude: Optional GPS latitude (-90 to 90)
            hor_angle: Optional horizontal viewing angle (0-360 degrees)
            ver_angle: Optional vertical viewing angle (-90 to 90 degrees)
            pitch: Optional camera pitch angle (degrees)
            captured_at: Optional capture timestamp (Unix epoch seconds)

        Returns:
            True if insertion successful

        Raises:
            ValueError: If input validation fails
        """
        if not self._connected:
            logger.warning("Milvus not connected, skipping insert")
            return False

        try:
            # ===== INPUT VALIDATION =====

            # Validate image_id
            if not image_id or not isinstance(image_id, str):
                raise ValueError("Invalid image_id: must be a non-empty string")
            if len(image_id) > 64:
                raise ValueError("Invalid image_id: must be 64 characters or less")

            # Validate tree_id
            if not tree_id or not isinstance(tree_id, str):
                raise ValueError("Invalid tree_id: must be a non-empty string")
            if len(tree_id) > 64:
                raise ValueError("Invalid tree_id: must be 64 characters or less")

            # Validate global_vector dimension
            if global_vector is None:
                raise ValueError("Invalid global_vector: cannot be None")
            if not isinstance(global_vector, np.ndarray):
                raise ValueError("Invalid global_vector: must be a numpy array")
            if global_vector.ndim != 1:
                raise ValueError(f"Invalid global_vector: must be 1D, got {global_vector.ndim}D")
            if global_vector.shape[0] != self.config.vector_dim:
                raise ValueError(
                    f"Invalid global_vector dimension: expected {self.config.vector_dim}, "
                    f"got {global_vector.shape[0]}"
                )

            # Validate GPS coordinates if provided
            if longitude is not None:
                if not isinstance(longitude, (int, float)) or longitude < -180 or longitude > 180:
                    raise ValueError("Invalid longitude: must be between -180 and 180")

            if latitude is not None:
                if not isinstance(latitude, (int, float)) or latitude < -90 or latitude > 90:
                    raise ValueError("Invalid latitude: must be between -90 and 90")

            # Validate angles if provided
            if hor_angle is not None:
                if not isinstance(hor_angle, (int, float)) or hor_angle < 0 or hor_angle > 360:
                    raise ValueError("Invalid hor_angle: must be between 0 and 360")

            if ver_angle is not None:
                if not isinstance(ver_angle, (int, float)) or ver_angle < -90 or ver_angle > 90:
                    raise ValueError("Invalid ver_angle: must be between -90 and 90")

            if pitch is not None:
                if not isinstance(pitch, (int, float)) or pitch < -180 or pitch > 180:
                    raise ValueError("Invalid pitch: must be between -180 and 180")

            # Validate captured_at if provided
            if captured_at is not None:
                if not isinstance(captured_at, int) or captured_at < 0:
                    raise ValueError("Invalid captured_at: must be a non-negative integer (Unix timestamp)")

            # Validate metadata is a dict
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("Invalid metadata: must be a dictionary")

            # ===== BUILD DATA DICT =====
            data = {
                "id": image_id,
                "tree_id": tree_id,
                "global_vector": global_vector.tolist(),
            }

            # === NEW: Add geospatial and angle fields ===
            if longitude is not None:
                data["longitude"] = longitude
            if latitude is not None:
                data["latitude"] = latitude
            if hor_angle is not None:
                data["hor_angle"] = hor_angle
            if ver_angle is not None:
                data["ver_angle"] = ver_angle
            if pitch is not None:
                data["pitch"] = pitch
            if captured_at is not None:
                data["captured_at"] = captured_at

            # Add metadata field (use empty dict as default)
            data["metadata"] = metadata if metadata else {}

            self.client.insert(
                collection_name=self.config.collection_name,
                data=[data]
            )

            # Flush to ensure data is searchable
            self.client.flush(self.config.collection_name)

            logger.info(f"Inserted image {image_id} for tree {tree_id} with geo/angle fields")

            return True

        except ValueError:
            raise  # Re-raise validation errors
        except Exception as e:
            logger.error(f"Error inserting to Milvus: {e}")
            return False

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 50,
        tree_id_filter: Optional[str] = None
    ) -> MilvusResult:
        """Search for similar vectors with optional tree_id filtering.

        Args:
            query_vector: DINO feature vector to search
            top_k: Number of results to return
            tree_id_filter: Optional filter for specific tree ID
                - If None: Search across all trees (for /search endpoint)
                - If provided: Only search within that tree (for /verify endpoint)

        Returns:
            MilvusResult with matched IDs and distances
        """
        if not self._connected:
            raise ValueError("Milvus not connected")

        try:
            # CRITICAL: Normalize query vector for COSINE metric
            # normalized_query = query_vector / (np.linalg.norm(query_vector) + 1e-8)

            # Build search parameters for MilvusClient
            search_params = {
                "collection_name": self.config.collection_name,
                "data": [query_vector.tolist()],
                "limit": top_k,
                "output_fields": ["id", "tree_id", "metadata"],
                "anns_field": "global_vector",  # FIXED: Specify which vector field to search
                "search_params": {"metric_type": self.config.metric_type, "nprobe": self.config.nprobe}
            }

            # Add filter only if tree_id_filter is provided and not empty
            # This allows:
            # 1. tree_id_filter=None -> Search all trees (global search)
            # 2. tree_id_filter="TREE_123" -> Search only within TREE_123
            if tree_id_filter and tree_id_filter.strip():
                search_params["filter"] = f'tree_id == "{tree_id_filter}"'
                if self.config.verbose:
                    logger.info(f"[Milvus] Searching with tree_id filter: {tree_id_filter}")

            # Don't explicitly load collection - MilvusClient handles this
            # The error "collection not loaded" happens when using old Connection API
            if self.config.verbose:
                logger.info(f"[Milvus] Executing search with anns_field={search_params.get('anns_field')}, filter={search_params.get('filter', 'None')}")

            result = self.client.search(**search_params)

            if result and len(result) > 0:
                hits = result[0]
                ids = []
                distances = []
                tree_ids = []
                metadatas = []

                for hit in hits:
                    hit_id = hit.get('id')
                    hit_distance = hit.get('distance')

                    # tree_id is in entity for MilvusClient (v2.4.15+)
                    if hit.get('entity'):
                        hit_tree_id = hit['entity'].get('tree_id', '')
                        hit_metadata = hit['entity'].get('metadata', {})
                    else:
                        # Fallback for older API
                        hit_tree_id = hit.get('tree_id', '')
                        hit_metadata = hit.get('metadata', {})

                    if hit_id is not None:  # Only add valid results
                        ids.append(hit_id)
                        distances.append(hit_distance)
                        tree_ids.append(hit_tree_id)
                        metadatas.append(hit_metadata)

                if self.config.verbose:
                    logger.info(f"[Milvus] Search returned {len(ids)}/{len(hits)} valid results")

                return MilvusResult(ids=ids, distances=distances, tree_ids=tree_ids, metadatas=metadatas)

            if self.config.verbose:
                if tree_id_filter:
                    logger.info(f"[Milvus] No hits found for tree_id={tree_id_filter}")
                else:
                    logger.info(f"[Milvus] No hits found in global search")
            return MilvusResult(ids=[], distances=[], tree_ids=[], metadatas=[])

        except Exception as e:
            error_msg = str(e).lower()

            # Handle "collection not loaded" error by attempting to load it
            if "collection not loaded" in error_msg:
                logger.warning(f"[Milvus] Collection not loaded, attempting to load...")
                try:
                    # Try to load the collection before searching again
                    self.client.load_collection(self.config.collection_name)
                    logger.info(f"[Milvus] Collection loaded, retrying search...")

                    # Retry the search
                    result = self.client.search(**search_params)

                    if result and len(result) > 0:
                        hits = result[0]
                        ids = []
                        distances = []
                        tree_ids = []
                        metadatas = []

                        logger.info(f"[Milvus] Found {len(hits)} hits after retry")

                        for hit in hits:
                            hit_id = hit.get('id')
                            hit_distance = hit.get('distance')
                            hit_tree_id = hit.get('tree_id') or (hit.get('entity', {}).get('tree_id') if hit.get('entity') else '')
                            hit_metadata = hit.get('metadata') or (hit.get('entity', {}).get('metadata') if hit.get('entity') else {})

                            ids.append(hit_id)
                            distances.append(hit_distance)
                            tree_ids.append(hit_tree_id)
                            metadatas.append(hit_metadata)

                            logger.debug(f"  Hit: id={hit_id}, distance={hit_distance:.4f}, tree_id={hit_tree_id}")

                        if tree_id_filter:
                            logger.info(f"[Milvus] ✓ Retry successful: {len(ids)} results for tree_id={tree_id_filter}")
                        else:
                            logger.info(f"[Milvus] ✓ Retry successful: {len(ids)} results across all trees")

                        return MilvusResult(ids=ids, distances=distances, tree_ids=tree_ids, metadatas=metadatas)

                    if tree_id_filter:
                        logger.info(f"[Milvus] No hits found for tree_id={tree_id_filter} (after loading)")
                    else:
                        logger.info(f"[Milvus] No hits found in global search (after loading)")
                    return MilvusResult(ids=[], distances=[], tree_ids=[], metadatas=[])

                except Exception as retry_e:
                    logger.error(f"[Milvus] Failed to load collection and retry: {retry_e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise retry_e

            # Other errors
            filter_info = f" with tree_id={tree_id_filter}" if tree_id_filter else " (global search)"
            logger.error(f"[Milvus] Error searching{filter_info}: {e}")
            raise e

    def search_with_bounding_box(
        self,
        query_vector: np.ndarray,
        top_k: int = 50,
        # GPS bounding box (longitude, latitude)
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        # Angle bounding box (horizontal, vertical angles)
        hor_angle_min: Optional[float] = None,
        hor_angle_max: Optional[float] = None,
        ver_angle_min: Optional[float] = None,
        ver_angle_max: Optional[float] = None,
        pitch_min: Optional[float] = None,
        pitch_max: Optional[float] = None,
        # Timestamp filter (Unix epoch seconds)
        captured_at_min: Optional[int] = None,
        captured_at_max: Optional[int] = None,
        # Tree ID filter
        tree_id_filter: Optional[str] = None,
    ) -> MilvusResult:
        """Search with bounding box filtering using Milvus internal filter expressions.

        This method uses Milvus filter expressions to filter by scalar fields
        (geo coordinates and angles) directly in the database for better performance.

        Args:
            query_vector: DINO feature vector to search
            top_k: Number of results to return
            lon_min, lon_max: Longitude range (-180 to 180)
            lat_min, lat_max: Latitude range (-90 to 90)
            hor_angle_min, hor_angle_max: Horizontal angle range (0-360)
            ver_angle_min, ver_angle_max: Vertical angle range (-90 to 90)
            pitch_min, pitch_max: Pitch angle range
            tree_id_filter: Optional filter for specific tree ID

        Returns:
            MilvusResult with matched IDs, distances, and metadata
        """
        if not self._connected:
            raise ValueError("Milvus not connected")

        try:
            # Build filter expression for scalar fields
            filter_parts = []

            # GPS bounding box filters
            if lon_min is not None:
                filter_parts.append(f"longitude >= {lon_min}")
            if lon_max is not None:
                filter_parts.append(f"longitude <= {lon_max}")
            if lat_min is not None:
                filter_parts.append(f"latitude >= {lat_min}")
            if lat_max is not None:
                filter_parts.append(f"latitude <= {lat_max}")

            # Angle bounding box filters
            if hor_angle_min is not None:
                filter_parts.append(f"hor_angle >= {hor_angle_min}")
            if hor_angle_max is not None:
                filter_parts.append(f"hor_angle <= {hor_angle_max}")
            if ver_angle_min is not None:
                filter_parts.append(f"ver_angle >= {ver_angle_min}")
            if ver_angle_max is not None:
                filter_parts.append(f"ver_angle <= {ver_angle_max}")
            if pitch_min is not None:
                filter_parts.append(f"pitch >= {pitch_min}")
            if pitch_max is not None:
                filter_parts.append(f"pitch <= {pitch_max}")

            # Timestamp filters
            if captured_at_min is not None:
                filter_parts.append(f"captured_at >= {captured_at_min}")
            if captured_at_max is not None:
                filter_parts.append(f"captured_at <= {captured_at_max}")

            # Tree ID filter
            if tree_id_filter and tree_id_filter.strip():
                filter_parts.append(f'tree_id == "{tree_id_filter}"')

            # Combine filter expressions
            filter_expr = " and ".join(filter_parts) if filter_parts else None

            # Build output fields
            output_fields = ["id", "tree_id", "metadata"]
            # Include geo/angle fields for debugging/verification
            if lon_min is not None or lon_max is not None or lat_min is not None or lat_max is not None:
                output_fields.extend(["longitude", "latitude"])
            if hor_angle_min is not None or hor_angle_max is not None or ver_angle_min is not None or ver_angle_max is not None or pitch_min is not None or pitch_max is not None:
                output_fields.extend(["hor_angle", "ver_angle", "pitch"])
            # Include timestamp field for time filtering and display
            if captured_at_min is not None or captured_at_max is not None:
                output_fields.append("captured_at")

            # Use larger limit to account for potential filtered-out results
            search_limit = top_k * 3 if filter_expr else top_k

            search_params = {
                "collection_name": self.config.collection_name,
                "data": [query_vector.tolist()],
                "limit": search_limit,
                "output_fields": output_fields,
                "anns_field": "global_vector",
                "search_params": {"metric_type": self.config.metric_type, "nprobe": self.config.nprobe}
            }

            # Add filter expression if we have any filters
            if filter_expr:
                search_params["filter"] = filter_expr
                logger.info(f"[Milvus] Searching with filter: {filter_expr}")

            result = self.client.search(**search_params)

            if not result or len(result) == 0:
                logger.info("[Milvus] No results from search")
                return MilvusResult(ids=[], distances=[], tree_ids=[], metadatas=[])

            hits = result[0]

            # Build result
            ids = []
            distances = []
            tree_ids = []
            metadatas = []

            for hit in hits[:top_k]:
                hit_id = hit.get('id')
                hit_distance = hit.get('distance')

                entity = hit.get('entity', {})
                hit_tree_id = entity.get('tree_id', '')
                hit_metadata = entity.get('metadata', {})

                if hit_id is not None:
                    ids.append(hit_id)
                    distances.append(hit_distance)
                    tree_ids.append(hit_tree_id)
                    metadatas.append(hit_metadata)

            logger.info(f"[Milvus] Bounding box filter: {len(ids)} results returned")
            return MilvusResult(ids=ids, distances=distances, tree_ids=tree_ids, metadatas=metadatas)

        except Exception as e:
            logger.error(f"[Milvus] Error in bounding box search: {e}")
            raise

    def coarse_retrieval(
        self,
        query_vector: np.ndarray,
        top_k: int,
        threshold: float = 0.0,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K candidates from Milvus with threshold filtering.

        This method encapsulates the coarse retrieval logic that was previously
        in verificationService._coarse_retrieval().

        Args:
            query_vector: Query global feature vector
            top_k: Number of candidates to retrieve
            threshold: Minimum similarity threshold (default: 0.0)
            verbose: Enable verbose logging

        Returns:
            List of candidate dictionaries with keys:
                - image_id: Candidate image ID
                - tree_id: Tree ID
                - similarity_score: Cosine similarity
                - rank: Rank in results
                - metadata: Additional metadata (if available)
        """
        result = self.search(
            query_vector=query_vector,
            top_k=top_k,
        )

        candidates = []

        if verbose:
            logger.info(f"[Coarse Retrieval] Got {len(result.ids)} hits, threshold={threshold}")
            logger.info(f"[Coarse Retrieval] Metric type: {self.config.metric_type}")

        # Zip with metadatas if available
        metadatas = result.metadatas if result.metadatas else [{}] * len(result.ids)

        for rank, (image_id, distance, tree_id, metadata) in enumerate(zip(result.ids, result.distances, result.tree_ids, metadatas)):
            # Milvus returns cosine similarity directly (already normalized to [0, 1])
            # where 1.0 = perfect match, 0 = no match
            similarity = distance

            if verbose and rank < 5:
                logger.info(f"[Coarse Retrieval] Hit {rank}: id={image_id}, similarity={similarity:.4f}, threshold={threshold}")

            if similarity >= threshold:
                candidate = {
                    'image_id': image_id,
                    'tree_id': tree_id,
                    'similarity_score': similarity,
                    'rank': rank,
                    'metadata': metadata
                }
                candidates.append(candidate)

        if verbose:
            logger.info(f"[Coarse Retrieval] Filtered to {len(candidates)} candidates above threshold")

        return candidates


    def delete(self, image_id: str) -> bool:
        """Delete an image from the database.

        Args:
            image_id: ID of image to delete

        Returns:
            True if deletion successful
        """
        if not self._connected:
            return False

        try:
            self.client.delete(
                collection_name=self.config.collection_name,
                ids=[image_id]
            )
            self.client.flush(self.config.collection_name)
            logger.info(f"Deleted image {image_id} from Milvus")
            return True
        except Exception as e:
            logger.error(f"Error deleting from Milvus: {e}")
            return False

    def get_vector(self, image_id: str) -> Optional[np.ndarray]:
        """Retrieve a vector by image ID.

        Args:
            image_id: ID of image to retrieve

        Returns:
            Feature vector or None if not found
        """
        if not self._connected:
            return None

        try:
            # Load collection before querying (required in Milvus 2.x)
            self.client.load_collection(self.config.collection_name)

            result = self.client.get(
                collection_name=self.config.collection_name,
                ids=[image_id],
                output_fields=["global_vector"]
            )

            if result and len(result) > 0:
                return np.array(result[0]['global_vector'], dtype=np.float32)
            return None
        except Exception as e:
            logger.error(f"Error getting vector from Milvus: {e}")
            return None

    def get_entity(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a complete entity (vector + metadata) by image ID.

        Args:
            image_id: ID of image to retrieve

        Returns:
            Dictionary with 'vector' and 'metadata' keys, or None if not found
        """
        if not self._connected:
            return None

        try:
            # Load collection before querying (required in Milvus 2.x)
            self.client.load_collection(self.config.collection_name)

            result = self.client.get(
                collection_name=self.config.collection_name,
                ids=[image_id],
                output_fields=["global_vector", "tree_id", "metadata"]
            )

            if result and len(result) > 0:
                entity = result[0]
                # Include tree_id in metadata for easier access
                metadata = entity.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {}
                if 'tree_id' not in metadata and entity.get('tree_id'):
                    metadata['tree_id'] = entity.get('tree_id')

                return {
                    'vector': np.array(entity.get('global_vector', []), dtype=np.float32),
                    'metadata': metadata
                }
            return None
        except Exception as e:
            logger.error(f"Error getting entity from Milvus: {e}")
            return None

    def close(self) -> None:
        """Clean up resources."""
        if self.client is not None:
            self.client.close()
            self._connected = False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            if not self._connected:
                self._init_client()

            # Get collection statistics
            collection = Collection(self.config.collection_name)
            collection.load()

            stats = {
                "entity_count": collection.num_entities,
                "collection_name": self.config.collection_name,
                "dimension": self.config.vector_dim,
                "metric_type": self.config.metric_type
            }

            collection.release()
            return stats
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            # Return safe default values
            return {
                "entity_count": 0,
                "collection_name": self.config.collection_name,
                "dimension": self.config.vector_dim,
                "metric_type": self.config.metric_type,
                "error": str(e)
            }

    def get_tree_images(self, tree_id: str) -> List[Dict[str, Any]]:
        """
        Get all images for a specific tree.

        Args:
            tree_id: Tree identifier

        Returns:
            List of dicts with image_id, global_vector, position_3d, metadata
        """
        if not self._connected:
            logger.warning("Milvus not connected")
            return []

        try:
            # Query by tree_id
            results = self.client.query(
                collection_name=self.config.collection_name,
                filter=f'tree_id == "{tree_id}"',
                output_fields=["id", "tree_id", "global_vector", "position_3d", "metadata"]
            )

            logger.info(f"Retrieved {len(results)} images for tree {tree_id}")

            return results

        except Exception as e:
            logger.error(f"Error querying tree images: {e}")
            return []

    def get_all_trees_metadata(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get metadata for all trees from the database.

        Returns:
            Dict mapping tree_id -> list of metadata dicts with:
                latitude, longitude, hor_angle, ver_angle, pitch, captured_at, image_id
        """
        if not self._connected:
            logger.warning("Milvus not connected")
            return {}

        try:
            # Query all entities with metadata fields
            results = self.client.query(
                collection_name=self.config.collection_name,
                filter="id != ''",
                output_fields=[
                    "id", "tree_id", "latitude", "longitude",
                    "hor_angle", "ver_angle", "pitch", "captured_at"
                ]
            )

            # Group by tree_id
            trees_metadata: Dict[str, List[Dict[str, Any]]] = {}
            for entity in results:
                tree_id = entity.get("tree_id", "")
                if not tree_id:
                    continue

                if tree_id not in trees_metadata:
                    trees_metadata[tree_id] = []

                # Extract metadata fields
                metadata = {
                    "image_id": entity.get("id", ""),
                    "latitude": entity.get("latitude"),
                    "longitude": entity.get("longitude"),
                    "hor_angle": entity.get("hor_angle"),
                    "ver_angle": entity.get("ver_angle"),
                    "pitch": entity.get("pitch"),
                    "captured_at": entity.get("captured_at"),
                }
                trees_metadata[tree_id].append(metadata)

            logger.info(f"Retrieved metadata for {len(trees_metadata)} trees")
            return trees_metadata

        except Exception as e:
            logger.error(f"Error getting all trees metadata: {e}")
            return {}


def create_milvus_repository(
    uri: Optional[str] = None,
    collection_name: Optional[str] = None,
    vector_dim: Optional[int] = None,
    verbose: Optional[bool] = None,
    app_config: Optional[AppConfig] = None
) -> MilvusRepository:
    """Factory function to create Milvus repository from AppConfig.

    Args:
        uri: Optional override for Milvus URI (uses app_config if not provided)
        collection_name: Optional override for collection name
        vector_dim: Optional override for vector dimension
        verbose: Optional override for verbose flag
        app_config: Optional AppConfig instance. Uses global config if not provided.

    Returns:
        Configured MilvusRepository instance
    """
    if app_config is None:
        app_config = get_config()

    # Create config from AppConfig with optional overrides
    config = MilvusConfig(
        uri=uri or app_config.milvus_uri,
        collection_name=collection_name or app_config.milvus_collection,
        vector_dim=vector_dim or app_config.milvus_vector_dim,
        verbose=verbose if verbose is not None else app_config.verbose
    )
    return MilvusRepository(config=config, app_config=app_config)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Milvus Repository Demo")
    parser.add_argument("--uri", default="http://localhost:19530")
    parser.add_argument("--collection", default="tree_features")
    args = parser.parse_args()

    # Create repository
    repo = create_milvus_repository(uri=args.uri, collection_name=args.collection, verbose=True)

    # Insert test vector (use appConfig dimension)
    from src.config.appConfig import get_config
    app_config = get_config()
    test_vector = np.random.randn(app_config.milvus_vector_dim).astype(np.float32)
    test_vector = test_vector / np.linalg.norm(test_vector)  # Normalize

    success = repo.insert(
        image_id="test_image_001",
        tree_id="test_tree_001",
        global_vector=test_vector,
        metadata={"camera": "iphone14", "angle": "front"}
    )

    print(f"Insert successful: {success}")

    # Search
    result = repo.search(test_vector, top_k=5)

    print(f"\n=== Search Results ===")
    print(f"Found {len(result.ids)} results")
    for i, (id_, dist, tree_id) in enumerate(zip(result.ids, result.distances, result.tree_ids)):
        print(f"  {i+1}. {id_} (tree: {tree_id}): distance = {dist:.4f}")

    repo.close()

