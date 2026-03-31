"""
Ingestion Service - Orchestrates the ingestion pipeline

This service coordinates:
1. Preprocessing (via preprocessor)
2. Feature extraction (via processors)
3. Storage (via repositories)

Uses dependency injection with @inject decorator.
Wire with: container.wire(modules=["src.service.ingestionService"])
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import cv2
import numpy as np

from dependency_injector.wiring import inject, Provide

from src.config.appConfig import AppConfig, get_config

logger = logging.getLogger(__name__)

from src.processor.dinoProcessor import DinoConfig, DinoProcessor, DinoResult
from src.processor.superPointProcessor import SuperPointProcessor, SuperPointResult, SuperPointConfig
from src.repository.sqlalchemyRepository import SQLAlchemyORMRepository, TreeRecord, _resolve_evidence_uuid
from src.repository.minioRepository import MinIORepository, MinIOConfig
from src.repository.databaseManager import get_db_session
from src.service.preprocessorService import PreprocessorService


@dataclass
class IngestionResult:
    """Result of ingestion operation."""
    success: bool
    image_id: str
    tree_id: str
    storage_keys: Dict[str, str]
    feature_info: Dict[str, Any]
    message: str


class IngestionService:
    """Service for ingesting tree images and extracting features.
    
    Uses @inject decorator for dependency injection with dependency_injector.
    Wire with: container.wire(modules=["src.service.ingestionService"])
    """

    @inject
    def __init__(
        self,
        preprocessor: PreprocessorService = Provide["preprocessor_service"],
        app_config: AppConfig = Provide["app_config"],
        dino_processor: DinoProcessor = Provide["dino_processor"],
        superpoint_processor: SuperPointProcessor = Provide["superpoint_processor"],
        postgres_repo: SQLAlchemyORMRepository = Provide["sqlalchemy_repo"],
        minio_repo: MinIORepository = Provide["minio_repo"],
        verbose: bool = False,
    ) -> None:
        """Initialize ingestion service with injected dependencies.

        Args:
            preprocessor: Image preprocessing module (injected via Provide)
            app_config: Application configuration (injected via Provide)
            dino_processor: DinoV2/V3 feature extractor (injected via Provide)
            superpoint_processor: SuperPoint feature extractor (injected via Provide)
            postgres_repo: PostgreSQL repository with pgvector (injected via Provide)
            minio_repo: MinIO feature store repository (injected via Provide)
            verbose: Enable verbose logging
        """
        self.preprocessor: PreprocessorService = preprocessor
        self.app_config: AppConfig = app_config
        self.dino_processor: DinoProcessor = dino_processor
        self.superpoint_processor: SuperPointProcessor = superpoint_processor
        self.postgres_repo: SQLAlchemyORMRepository = postgres_repo
        self.minio_repo: MinIORepository = minio_repo
        self.verbose: bool = verbose
        logger.info(f"IngestionService initialized with PostgreSQL for vector storage")

    @staticmethod
    def _parse_captured_at(value) -> Optional[int]:
        """Parse captured_at value to Unix epoch seconds (int).

        Accepts:
            - int/float: treated as epoch seconds directly, unless the value
              is >= 10^12 (millisecond precision), in which case it is divided
              by 1000 to convert to seconds.
            - str: parsed as ISO 8601 datetime string

        Returns:
            Unix epoch seconds as int, or None if value is None or unparseable.
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            seconds = int(value)
            # Auto-detect millisecond timestamps: any value >= 10^12 represents
            # a date after year 2001 (in UTC).  Dividing by 1000 yields a
            # reasonable Unix-epoch second value (e.g. 1772263670 → year 2026).
            if seconds >= 10**12:
                seconds = seconds // 1000
            return seconds
        if isinstance(value, str):
            try:
                from datetime import datetime
                # Try ISO 8601 parsing (Python 3.7+)
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                return int(dt.timestamp())
            except (ValueError, TypeError):
                logger.warning(f"Could not parse captured_at value: {value}")
                return None
        return None

    def ingest(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        image_id: str,
        tree_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionResult:
        """Ingest a tree image and extract features.

        Args:
            image: Input image (numpy array)
            mask: Segmentation mask
            image_id: Unique image identifier
            tree_id: Tree identifier
            metadata: Optional metadata

        Returns:
            IngestionResult with storage info
        """
        try:
            self._validate_inputs(image, mask, image_id, tree_id)

            # Step 1: Preprocessing
            logger.info(f"[STEP 1] Preprocessing...")
            try:
                # Convert mask to grayscale if needed
                if len(mask.shape) == 3:
                    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                else:
                    mask_gray = mask

                masked_image: np.ndarray = self.preprocessor.apply_mask(image, mask_gray)
                cropped_image: np.ndarray
                cropped_mask: np.ndarray
                bbox: Tuple[int, int, int, int]
                bbox = self.preprocessor.get_bounding_box(mask_gray)

                cropped_image, cropped_mask, _ = self.preprocessor.crop_to_bounding_box(
                    masked_image, bbox, mask=mask_gray
                ) 
                logger.info(f"[STEP 1] ✓ Preprocessing complete")
            except Exception as e:
                logger.error(f"[STEP 1] ✗ Preprocessing failed: {e}")
                raise

            # Step 2: Extract global features (DinoV2/V3) with CUDA stream optimization
            logger.info(f"[STEP 2] Extracting global features (DinoV2/V3)...")

            global_features = None

            if self.dino_processor:
                try:
                    dino_input: np.ndarray = self.preprocessor.prepare_for_dino(cropped_image)
                    logger.info(f"[STEP 2] Extracting global features (DinoV2/V3)...")

                    global_features: Optional[DinoResult] = self.dino_processor.extract(dino_input)
                    self._validate_dino_result(global_features, get_config().postgres_vector_dim)
                except Exception as e:
                    logger.error(f"[STEP 2] ✗ Global feature extraction failed: {e}")
                    raise ValueError(f"Failed to extract DINO features: {e}")

            if not global_features:
                raise ValueError(
                    "DINO feature extraction failed — global_features is None. "
                    "dino_processor must be configured and must return a valid vector."
                )

            if global_features:
                logger.info(f"[STEP 2] ✓ Global features ready: dim={len(global_features.vector)}")

            # Step 3: Extract local features (SuperPoint) with CUDA stream optimization
            logger.info(f"[STEP 3] Extracting local features (SuperPoint)...")

            local_features = None
            if self.superpoint_processor:
                try:
                    sp_input: np.ndarray
                    sp_input, _ = self.preprocessor.prepare_for_superpoint(cropped_image)

                    sp_gray: np.ndarray = self.preprocessor.to_grayscale(sp_input)

                    # IMPORTANT: Pass the cropped_mask to SuperPoint for uniform distribution
                    # Force CPU for SuperPoint if CUDA fails
                    import torch
                    if hasattr(self.superpoint_processor, 'config') and self.superpoint_processor.config.device == "cuda":
                        if not torch.cuda.is_available():
                            logger.warning("CUDA not available, forcing CPU for SuperPoint")
                            self.superpoint_processor.config.device = "cpu"
                            self.superpoint_processor.__init__(self.superpoint_processor.config)
                        try:
                            local_features: Optional[SuperPointResult] = self.superpoint_processor.extract(sp_gray, mask=cropped_mask)
                        except RuntimeError as e:
                            if "CUDA error" in str(e):
                                logger.warning("CUDA error with SuperPoint, falling back to CPU")
                                self.superpoint_processor.config.device = "cpu"
                                self.superpoint_processor.__init__(self.superpoint_processor.config)
                                local_features = self.superpoint_processor.extract(sp_gray, mask=cropped_mask)
                            else:
                                raise e
                    else:
                        local_features = self.superpoint_processor.extract(sp_gray, mask=cropped_mask)

                    self._validate_superpoint_result(local_features)
                    logger.info(f"  SuperPoint features extracted: keypoints={len(local_features.keypoints)}, descriptors_shape={local_features.descriptors.shape}")
                    logger.info(f"  Keypoint scores - min: {local_features.scores.min():.4f}, max: {local_features.scores.max():.4f}, mean: {local_features.scores.mean():.4f}")
                    logger.info(f"[STEP 3] ✓ Local features ready: {len(local_features.keypoints)} keypoints")
                except Exception as e:
                    logger.error(f"[STEP 3] ✗ Local feature extraction failed: {e}")
                    raise e
            
            # Compute the evidence UUID once — shared by MinIO key and PostgreSQL PK.
            # Using a stable UUID5 for slug-style image_ids (e.g. "tree-001") means
            # re-ingesting the same image overwrites the same MinIO object and PG row.
            evidence_uuid = _resolve_evidence_uuid(image_id)
            evidence_id_str = str(evidence_uuid)

            # Step 4: Store local features in MinIO first (to get minio_key)
            logger.info(f"[STEP 4] Storing local features in MinIO...")

            if self.minio_repo and local_features:
                try:
                    self._validate_local_features_for_storage(local_features)

                    # Build features dict
                    features_to_save = {
                        'keypoints': local_features.keypoints,
                        'descriptors': local_features.descriptors,
                        'scores': local_features.scores,
                    }

                    # Use the evidence UUID as the MinIO object name so that
                    # load_features(evidence_id) always finds the right file.
                    minio_result = self.minio_repo.save_features(
                        image_id=evidence_id_str,
                        features=features_to_save,
                        metadata={
                            'tree_id': tree_id,
                            'image_id': image_id,
                            'image_size': local_features.image_size
                        }
                    )

                    # Validate storage result
                    if not minio_result:
                        raise ValueError("MinIO save returned None")
                    if hasattr(minio_result, 'success') and not minio_result.success:
                        raise ValueError(f"MinIO save failed: {getattr(minio_result, 'message', '')}")

                    # Extract storage_key from MinIOResult object
                    minio_key = minio_result.storage_key if hasattr(minio_result, 'storage_key') else str(minio_result)

                    logger.info(f"[STEP 4] ✓ Local features stored in MinIO: {minio_key}")
                except Exception as e:
                    logger.error(f"[STEP 4] ✗ MinIO storage failed: {e}")
                    raise
            else:
                minio_key = "minio_not_configured"
                logger.warning(f"[STEP 4] ⚠️  MinIO not configured or no local features")

            # ============================================================================
            # STEP 5: Store global features in PostgreSQL
            # ============================================================================
            # Store global DINO features in PostgreSQL with pgvector
            # ============================================================================
            logger.info(f"[STEP 5] Storing global features in PostgreSQL...")

            # Extract geospatial and angle fields from metadata
            geo_metadata = metadata or {}
            longitude = geo_metadata.get("longitude")
            latitude = geo_metadata.get("latitude")
            hor_angle = geo_metadata.get("hor_angle")
            ver_angle = geo_metadata.get("ver_angle")
            pitch = geo_metadata.get("pitch")
            captured_at = self._parse_captured_at(geo_metadata.get("captured_at"))

            vector_key = None
            postgres_key = None

            # ============================================================================
            # Data Transformation: Convert numpy vector to list for PostgreSQL
            # ============================================================================
            # Data Transformation: Convert numpy vector to list for PostgreSQL
            # ============================================================================
            # PostgreSQL/pgvector requires list format, not numpy array
            # ============================================================================
            vector_list = []
            if global_features and hasattr(global_features, 'vector') and global_features.vector is not None:
                vector_list = global_features.vector.tolist() if hasattr(global_features.vector, 'tolist') else list[Any](global_features.vector)

            def _persist_postgres(repo: SQLAlchemyORMRepository) -> Optional[str]:
                """Persist tree + evidence via the provided repository instance."""
                tree: Optional[TreeRecord] = repo.get_tree(tree_id)
                if not tree:
                    # Create the tree if it doesn't exist - get GPS from metadata
                    repo.create_tree(
                        tree_id=tree_id,
                        region_code=geo_metadata.get("region_code", "default"),
                        farm_id=geo_metadata.get("farm_id", "default_farm"),
                        geohash_7=geo_metadata.get("geohash_7", "default"),
                        row_idx=geo_metadata.get("row_idx") if geo_metadata else None,
                        col_idx=geo_metadata.get("col_idx") if geo_metadata else None,
                        longitude=longitude,
                        latitude=latitude,
                    )

                evidence_metadata = dict[str, Any](geo_metadata) if geo_metadata else {}
                evidence_metadata["image_id"] = image_id
                evidence_metadata["minio_key"] = minio_key

                return repo.create_evidence(
                    tree_id=tree_id,
                    region_code=geo_metadata.get("region_code", "default"),
                    global_vector=vector_list,
                    storage_cid="",
                    # Pass pre-computed UUID string — no slug ambiguity in create_evidence
                    evidence_id=evidence_id_str,
                    evidence_hash=geo_metadata.get("evidence_hash"),
                    camera_heading=int(hor_angle) if hor_angle is not None else None,
                    camera_pitch=int(pitch) if pitch is not None else None,
                    camera_roll=int(ver_angle) if ver_angle is not None else None,
                    is_c2pa_verified=geo_metadata.get("is_c2pa_verified", False),
                    raw_telemetry={
                        "model": global_features.model_name,
                        "bbox": bbox,
                        "image_size": global_features.image_size,
                        "ver_angle": ver_angle,
                    },
                    metadata=evidence_metadata,
                    latitude=latitude,
                    longitude=longitude,
                    captured_at=captured_at,
                )

            # === Store to PostgreSQL (atomic — tree + evidence must commit together) ===
            # Use a fresh per-request repo to avoid mutating the shared singleton instance.
            if global_features and self.postgres_repo:
                try:
                    with get_db_session() as session:
                        # Fresh repo instance per request — prevents concurrent session overwrite
                        repo = SQLAlchemyORMRepository(session)
                        postgres_key = _persist_postgres(repo)
                        # get_db_session commits on normal exit, rolls back on exception
                except Exception as primary_err:
                    # Fallback for tests / mocked repositories where global DB manager
                    # is intentionally not initialized.
                    logger.warning(
                        "[STEP 5] DB session path failed, retrying with injected postgres_repo: %s",
                        primary_err,
                    )
                    try:
                        postgres_key = _persist_postgres(self.postgres_repo)
                    except Exception as e:
                        logger.error(f"[STEP 5] ✗ PostgreSQL storage failed: {e}")
                        raise ValueError(f"PostgreSQL storage failed: {e}") from e

                if postgres_key:
                    logger.info(f"[STEP 5] ✓ Global features stored in PostgreSQL (evidence_id: {postgres_key})")
                else:
                    raise ValueError("PostgreSQL create_evidence returned None")
            elif global_features and not self.postgres_repo:
                raise ValueError(
                    "PostgreSQL repository is not configured; "
                    "cannot persist global vector for ingestion."
                )

            # === Set vector_key to PostgreSQL evidence_id ===
            if postgres_key:
                vector_key = str(postgres_key)

            logger.info(f"[STEP 6] Preparing ingestion result...")
            logger.info(f"  Global features dim: {len(global_features.vector) if global_features else 0}")
            logger.info(f"  Local keypoints: {len(local_features.keypoints) if local_features else 0}")
            logger.info(f"  Descriptor dim: {local_features.descriptors.shape[1] if local_features and len(local_features.descriptors.shape) > 1 else 0}")

            # Build storage keys for PostgreSQL only
            storage_keys = {
                "minio_key": minio_key,
            }
            if postgres_key:
                storage_keys["postgres_key"] = str(postgres_key)
            if vector_key:
                storage_keys["vector_key"] = str(vector_key)

            result = IngestionResult(
                success=True,
                image_id=image_id,
                tree_id=tree_id,
                storage_keys=storage_keys,
                feature_info={
                    "global_dim": len(global_features.vector) if global_features else 0,
                    "n_keypoints": len(local_features.keypoints) if local_features else 0,
                    "descriptor_dim": local_features.descriptors.shape[1] if local_features and len(local_features.descriptors.shape) > 1 else 0,
                },
                message=f"Successfully ingested {image_id}",
            )

            logger.info(f"{'='*80}")
            logger.info(f"INGEST COMPLETE: {image_id} -> {tree_id}")
            if postgres_key:
                logger.info(f"  PostgreSQL: {postgres_key}")
            logger.info(f"  MinIO: {minio_key}")
            logger.info(f"{'='*80}")

            return result
        
        except Exception as e:
            logger.error(f"{'='*80}")
            logger.error(f"INGEST FAILED: {image_id}")
            logger.error(f"  Error: {e}")
            logger.error(f"  Error type: {type(e).__name__}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
            logger.error(f"{'='*80}")

            return IngestionResult(
                success=False,
                image_id=image_id,
                tree_id=tree_id,
                storage_keys={},
                feature_info={},
                message=f"Ingestion failed: {str(e)}",
            )

    def ingest_batch(
        self,
        items: list,  # List of dicts: {image, mask, image_id, tree_id, metadata?}
        batch_size: int = 8,
    ) -> list:
        """Ingest multiple images using true batch processing for DINO and SuperPoint.

        Args:
            items: List of dicts with keys: image, mask, image_id, tree_id, metadata (optional)
            batch_size: GPU batch size for DINO and SuperPoint (default: 8 for RTX 5060 Ti)

        Returns:
            List of IngestionResult, one per item
        """
        if not items:
            return []

        results = [None] * len(items)

        # ── Step 1: Preprocess all images ──────────────────────────────────────
        preprocessed = []
        for i, item in enumerate(items):
            try:
                image = item['image']
                mask = item['mask']
                image_id = item['image_id']
                tree_id = item['tree_id']

                if len(mask.shape) == 3:
                    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                else:
                    mask_gray = mask

                masked = self.preprocessor.apply_mask(image, mask_gray)
                bbox = self.preprocessor.get_bounding_box(mask_gray)
                cropped_image, cropped_mask, _ = self.preprocessor.crop_to_bounding_box(masked, bbox, mask=mask_gray)

                preprocessed.append({
                    'idx': i,
                    'image_id': image_id,
                    'tree_id': tree_id,
                    'metadata': item.get('metadata'),
                    'cropped_image': cropped_image,
                    'cropped_mask': cropped_mask,
                    'bbox': bbox,
                })
            except Exception as e:
                logger.error(f"[ingest_batch] Preprocessing failed for item {i}: {e}")
                results[i] = IngestionResult(
                    success=False, image_id=item.get('image_id', ''), tree_id=item.get('tree_id', ''),
                    storage_keys={}, feature_info={}, message=f"Preprocessing failed: {e}"
                )

        valid = [p for p in preprocessed if p is not None]
        if not valid:
            return results

        # ── Step 2: Batch DINO extraction ──────────────────────────────────────
        dino_inputs = [self.preprocessor.prepare_for_dino(p['cropped_image']) for p in valid]
        dino_results = []
        if self.dino_processor:
            try:
                dino_results = self.dino_processor.extract_batch(dino_inputs, batch_size=batch_size)
                logger.info(f"[ingest_batch] DINO batch done: {len(dino_results)} features")
            except Exception as e:
                logger.error(f"[ingest_batch] DINO batch failed: {e}")
                dino_results = [None] * len(valid)
        else:
            dino_results = [None] * len(valid)

        # ── Step 3: Batch SuperPoint extraction ────────────────────────────────
        sp_inputs = []
        sp_masks = []
        for p in valid:
            sp_img, _ = self.preprocessor.prepare_for_superpoint(p['cropped_image'])
            sp_gray = self.preprocessor.to_grayscale(sp_img)
            sp_inputs.append(sp_gray)
            sp_masks.append(p['cropped_mask'])

        sp_results = []
        if self.superpoint_processor:
            try:
                sp_results = self.superpoint_processor.extract_batch(sp_inputs, masks=sp_masks, batch_size=batch_size)
                logger.info(f"[ingest_batch] SuperPoint batch done: {len(sp_results)} features")
            except Exception as e:
                logger.error(f"[ingest_batch] SuperPoint batch failed: {e}")
                sp_results = [None] * len(valid)
        else:
            sp_results = [None] * len(valid)

        # ── Step 4: Store each item (bark texture extraction removed) ────────────

        for j, p in enumerate(valid):
            i = p['idx']
            image_id = p['image_id']
            tree_id = p['tree_id']
            cropped_image = p['cropped_image']
            cropped_mask = p['cropped_mask']
            bbox = p['bbox']

            global_features = dino_results[j] if j < len(dino_results) else None
            local_features = sp_results[j] if j < len(sp_results) else None

            try:
                if global_features is None:
                    raise ValueError("DINO feature extraction failed - global_features is None")

                # Compute evidence UUID once — shared across MinIO and PostgreSQL.
                evidence_uuid = _resolve_evidence_uuid(image_id)
                evidence_id_str = str(evidence_uuid)

                minio_key = "minio_not_configured"
                if self.minio_repo and local_features:
                    features_to_save = {
                        'keypoints': local_features.keypoints,
                        'descriptors': local_features.descriptors,
                        'scores': local_features.scores,
                    }
                    minio_result = self.minio_repo.save_features(
                        image_id=evidence_id_str,
                        features=features_to_save,
                        metadata={'tree_id': tree_id, 'image_id': image_id, 'image_size': local_features.image_size}
                    )
                    minio_key = minio_result.storage_key if hasattr(minio_result, 'storage_key') else str(minio_result)

                # Vector store: PostgreSQL only
                postgres_key = None
                geo_metadata = p.get('metadata', {}) or {}
                longitude = geo_metadata.get("longitude")
                latitude = geo_metadata.get("latitude")
                hor_angle = geo_metadata.get("hor_angle")
                ver_angle = geo_metadata.get("ver_angle")
                pitch = geo_metadata.get("pitch")
                captured_at = self._parse_captured_at(geo_metadata.get("captured_at"))

                vector_list = []
                if global_features and hasattr(global_features, 'vector') and global_features.vector is not None:
                    vector_list = global_features.vector.tolist() if hasattr(global_features.vector, 'tolist') else list(global_features.vector)

                def _persist_postgres_batch(repo: SQLAlchemyORMRepository) -> Optional[str]:
                    tree = repo.get_tree(tree_id)
                    if not tree:
                        lat = geo_metadata.get("latitude") if geo_metadata else None
                        lon = geo_metadata.get("longitude") if geo_metadata else None
                        repo.create_tree(
                            tree_id=tree_id,
                            region_code=geo_metadata.get("region_code", "default"),
                            farm_id=geo_metadata.get("farm_id", "default_farm"),
                            geohash_7=geo_metadata.get("geohash_7", "default"),
                            row_idx=geo_metadata.get("row_idx") if geo_metadata else None,
                            col_idx=geo_metadata.get("col_idx") if geo_metadata else None,
                            longitude=lon,
                            latitude=lat,
                        )

                    evidence_meta = dict(geo_metadata)
                    evidence_meta["image_id"] = image_id
                    evidence_meta["minio_key"] = minio_key

                    evidence_id = repo.create_evidence(
                        tree_id=tree_id,
                        region_code=geo_metadata.get("region_code", "default"),
                        global_vector=vector_list,
                        storage_cid=minio_key,
                        evidence_id=evidence_id_str,
                        evidence_hash=geo_metadata.get("evidence_hash"),
                        camera_heading=int(hor_angle) if hor_angle is not None else None,
                        camera_pitch=int(pitch) if pitch is not None else None,
                        camera_roll=int(ver_angle) if ver_angle is not None else None,
                        is_c2pa_verified=geo_metadata.get("is_c2pa_verified", False),
                        raw_telemetry={
                            "model": global_features.model_name,
                            "bbox": bbox,
                            "image_size": global_features.image_size,
                            "ver_angle": ver_angle,
                        },
                        metadata=evidence_meta,
                        latitude=latitude,
                        longitude=longitude,
                        captured_at=captured_at,
                    )
                    return str(evidence_id) if evidence_id else None

                # Store to PostgreSQL (wrapped in transaction so evidence is committed atomically)
                # Use a fresh per-request repo to avoid mutating the shared singleton instance.
                if global_features and self.postgres_repo:
                    try:
                        with get_db_session() as session:
                            # Fresh repo instance per request — prevents concurrent session overwrite
                            repo = SQLAlchemyORMRepository(session)
                            postgres_key = _persist_postgres_batch(repo)
                        # get_db_session commits on normal exit, rolls back on exception

                    except Exception as primary_err:
                        logger.warning(
                            "[ingest_batch] DB session path failed, retrying with injected postgres_repo: %s",
                            primary_err,
                        )
                        try:
                            postgres_key = _persist_postgres_batch(self.postgres_repo)
                        except Exception as e:
                            logger.error(f"[ingest_batch] PostgreSQL insert failed: {e}")
                            raise ValueError(f"PostgreSQL storage failed: {e}") from e

                    if postgres_key:
                        logger.info(f"[ingest_batch] ✓ Stored in PostgreSQL: {postgres_key}")
                    else:
                        raise ValueError("PostgreSQL create_evidence returned None")
                elif global_features and not self.postgres_repo:
                    raise ValueError(
                        "PostgreSQL repository is not configured; "
                        "cannot persist global vector for ingestion."
                    )

                # Build storage keys for PostgreSQL only
                storage_keys = {
                    "minio_key": minio_key,
                }
                if postgres_key:
                    storage_keys["postgres_key"] = postgres_key
                    storage_keys["vector_key"] = postgres_key

                results[i] = IngestionResult(
                    success=True,
                    image_id=image_id,
                    tree_id=tree_id,
                    storage_keys=storage_keys,
                    feature_info={
                        "global_dim": len(global_features.vector) if global_features else 0,
                        "n_keypoints": len(local_features.keypoints) if local_features else 0,
                        "descriptor_dim": local_features.descriptors.shape[1] if local_features and len(local_features.descriptors.shape) > 1 else 0,
                    },
                    message=f"Successfully ingested {image_id}",
                )
            except Exception as e:
                logger.error(f"[ingest_batch] Storage failed for {image_id}: {e}")
                results[i] = IngestionResult(
                    success=False, image_id=image_id, tree_id=tree_id,
                    storage_keys={}, feature_info={}, message=f"Storage failed: {e}"
                )

        return results

    # ── Validators ────────────────────────────────────────────────────────────

    def _validate_inputs(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        image_id: str,
        tree_id: str,
    ) -> None:
        """Validate inputs for ingest(). Raises ValueError on any violation."""
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError(f"Invalid image: expected numpy array, got {type(image)}")
        if image.size == 0:
            raise ValueError("Image is empty")
        if len(image.shape) not in [2, 3]:
            raise ValueError(f"Invalid image shape: {image.shape}, expected 2D or 3D array")

        if mask is None or not isinstance(mask, np.ndarray):
            raise ValueError(f"Invalid mask: expected numpy array, got {type(mask)}")
        if mask.size == 0:
            raise ValueError("Mask is empty")
        if mask.shape[:2] != image.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape}doesn't match image shape {image.shape}")

        if not image_id or not isinstance(image_id, str):
            raise ValueError(f"Invalid image_id: {image_id}")
        if not tree_id or not isinstance(tree_id, str):
            raise ValueError(f"Invalid tree_id: {tree_id}")

    def _validate_dino_result(self, result: Any, expected_dim: int) -> None:
        """Validate a DinoResult. Raises ValueError on any violation."""
        if result is None:
            raise ValueError("DINO extraction returned None")
        if not hasattr(result, 'vector'):
            raise ValueError("DINO result missing 'vector' attribute")
        if result.vector is None or result.vector.size == 0:
            raise ValueError("DINO vector is empty")
        if not isinstance(result.vector, np.ndarray):
            raise ValueError(f"DINO vector is not numpy array: {type(result.vector)}")
        if len(result.vector.shape) != 1:
            raise ValueError(f"DINO vector should be 1D, got shape {result.vector.shape}")
        if np.isnan(result.vector).any() or np.isinf(result.vector).any():
            raise ValueError("DINO vector contains NaN or Inf values")
        actual_dim = len(result.vector)
        if actual_dim != expected_dim:
            raise ValueError(
                f"DINO output dimension mismatch!\n"
                f"  Expected (from appConfig.milvus_vector_dim): {expected_dim}\n"
                f"  Got from model: {actual_dim}\n"
                f"  Model: {result.model_name}\n"
                f"  Fix: Update appConfig.dino_model_type or milvus_vector_dim"
            )

    def _validate_superpoint_result(self, result: Any) -> None:
        """Validate a SuperPointResult. Raises ValueError on any violation."""
        if result is None:
            raise ValueError("SuperPoint extraction returned None")
        if not hasattr(result, 'keypoints') or not hasattr(result, 'descriptors'):
            raise ValueError("SuperPoint result missing 'keypoints' or 'descriptors' attribute")
        if result.keypoints is None or result.keypoints.size == 0:
            logger.warning("[STEP 3] ⚠️  No keypoints detected")
        else:
            if not isinstance(result.keypoints, np.ndarray):
                raise ValueError(f"Keypoints is not numpy array: {type(result.keypoints)}")
            if len(result.keypoints.shape) != 2 or result.keypoints.shape[1] != 2:
                raise ValueError(f"Keypoints should be Nx2, got shape {result.keypoints.shape}")
            if np.isnan(result.keypoints).any() or np.isinf(result.keypoints).any():
                raise ValueError("Keypoints contain NaN or Inf values")
        if result.descriptors is None or result.descriptors.size == 0:
            raise ValueError("Descriptors are empty")
        if not isinstance(result.descriptors, np.ndarray):
            raise ValueError(f"Descriptors is not numpy array: {type(result.descriptors)}")
        if len(result.descriptors.shape) != 2:
            raise ValueError(f"Descriptors should be 2D, got shape {result.descriptors.shape}")
        if np.isnan(result.descriptors).any() or np.isinf(result.descriptors).any():
            raise ValueError("Descriptors contain NaN or Inf values")

    def _validate_local_features_for_storage(self, local_features: Any) -> None:
        """Validate local features before MinIO storage. Raises ValueError on any violation."""
        if local_features.keypoints is None or local_features.keypoints.size == 0:
            raise ValueError("Cannot store empty keypoints")
        if local_features.descriptors is None or local_features.descriptors.size == 0:
            raise ValueError("Cannot store empty descriptors")
        if len(local_features.keypoints) != local_features.descriptors.shape[0]:
            raise ValueError(
                f"Keypoint count {len(local_features.keypoints)} "
                f"doesn't match descriptor count {local_features.descriptors.shape[0]}"
            )
        if len(local_features.scores) != local_features.descriptors.shape[0]:
            raise ValueError(
                f"Score count {len(local_features.scores)} "
                f"doesn't match descriptor count {local_features.descriptors.shape[0]}"
            )

    # ── High-level entry points (replaces IngestionPipeline) ──────────────────

    def ingest_raw(self, image: np.ndarray, image_id: str, tree_id: str,
                   gps_angle: Optional[Dict[str, Any]] = None,
                   captured_at: Optional[int] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> IngestionResult:
        """SAM3 segment → ingest.

        segment_with_sam3 returns (segmented_image, mask).  We pass the original
        image + mask to ingest() so that Step 1 does the single apply_mask there,
        avoiding a redundant double-mask.

        Args:
            image: Input image as numpy array
            image_id: Unique identifier for the image
            tree_id: Tree identifier
            gps_angle: GPS and angle dict with latitude, longitude, hor_angle, ver_angle, pitch
            captured_at: Optional capture timestamp as Unix epoch seconds
            metadata: Optional additional metadata dict (other fields)
        """
        try:
            _, mask = self.preprocessor.segment_with_sam3(image)
        except Exception as e:
            logger.error(f"SAM3 processing failed: {e}")
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        # Build final metadata combining gps_angle and additional metadata
        final_metadata = metadata.copy() if metadata else {}

        # Add GPS/angle fields from gps_angle dict
        if gps_angle:
            for key in ['latitude', 'longitude', 'hor_angle', 'ver_angle', 'pitch']:
                if key in gps_angle:
                    final_metadata[key] = gps_angle[key]

        # Add captured_at (takes precedence)
        if captured_at is not None:
            final_metadata['captured_at'] = captured_at

        return self.ingest(image, mask, image_id, tree_id, final_metadata)

    def ingest_raw_batch(self, items: list, batch_size: int = 8) -> list:
        """SAM3 segment all → batch ingest.

        items: List of dicts with keys: image, image_id, tree_id, metadata (optional)
        """
        if not items:
            return []

        segmented = []
        for item in items:
            try:
                _, mask = self.preprocessor.segment_with_sam3(item['image'])
            except Exception as e:
                logger.error(f"[ingest_raw_batch] SAM3 failed for {item.get('image_id')}: {e}")
                mask = np.ones(item['image'].shape[:2], dtype=np.uint8) * 255
            segmented.append({
                'image': item['image'],  # original image — ingest() does apply_mask once
                'mask': mask,
                'image_id': item['image_id'],
                'tree_id': item['tree_id'],
                'metadata': item.get('metadata'),
            })

        return self.ingest_batch(segmented, batch_size=batch_size)

    def ingest_raw_box(
        self,
        image: np.ndarray,
        bounding_boxes: list,
        image_id: str,
        tree_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionResult:
        """SAM3 box-prompt segment -> ingest using the original image + generated mask."""
        try:
            _, mask = self.preprocessor.segment_with_sam3_box(image, bounding_boxes)
        except Exception as e:
            logger.error(f"SAM3 box processing failed: {e}")
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        return self.ingest(image, mask, image_id, tree_id, metadata)

    # def ingest_raw_box_batch(self, items: list, batch_size: int = 8) -> list:
    #     """SAM3 box-prompt segment all → batch ingest.

    #     items: List of dicts with keys:
    #         - image: np.ndarray (BGR)
    #         - image_id: str
    #         - tree_id: str
    #         - bounding_boxes: list of [x1,y1,x2,y2] (optional, defaults to full image)
    #         - metadata: dict (optional)
    #     """
    #     if not items:
    #         return []

    #     segmented = []
    #     for item in items:
    #         image = item['image']
    #         bounding_boxes = item.get('bounding_boxes')
    #         if not bounding_boxes:
    #             h, w = image.shape[:2]
    #             bounding_boxes = [[0, 0, w, h]]
    #         try:
    #             _, mask = self.preprocessor.segment_with_sam3_box(image, bounding_boxes)
    #         except Exception as e:
    #             logger.error(f"[ingest_raw_box_batch] SAM3 box failed for {item.get('image_id')}: {e}")
    #             mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    #         segmented.append({
    #             'image': image,
    #             'mask': mask,
    #             'image_id': item['image_id'],
    #             'tree_id': item['tree_id'],
    #             'metadata': item.get('metadata'),
    #         })

    #     return self.ingest_batch(segmented, batch_size=batch_size)


# def create_ingestion_service(
#     preprocessor: Optional[PreprocessorService] = None,
#     app_config: Optional[AppConfig] = None,
#     dino_processor: Optional[DinoProcessor] = None,
#     superpoint_processor: Optional[SuperPointProcessor] = None,
#     postgres_repo: Optional[SQLAlchemyORMRepository] = None,
#     minio_repo: Optional[MinIORepository] = None,
#     verbose: bool = False,
# ) -> "IngestionService":
#     """
#     Factory function to create IngestionService with dependency injection.

#     This factory function is the recommended way to create an IngestionService.
#     It accepts optional dependencies (injected via container) and optional configuration.

#     Args:
#         preprocessor: Image preprocessing service (injected via container)
#         app_config: Application configuration (injected via container)
#         dino_processor: DINO feature extractor (injected via container)
#         superpoint_processor: SuperPoint feature extractor (injected via container)
#         postgres_repo: PostgreSQL repository (injected via container)
#         minio_repo: MinIO repository (injected via container)
#         verbose: Enable verbose logging

#     Returns:
#         Configured IngestionService

#     Example:
#         # Using container DI:
#         service = create_ingestion_service(
#             preprocessor=container.preprocessor_service(),
#             app_config=container.app_config(),
#             dino_processor=container.dino_processor(),
#             ...
#         )
        
#         # Using with explicit overrides:
#         service = create_ingestion_service(
#             dino_processor=my_dino,
#             postgres_repo=my_repo,
#             verbose=True
#         )
#     """
#     return IngestionService(
#         preprocessor=preprocessor,
#         app_config=app_config,
#         dino_processor=dino_processor,
#         superpoint_processor=superpoint_processor,
#         postgres_repo=postgres_repo,
#         minio_repo=minio_repo,
#         verbose=verbose,
#     )


# def create_ingestion_service_legacy(config_dict: Optional[Dict[str, Any]] = None, verbose: bool = False) -> "IngestionService":
#     """Legacy factory - creates dependencies manually. Use create_ingestion_service instead."""
#     app_config: AppConfig = get_config()

#     if config_dict is None:
#         config_dict = {
#             "dino_model_type": app_config.dino_model_type,
#             "dino_device": app_config.dino_device,
#             "dino_use_multi_gpu": app_config.dino_use_multi_gpu,
#             "dino_gpu_ids": app_config.dino_gpu_ids,
#             "dino_enable_memory_optimization": app_config.dino_enable_memory_optimization,
#             "dino_use_gradient_checkpointing": app_config.dino_use_gradient_checkpointing,
#             "dino_use_4bit_quantization": app_config.dino_use_4bit_quantization,
#             "hf_token": app_config.dino_hf_token,
#             "image_size": app_config.dino_image_size,
#             "sp_max_keypoints": app_config.sp_max_keypoints,
#             "sp_max_dimension": app_config.sp_max_dimension,
#             "sp_device": app_config.sp_device,
#             "minio_endpoint": app_config.minio_endpoint,
#             "minio_bucket": app_config.minio_bucket,
#             "verbose": verbose,
#         }

#     preprocessor = PreprocessorService(background_color="black")

#     dino = None
#     try:
#         dino = DinoProcessor(DinoConfig(
#             model_type=config_dict.get("dino_model_type", "dinov3-vitl16"),
#             device=config_dict.get("dino_device", "cuda"),
#             hf_token=config_dict.get("hf_token", ""),
#             verbose=config_dict.get("verbose", False),
#             use_multi_gpu=config_dict.get("dino_use_multi_gpu", False),
#             gpu_ids=config_dict.get("dino_gpu_ids", None),
#             enable_memory_optimization=config_dict.get("dino_enable_memory_optimization", True),
#             use_gradient_checkpointing=config_dict.get("dino_use_gradient_checkpointing", True),
#             use_4bit_quantization=config_dict.get("dino_use_4bit_quantization", False),
#         ))
#     except Exception as e:
#         if verbose:
#             logger.warning(f"DINO processor skipped: {e}")

#     superpoint = None
#     try:
#         superpoint = SuperPointProcessor(SuperPointConfig(
#             max_keypoints=config_dict.get("sp_max_keypoints", app_config.sp_max_keypoints),
#             max_dimension=config_dict.get("sp_max_dimension", app_config.sp_max_dimension),
#             device=config_dict.get("sp_device", app_config.sp_device),
#             verbose=config_dict.get("verbose", False),
#         ))
#     except ImportError:
#         if verbose:
#             logger.warning("SuperPoint processor skipped - package not available")

#     # ============================================================================
#     # SQLAlchemy ORM Repository Initialization
#     # ============================================================================
#     # Creates SQLAlchemyORMRepository instance for vector storage with pgvector
#     # ============================================================================
#     postgres = None
#     try:
#         from src.repository.databaseManager import init_db
#         db_manager = init_db(
#             host=app_config.postgres_host,
#             port=app_config.postgres_port,
#             database=app_config.postgres_db,
#             user=app_config.postgres_user,
#             password=app_config.postgres_password,
#         )
#         if db_manager.is_connected:
#             postgres = SQLAlchemyORMRepository()
#             logger.info(f"Connected to PostgreSQL via SQLAlchemy: {app_config.postgres_host}:{app_config.postgres_port}/{app_config.postgres_db}")
#         else:
#             logger.error("PostgreSQL connection failed, ingestion will not work")
#             postgres = None
#     except Exception as e:
#         logger.error(f"SQLAlchemy repository creation failed: {e}")
#         postgres = None

#     minio = MinIORepository(MinIOConfig(
#         endpoint=config_dict.get("minio_endpoint", "localhost:9000"),
#         bucket=config_dict.get("minio_bucket", "tree-features"),
#         verbose=config_dict.get("verbose", False),
#     ))

#     return IngestionService(
#         preprocessor=preprocessor,
#         dino_processor=dino,
#         superpoint_processor=superpoint,
#         postgres_repo=postgres,
#         minio_repo=minio,
#         verbose=verbose,
#         app_config=app_config,
#     )
