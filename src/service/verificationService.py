"""
Verification Service - Orchestrates the verification pipeline

This service coordinates:
1. Preprocessing (via preprocessor)
2. Feature extraction (via processors)
3. Coarse retrieval (via Milvus)
4. Fine-grained matching (via LightGlue)
5. Decision making (via strategy)

UPDATED: Now uses Hierarchical Matching by default for improved accuracy.

Dependency Injection:
    This service uses @inject decorator for DI with dependency_injector.
    Use Depends(Provide[Container.verification_service]) in FastAPI routes.
    
    Usage in FastAPI:
        from src.api.dependencies import get_verification_service
        
        @router.post("/verify")
        async def verify(
            service: VerificationService = Depends(get_verification_service)
        ):
            return await service.verify(...)
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
import cv2
import numpy as np
logger = logging.getLogger(__name__)

from dependency_injector.wiring import inject, Provide
from dependency_injector import containers

from src.config.appConfig import AppConfig

# Import hierarchical matching
from .hierarchicalMatchingService import (
    HierarchicalMatchingService,
    HierarchicalVerificationService,
)

from src.service.preprocessorService import PreprocessorService
from src.processor.dinoProcessor import DinoProcessor, DinoResult
from src.processor.superPointProcessor import SuperPointProcessor, SuperPointResult
from src.processor.lightGlueProcessor import LightGlueProcessor
from src.repository.milvusRepository import MilvusRepository
from src.repository.sqlalchemyRepository import SQLAlchemyORMRepository
from src.repository.minioRepository import MinIORepository
from src.utils.matchingStrategy import MatchingStrategy


@dataclass
class MatchCandidate:
    """A candidate match from coarse retrieval."""
    image_id: str
    tree_id: str
    similarity_score: float
    rank: int
    metadata: Optional[Dict[str, Any]] = None


class VerificationService:
    """Service for verifying tree identity through feature matching.
    
    Uses @inject decorator for dependency injection with dependency_injector.
    Wire with: container.wire(modules=[__name__])
    
    Usage in FastAPI:
        from src.api.dependencies import get_verification_service
        
        @router.post("/verify")
        async def verify(
            service: VerificationService = Depends(get_verification_service)
        ):
            return await service.verify(...)
    """
    
    @inject
    def __init__(
        self,
        preprocessor: PreprocessorService = Provide["preprocessor_service"],
        app_config: AppConfig = Provide["app_config"],
        dino_processor: DinoProcessor = Provide["dino_processor"],
        superpoint_processor: SuperPointProcessor = Provide["superpoint_processor"],
        lightglue_processor: LightGlueProcessor = Provide["lightglue_processor"],
        milvus_repo: MilvusRepository = Provide["milvus_repo"],
        postgres_repo: SQLAlchemyORMRepository = Provide["sqlalchemy_repo"],
        minio_repo: MinIORepository = Provide["minio_repo"],
        hierarchical_matching_service: HierarchicalMatchingService = Provide["hierarchical_matching_service"],
        top_k: int = 10,
        coarse_threshold: float = 0.6,
        inlier_threshold: int = 15,
        verbose: bool = False,
        use_hierarchical: bool = True,
        vector_store_type: str = "postgres",
        **kwargs: Any
    ) -> None:
        """Initialize verification service with injected dependencies.

        Args:
            preprocessor: Image preprocessing service (injected via Provide)
            app_config: Application configuration (injected via Provide)
            dino_processor: DinoV2/V3 feature extractor (injected via Provide)
            superpoint_processor: SuperPoint feature extractor (injected via Provide)
            lightglue_processor: LightGlue matcher (injected via Provide)
            milvus_repo: Milvus vector store repository (injected via Provide)
            postgres_repo: SQLAlchemy ORM repository with pgvector (injected via Provide)
            minio_repo: MinIO feature store repository (injected via Provide)
            hierarchical_matching_service: Hierarchical matching service (injected via Provide)
            top_k: Number of candidates to retrieve
            coarse_threshold: Minimum similarity for coarse retrieval
            inlier_threshold: Minimum inliers for match decision
            verbose: Enable verbose logging
            use_hierarchical: Use hierarchical matching if available (DEFAULT: True)
            vector_store_type: "postgres" or "milvus" for vector storage
        """
        self.preprocessor: PreprocessorService = preprocessor
        self.app_config: AppConfig = app_config
        self.dino_processor: DinoProcessor = dino_processor
        self.superpoint_processor: SuperPointProcessor = superpoint_processor
        self.lightglue_processor: LightGlueProcessor = lightglue_processor
        self.milvus_repo: MilvusRepository = milvus_repo
        self.postgres_repo: SQLAlchemyORMRepository = postgres_repo
        self.minio_repo: MinIORepository = minio_repo
        self.top_k: int = top_k
        self.coarse_threshold: float = coarse_threshold
        self.inlier_threshold: int = inlier_threshold
        self.use_hierarchical: bool = use_hierarchical
        self.verbose: bool = verbose
        self.vector_store_type: str = vector_store_type
        self.matching_strategy: Optional[MatchingStrategy] = None

        # Initialize default matching strategy
        from src.utils.matchingStrategy import MatchingStrategy, MatchingStrategyConfig
        config = MatchingStrategyConfig(inlier_threshold=inlier_threshold)
        self.matching_strategy = MatchingStrategy(config=config)

        # Use injected hierarchical matching service (DI)
        self.hierarchical_service: Optional[HierarchicalMatchingService] = None
        self.verification_service: Optional[HierarchicalVerificationService] = None
        if use_hierarchical:
            try:
                # Use DI-injected hierarchical matching service
                self.hierarchical_service = hierarchical_matching_service
                self.verification_service = HierarchicalVerificationService(
                    hierarchical_service=self.hierarchical_service,
                    confidence_threshold=0.7,
                    verbose=verbose
                )
                if verbose:
                    logger.info("✅ Hierarchical matching service initialized via DI")
            except Exception as e:
                logger.warning(f"Failed to initialize hierarchical matching: {e}")
                logger.warning("Falling back to coarse matching")
                self.use_hierarchical = False

        # Update matching strategy thresholds
        if self.matching_strategy:
            self.matching_strategy.set_thresholds(inlier_threshold=inlier_threshold)

    def _prepare_verification(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[DinoResult, SuperPointResult, Optional[np.ndarray]]:
        """Segment image with SAM3, crop, and extract DINO / SuperPoint / texture.

        Exactly mirrors the ingestion pipeline in IngestionService so that all
        feature vectors are in the same space:
            - If mask provided: use mask directly (same as ingestion)
            - If mask None: run SAM3 to generate mask (fallback for raw images)
            → apply_mask → crop_to_bbox
            → prepare_for_dino        → DINO extract
            → prepare_for_superpoint → to_grayscale → SuperPoint extract
            → extract_texture_features

        Args:
            image: Raw input image (BGR numpy array)
            mask: Optional segmentation mask. If provided, used directly (same as ingestion).
                  If None, SAM3 runs to generate mask (fallback for raw/unprocessed images).

        Returns:
            Tuple of (dino_result, superpoint_result, texture_histogram)
        """
        # ── Step 1: Get segmentation mask ────────────────────────────────────────
        # Mirror ingestion: use provided mask, or fallback to SAM3
        if mask is not None:
            # Use provided mask directly (same as ingestion)
            if len(mask.shape) == 3:
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask
            logger.info("[VERIFY] Using provided mask for verification")
        else:
            # Fallback: run SAM3 (for raw/unprocessed images)
            _, seg_mask = self.preprocessor.segment_with_sam3(image)
            if len(seg_mask.shape) == 3:
                mask_gray = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = seg_mask
            logger.info("[VERIFY] Generated mask via SAM3 (mask was None)")

        # ── Step 2: apply mask + crop (same as ingestion) ─────────────────────
        masked_image = self.preprocessor.apply_mask(image, mask_gray)
        bbox = self.preprocessor.get_bounding_box(mask_gray)
        cropped_image, cropped_mask, _ = self.preprocessor.crop_to_bounding_box(
            masked_image, bbox, mask=mask_gray
        )

        # ── Step 3a: DINO (same as ingestion) ─────────────────────────────────
        dino_input = self.preprocessor.prepare_for_dino(cropped_image)
        dino_result = self.dino_processor.extract(dino_input)
        logger.info(
            f"[VERIFY] DINO first 5: {dino_result.global_descriptor[:5]}, "
            f"norm={np.linalg.norm(dino_result.global_descriptor):.4f}"
        )

        # ── Step 3b: SuperPoint — MUST mirror ingestion exactly ────────────────
        # Ingestion: prepare_for_superpoint → to_grayscale → SP.extract(gray)
        # NOT: SP.extract(cropped_image)  ← wrong, different image format
        sp_input, _ = self.preprocessor.prepare_for_superpoint(cropped_image)
        sp_gray = self.preprocessor.to_grayscale(sp_input)
        superpoint_result = self.superpoint_processor.extract(sp_gray, mask=cropped_mask)

        # ── Step 3c: Bark texture (removed) ──────────────────────────
        # Bark texture processor removed - texture_histogram is always None now

        return dino_result, superpoint_result, None

    async def verify(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        known_tree_id: Optional[str] = None,
        geo_filter: Optional[Dict[str, float]] = None,
        angle_filter: Optional[Dict[str, float]] = None,
        time_filter: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Verify tree identity by matching features.

        Uses hierarchical matching by default. When known_tree_id is NOT provided,
        uses multi-tree hierarchical flow for /verify endpoint.

        Args:
            image: Query image data
            mask: Optional segmentation mask. If provided, used directly (same as ingestion).
                  If None, SAM3 runs to generate mask (fallback for raw images).
            known_tree_id: Optional known tree ID for single-tree matching. If None, uses multi-tree mode
            geo_filter: Optional dict with lat_min, lat_max, lon_min, lon_max for location filtering
            angle_filter: Optional dict with angle bounds (hor_angle_min/max, ver_angle_min/max, pitch_min/max)
            time_filter: Optional dict with captured_at_min, captured_at_max (epoch seconds)

        Returns:
            Dict with verification result
        """

        try:
            is_valid, error_msg = self.validate(image=image, mask=mask, known_tree_id=known_tree_id)
            if not is_valid:
                raise ValueError(error_msg)

            if not self.use_hierarchical:
                raise ValueError("Hierarchical matching is not enabled")

            dino_result, superpoint_result, _ = self._prepare_verification(image, mask=mask)

            result = await self.hierarchical_service.match_async(
                query_dino=dino_result,
                query_superpoint=superpoint_result,
                geo_filter=geo_filter,
                angle_filter=angle_filter,
                time_filter=time_filter,
            )

            if result is None:
                raise ValueError("Hierarchical verification returned None")
            return result

        except Exception as e:
            logger.error(f"[VERIFY FAILED] {type(e).__name__}: {e}")
            return {
                "decision": "ERROR",
                "reason": str(e),
                "best_match": None,
                "all_matches": [],
            }

    def validate(
        self,
        image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        dino_result: Optional[DinoResult] = None,
        superpoint_result: Optional[SuperPointResult] = None,
        known_tree_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate inputs for verification functions.

        Performs comprehensive validation of all inputs commonly used in verify functions.
        Returns (is_valid, error_message) tuple.

        Args:
            image: Optional query image data (numpy array)
            mask: Optional segmentation mask (numpy array)
            dino_result: Optional DINO extraction result
            superpoint_result: Optional SuperPoint extraction result
            known_tree_id: Optional known tree ID string

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
            - is_valid: True if all validations pass, False otherwise
            - error_message: Error description if validation fails, None otherwise
        """
        # Validate image if provided
        if image is not None:
            if not isinstance(image, np.ndarray):
                return False, f"Invalid image: expected numpy array, got {type(image)}"
            if image.size == 0:
                return False, "Image is empty"
            if len(image.shape) not in [2, 3]:
                return False, f"Invalid image shape: {image.shape}, expected 2D or 3D array"

        # Validate mask if provided
        if mask is not None:
            if not isinstance(mask, np.ndarray):
                return False, f"Invalid mask: expected numpy array, got {type(mask)}"
            if mask.size == 0:
                return False, "Mask is empty"
            if image is not None and mask.shape[:2] != image.shape[:2]:
                return False, f"Mask shape {mask.shape} doesn't match image shape {image.shape}"

        # Validate DINO result if provided
        if dino_result is not None:
            if not hasattr(dino_result, 'global_descriptor'):
                return False, "DINO result missing 'global_descriptor' attribute"
            if dino_result.global_descriptor is None or dino_result.global_descriptor.size == 0:
                return False, "DINO descriptor is empty"
            if np.isnan(dino_result.global_descriptor).any() or np.isinf(dino_result.global_descriptor).any():
                return False, "DINO descriptor contains NaN or Inf values"

        # Validate SuperPoint result if provided
        if superpoint_result is not None:
            if not hasattr(superpoint_result, 'keypoints') or not hasattr(superpoint_result, 'descriptors'):
                return False, "SuperPoint result missing 'keypoints' or 'descriptors' attribute"

            # Validate keypoints if present
            if superpoint_result.keypoints is not None and superpoint_result.keypoints.size > 0:
                if not isinstance(superpoint_result.keypoints, np.ndarray):
                    return False, f"Keypoints is not numpy array: {type(superpoint_result.keypoints)}"
                if len(superpoint_result.keypoints.shape) != 2 or superpoint_result.keypoints.shape[1] != 2:
                    return False, f"Keypoints should be Nx2, got shape {superpoint_result.keypoints.shape}"
                if np.isnan(superpoint_result.keypoints).any() or np.isinf(superpoint_result.keypoints).any():
                    return False, "Keypoints contain NaN or Inf values"

            # Validate descriptors
            if superpoint_result.descriptors is None or superpoint_result.descriptors.size == 0:
                return False, "Descriptors are empty"
            if not isinstance(superpoint_result.descriptors, np.ndarray):
                return False, f"Descriptors is not numpy array: {type(superpoint_result.descriptors)}"
            if len(superpoint_result.descriptors.shape) != 2:
                return False, f"Descriptors should be 2D, got shape {superpoint_result.descriptors.shape}"
            if np.isnan(superpoint_result.descriptors).any() or np.isinf(superpoint_result.descriptors).any():
                return False, "Descriptors contain NaN or Inf values"

        # Validate known_tree_id if provided
        if known_tree_id is not None:
            if not isinstance(known_tree_id, str):
                return False, f"known_tree_id must be string, got {type(known_tree_id)}"
            if not known_tree_id.strip():
                return False, "known_tree_id cannot be empty or whitespace"

        return True, None

    # async def _extract_dino(self, image: np.ndarray, use_async: bool):
    #     """Extract DINO features."""
    #     import asyncio

    #     if self.dino_processor is None:
    #         raise ValueError("DINO processor required for hierarchical matching")

    #     logger.info(f"[{'ASYNC' if use_async else 'SYNC'}] Extracting DINO features...")
    #     dino_result = await asyncio.to_thread(self.dino_processor.extract, image) if use_async else self.dino_processor.extract(image)

    #     if dino_result is None:
    #         logger.warning("⚠️ DINO extraction failed, falling back to SuperPoint-only")
    #         return None

    #     is_valid, error_msg = self.validate(dino_result=dino_result)
    #     if not is_valid:
    #         raise ValueError(error_msg)

    #     logger.info(f"  ✓ DINO extracted: {dino_result.global_descriptor.shape}, {dino_result.model_name}")
    #     return dino_result

    # def _prepare_grayscale(self, image: np.ndarray) -> np.ndarray:
    #     """Convert image to grayscale if needed."""
    #     if len(image.shape) == 3 and image.shape[2] == 3:
    #         return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     return image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # async def _extract_superpoint(self, gray: np.ndarray, mask: Optional[np.ndarray], use_async: bool):
    #     """Extract SuperPoint features."""
    #     import asyncio

    #     logger.info(f"[{'ASYNC' if use_async else 'SYNC'}] Extracting SuperPoint features...")
    #     superpoint_result = await asyncio.to_thread(self.superpoint_processor.extract, gray, mask) if use_async else self.superpoint_processor.extract(gray, mask=mask)

    #     if superpoint_result is None:
    #         raise ValueError("SuperPoint extraction returned None")

    #     is_valid, error_msg = self.validate(superpoint_result=superpoint_result)
    #     if not is_valid:
    #         if superpoint_result.keypoints is None or superpoint_result.keypoints.size == 0:
    #             logger.warning("⚠️ No keypoints detected")
    #         else:
    #             raise ValueError(error_msg)

    #     return superpoint_result


    # def _coarse_retrieval(self, query_vector: np.ndarray) -> List[MatchCandidate]:
    #     """Retrieve top-K candidates from vector store (PostgreSQL or Milvus).

    #     Args:
    #         query_vector: Query global feature vector

    #     Returns:
    #         List of MatchCandidate objects
    #     """
    #     candidate_dicts = []

    #     if self.vector_store_type == "postgres" and self.postgres_repo:
    #         # Use PostgreSQL with pgvector
    #         # Convert numpy vector to list
    #         vector_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)

    #         search_results = self.postgres_repo.search_similar_vectors(
    #             query_vector=vector_list,
    #             top_k=self.top_k,
    #             min_similarity=self.coarse_threshold,
    #         )

    #         # Convert VectorSearchResult to candidate dicts
    #         for rank, result in enumerate(search_results):
    #             candidate_dicts.append({
    #                 'image_id': result.metadata.get('image_id', ''),
    #                 'tree_id': result.tree_id,
    #                 'similarity_score': result.similarity,
    #                 'rank': rank,
    #                 'metadata': result.metadata,
    #             })

    #     elif self.vector_store_type == "milvus" and self.milvus_repo:
    #         # Use Milvus (original implementation)
    #         candidate_dicts = self.milvus_repo.coarse_retrieval(
    #             query_vector=query_vector,
    #             top_k=self.top_k,
    #             threshold=self.coarse_threshold,
    #             verbose=self.verbose
    #         )
    #     else:
    #         logger.warning(f"No vector store available (type: {self.vector_store_type})")
    #         return []

    #     # Convert dictionaries to MatchCandidate objects for backward compatibility
    #     candidates: List[MatchCandidate] = []
    #     for cand_dict in candidate_dicts:
    #         candidate = MatchCandidate(
    #             image_id=cand_dict['image_id'],
    #             tree_id=cand_dict['tree_id'],
    #             similarity_score=cand_dict['similarity_score'],
    #             rank=cand_dict['rank'],
    #         )
    #         # Attach metadata to candidate for later use
    #         candidate.metadata = cand_dict.get('metadata', {})
    #         candidates.append(candidate)

    #     return candidates

    # def _fine_grained_matching(
    #     self,
    #     query_local: Dict[str, Any],
    #     candidates: List[MatchCandidate],
    # ) -> List[Dict[str, Any]]:
    #     """Perform fine-grained matching for each candidate.

    #     Args:
    #         query_local: Query local features
    #         candidates: List of candidate matches

    #     Returns:
    #         List of match results with inlier counts
    #     """
    #     if self.lightglue_processor is None:
    #         if self.verbose:
    #             logger.warning("LightGlue processor not available, using coarse scores only")
    #         return [{
    #             "image_id": c.image_id,
    #             "tree_id": c.tree_id,
    #             "coarse_score": c.similarity_score,
    #             "coarse_similarity": c.similarity_score,
    #             "n_inliers": 0,
    #             "match_score": 0.0,
    #             "confidence": 0.0,
    #         } for c in candidates]

    #     match_results = []
    #     for candidate in candidates:
    #         try:
    #             minio_key = candidate.metadata.get('minio_key') if candidate.metadata else None
    #             if not minio_key:
    #                 raise NotImplementedError("Image ID-based lookup is not supported")

    #             candidate_local = self.minio_repo.load_features_by_key(minio_key)
    #             match_result = self.lightglue_processor.match(
    #                 query_features=query_local,
    #                 candidate_features=candidate_local,
    #             )

    #             match_results.append({
    #                 "image_id": candidate.image_id,
    #                 "tree_id": candidate.tree_id,
    #                 "coarse_score": candidate.similarity_score,
    #                 "coarse_similarity": candidate.similarity_score,
    #                 "n_inliers": match_result.n_inliers,
    #                 "match_score": match_result.match_score,
    #                 "confidence": match_result.confidence,
    #             })

    #         except Exception as e:
    #             import traceback
    #             logger.warning(f"Failed to match candidate {candidate.image_id}: {e}")
    #             if self.verbose:
    #                 logger.warning(f"Traceback: {traceback.format_exc()}")
    #             continue
        
    #     return match_results

    # def get_feature_info(self) -> Dict[str, Any]:
    #     """Get information about stored features."""
    #     try:
    #         # Check if Milvus is configured
    #         if self.milvus_repo is None:
    #             return {
    #                 "total_entities": 0,
    #                 "total_features": 0,
    #                 "features_by_tree": {},
    #                 "message": "Milvus not configured (using PostgreSQL)",
    #                 "status": "retrieved successfully"
    #             }

    #         # Get info from Milvus
    #         milvus_info = self.milvus_repo.get_collection_info()

    #         # Get info from MinIO
    #         minio_info = self.minio_repo.get_feature_count()

    #         return {
    #             "total_entities": milvus_info.get("entity_count", 0),
    #             "total_features": minio_info.get("total_features", 0),
    #             "features_by_tree": minio_info.get("features_by_tree", {}),
    #             "status": "retrieved successfully"
    #         }
    #     except Exception as e:
    #         logger.error(f"Failed to get feature info: {e}")
    #         return {
    #             "error": str(e),
    #             "total_entities": 0,
    #             "total_features": 0,
    #             "features_by_tree": {}
    #         }


# def create_verification_service(
#     preprocessor: Optional[PreprocessorService] = None,
#     app_config: Optional[AppConfig] = None,
#     dino_processor: Optional[DinoProcessor] = None,
#     superpoint_processor: Optional[SuperPointProcessor] = None,
#     lightglue_processor: Optional[LightGlueProcessor] = None,
#     milvus_repo: Optional[MilvusRepository] = None,
#     postgres_repo: Optional[SQLAlchemyORMRepository] = None,
#     minio_repo: Optional[MinIORepository] = None,
#     matching_strategy: Optional[MatchingStrategy] = None,
#     top_k: int = 10,
#     coarse_threshold: float = 0.6,
#     inlier_threshold: int = 15,
#     verbose: bool = False,
#     use_hierarchical: bool = True,
#     vector_store_type: str = "postgres",
#     **kwargs: Any
# ) -> "VerificationService":
#     """
#     Factory function to create VerificationService with dependency injection.

#     This factory function is the recommended way to create a VerificationService.
#     It accepts optional dependencies (injected via container) and optional configuration.

#     Args:
#         preprocessor: Image preprocessing service (injected via container)
#         app_config: Application configuration (injected via container)
#         dino_processor: DINO feature extractor (injected via container)
#         superpoint_processor: SuperPoint feature extractor (injected via container)
#         lightglue_processor: LightGlue matcher (injected via container)
#         milvus_repo: Milvus repository (injected via container)
#         postgres_repo: PostgreSQL repository (injected via container)
#         minio_repo: MinIO repository (injected via container)
#         matching_strategy: Strategy for decision making (optional)
#         top_k: Number of candidates to retrieve
#         coarse_threshold: Minimum similarity for coarse retrieval
#         inlier_threshold: Minimum inliers for match decision
#         verbose: Enable verbose logging
#         use_hierarchical: Use hierarchical matching if available (DEFAULT: True)
#         vector_store_type: "postgres" or "milvus" for vector storage
#         **kwargs: Additional arguments

#     Returns:
#         Configured VerificationService

#     Example:
#         # Using container DI:
#         service = create_verification_service(
#             preprocessor=container.preprocessor_service(),
#             app_config=container.app_config(),
#             dino_processor=container.dino_processor(),
#             ...
#         )
        
#         # Using with explicit overrides:
#         service = create_verification_service(
#             postgres_repo=my_repo,
#             minio_repo=my_minio,
#             use_hierarchical=True,
#             verbose=True
#         )
#     """
#     return VerificationService(
#         preprocessor=preprocessor,
#         app_config=app_config,
#         dino_processor=dino_processor,
#         superpoint_processor=superpoint_processor,
#         lightglue_processor=lightglue_processor,
#         milvus_repo=milvus_repo,
#         postgres_repo=postgres_repo,
#         minio_repo=minio_repo,
#         matching_strategy=matching_strategy,
#         top_k=top_k,
#         coarse_threshold=coarse_threshold,
#         inlier_threshold=inlier_threshold,
#         verbose=verbose,
#         use_hierarchical=use_hierarchical,
#         vector_store_type=vector_store_type,
#         **kwargs
#     )


# def create_verification_pipeline(
#     config_dict: Optional[Dict[str, Any]] = None,
#     app_config: Optional[AppConfig] = None,
#     verbose: bool = False,
# ) -> "VerificationService":
#     """Create verification service for 1-1 verification.

#     This factory function creates and returns a VerificationService instance
#     configured with all necessary processors and repositories.

#     Args:
#         config_dict: Optional configuration dictionary
#         app_config: Optional AppConfig (will use get_config() if not provided)
#         verbose: Enable verbose logging

#     Returns:
#         VerificationService configured for both verification and identification

#     Note:
#         This function was moved from verificationPipeline.py after the pipeline
#         layer was simplified. The service is now created directly.
#     """
#     from src.config.appConfig import AppConfig, get_config
#     from src.processor.dinoProcessor import DinoConfig
#     from src.processor.superPointProcessor import SuperPointConfig
#     from src.processor.lightGlueProcessor import LightGlueConfig
#     from src.repository.milvusRepository import MilvusConfig
#     from src.repository.minioRepository import MinIOConfig
#     from src.utils.matchingStrategy import MatchingStrategy, MatchingStrategyConfig

#     # Get global config if not provided
#     if app_config is None:
#         app_config: AppConfig = get_config()

#     # Build config_dict from appConfig if not provided
#     if config_dict is None:
#         config_dict = {
#             "dino_model_type": app_config.dino_model_type,
#             "dino_device": app_config.dino_device,
#             "dino_use_multi_gpu": app_config.dino_use_multi_gpu,
#             "dino_gpu_ids": app_config.dino_gpu_ids,
#             "dino_enable_memory_optimization": app_config.dino_enable_memory_optimization,
#             "dino_image_size": app_config.dino_image_size,
#             "hf_token": app_config.dino_hf_token,
#             "sp_max_keypoints": app_config.sp_max_keypoints,
#             "sp_max_dimension": app_config.sp_max_dimension,
#             "sp_device": app_config.sp_device,
#             "lg_device": app_config.lg_device,
#             "lg_filter_threshold": app_config.lg_filter_threshold,
#             "lg_confidence": app_config.lg_confidence,
#             "milvus_uri": app_config.milvus_uri,
#             "milvus_collection": app_config.milvus_collection,
#             "milvus_vector_dim": app_config.milvus_vector_dim,
#             "milvus_search_top_k": app_config.milvus_search_top_k,
#             "milvus_search_nprobe": app_config.milvus_search_nprobe,
#             "minio_endpoint": app_config.minio_endpoint,
#             "minio_bucket": app_config.minio_bucket,
#             "top_k": app_config.milvus_top_k,
#             "coarse_threshold": app_config.coarse_threshold,
#             "inlier_threshold": app_config.inlier_threshold,
#             "dino_threshold": 0.5,  # FIXED: Was 0.2 - raised to 0.5 for stricter filtering
#             "verbose": verbose,
#         }

#     # Initialize preprocessor
#     preprocessor = PreprocessorService(
#         background_color="black",
#         sam3_model_path=None
#     )

#     dino_config = DinoConfig(
#         model_type=config_dict.get("dino_model_type", app_config.dino_model_type),
#         device=config_dict.get("dino_device", app_config.dino_device),
#         hf_token=config_dict.get("hf_token", app_config.dino_hf_token),
#         verbose=config_dict.get("verbose", False),
#         use_multi_gpu=config_dict.get("dino_use_multi_gpu", app_config.dino_use_multi_gpu),
#         gpu_ids=config_dict.get("dino_gpu_ids", app_config.dino_gpu_ids),
#         enable_memory_optimization=config_dict.get("dino_enable_memory_optimization", app_config.dino_enable_memory_optimization),
#     )
#     dino: DinoProcessor = DinoProcessor(dino_config)

#     superpoint_config = SuperPointConfig(
#         device=config_dict.get("sp_device", app_config.sp_device),
#         max_keypoints=config_dict.get("sp_max_keypoints", app_config.sp_max_keypoints),
#         max_dimension=config_dict.get("sp_max_dimension", app_config.sp_max_dimension),
#         verbose=config_dict.get("verbose", False)
#     )
#     superpoint: SuperPointProcessor = SuperPointProcessor(superpoint_config)

#     lightglue_config = LightGlueConfig(
#         device=config_dict.get("lg_device", app_config.lg_device),
#         filter_threshold=config_dict.get("lg_filter_threshold", app_config.lg_filter_threshold),
#         verbose=config_dict.get("verbose", False)
#     )
#     lightglue: LightGlueProcessor = LightGlueProcessor(lightglue_config)

#     # Initialize Milvus repository
#     milvus_config = MilvusConfig(
#         uri=config_dict.get("milvus_uri", app_config.milvus_uri),
#         collection_name=config_dict.get("milvus_collection", app_config.milvus_collection),
#         vector_dim=config_dict.get("milvus_vector_dim", app_config.milvus_vector_dim),
#         nprobe=config_dict.get("milvus_search_nprobe", app_config.milvus_search_nprobe),
#         metric_type="COSINE",
#         index_type="IVF_FLAT",
#         verbose=config_dict.get("verbose", False),
#     )
#     # Initialize Milvus only if vector_store_type is "milvus"
#     milvus: Optional[MilvusRepository] = None
#     if app_config.vector_store_type == "milvus":
#         milvus = MilvusRepository(config=milvus_config, app_config=app_config)
#     else:
#         logger.info("Milvus not initialized (using PostgreSQL for vector storage)")

#     # Initialize SQLAlchemy ORM repository (for pgvector search)
#     postgres: Optional[SQLAlchemyORMRepository] = None
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
#             logger.info(f"Verification service connected to PostgreSQL via SQLAlchemy: {app_config.postgres_host}:{app_config.postgres_port}")
#         else:
#             logger.warning("PostgreSQL connection failed, falling back to Milvus")
#             postgres = None
#     except Exception as e:
#         logger.warning(f"SQLAlchemy repository creation failed: {e}")
#         postgres = None

#     # Initialize MinIO repository
#     minio_config: MinIOConfig = MinIOConfig(
#         endpoint=config_dict.get("minio_endpoint", app_config.minio_endpoint),
#         bucket=config_dict.get("minio_bucket", app_config.minio_bucket),
#         verbose=config_dict.get("verbose", False),
#     )
#     minio: Optional[MinIORepository] = None
#     try:
#         minio = MinIORepository(minio_config)
#         if verbose:
#             logger.info("[create_verification_pipeline] MinIO initialized successfully")
#     except Exception as e:
#         logger.warning(f"[create_verification_pipeline] MinIO initialization failed (non-critical): {e}")
#         logger.warning("[create_verification_pipeline] Continuing without bark texture support - using Milvus only")
#         minio = None

#     # Initialize matching strategy
#     matching_config: MatchingStrategyConfig = MatchingStrategyConfig(
#         inlier_threshold=config_dict.get("inlier_threshold", app_config.inlier_threshold),
#         coarse_similarity_threshold=config_dict.get("coarse_threshold", app_config.coarse_threshold),
#         top_k_coarse=config_dict.get("top_k", app_config.milvus_top_k),
#         min_inlier_ratio=0.01
#     )
#     matching_strategy: MatchingStrategy = MatchingStrategy(matching_config)

#     # Get vector store type from config
#     vector_store_type = app_config.vector_store_type

#     # Create verification service
#     service: VerificationService = VerificationService(
#         preprocessor=preprocessor,
#         dino_processor=dino,
#         superpoint_processor=superpoint,
#         lightglue_processor=lightglue,
#         milvus_repo=milvus,
#         postgres_repo=postgres,
#         matching_strategy=matching_strategy,
#         minio_repo=minio,
#         verbose=config_dict.get("verbose", True),
#         dino_threshold=config_dict.get("dino_threshold", 0.5),
#         vector_store_type=vector_store_type,
#     )

#     return service
