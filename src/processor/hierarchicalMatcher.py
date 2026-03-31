"""
Hierarchical Feature Matching with Geometric Verification

This module implements an advanced hierarchical matching strategy that combines:
1. DINO global features for fast candidate retrieval
2. SuperPoint local features for precise matching
3. LightGlue matcher for robust feature matching
4. Geometric verification with RANSAC

This hybrid approach provides the best balance of accuracy and speed.
"""
import time
import numpy as np
import cv2
import logging
from typing import Any, List, Dict, Tuple, Optional, TypedDict
from dataclasses import dataclass, field
import torch

# Try to import lightglue, but allow running without it for testing
LightGlue = None
SuperPoint_LG = None
try:
    from lightglue import LightGlue, SuperPoint as SuperPoint_LG
except ImportError:
    LightGlue = None
    SuperPoint_LG = None
    logger = logging.getLogger(__name__)
    logger.warning("LightGlue not installed. Hierarchical matching will use fallback mode.")

logger = logging.getLogger(__name__)


# Import dependency_injector for DI support
from dependency_injector.wiring import inject, Provide

from src.utils.similarityUtils import (
    cosine_similarity,
    cosine_similarity_batch,
    cosine_similarity_batch_auto,
)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class HierarchicalMatcherConfig:
    """Configuration for HierarchicalMatcher.
    
    Default values are sourced from appConfig to maintain consistency.
    Use dependency injection with HierarchicalMatcherConfig for proper DI.
    
    Note: For testing, you can override these defaults directly in __init__
    or use the dataclass factory pattern with get_config().
    """
    # Matching thresholds
    dino_threshold: float = 0.2  # Very permissive - SuperPoint/RANSAC will filter
    superpoint_match_ratio: float = 0.1  # Minimum match ratio
    ransac_threshold: float = 5.0  # Pixels
    
    # Matching criteria
    min_inliers: int = 7  # From appConfig.min_inliers
    min_matches: int = 15  # From appConfig.min_matches
    use_fundamental: bool = False  # Use fundamental matrix instead of homography
    
    # SuperPoint configuration - align with appConfig
    sp_max_keypoints: int = 4096  # From appConfig.sp_max_keypoints
    
    # LightGlue configuration - align with appConfig
    lg_device: str = "cuda"  # From appConfig.lg_device
    lg_confidence: float = 0.1  # From appConfig.lg_confidence
    
    # CUDA Streams configuration for parallel processing
    stream_count: int = 8  # Number of CUDA streams for parallel matching (optimal for RTX)
    enable_cuda_streams: bool = True  # Enable CUDA streams for parallel processing
    
    # Default weights for scoring
    # DINO provides the coarse ranking (0.65), but LightGlue match count
    # (0.15) and RANSAC inlier ratio (0.20) together get 35% influence.
    weights: Dict[str, float] = field(default_factory=lambda: {
        'dino_similarity': 0.65,
        'superpoint_match_ratio': 0.15,
        'inlier_ratio': 0.20
    })
    
    @classmethod
    def from_app_config(cls, app_config) -> 'HierarchicalMatcherConfig':
        """Create config from AppConfig instance.
        
        Args:
            app_config: AppConfig instance with settings
            
        Returns:
            HierarchicalMatcherConfig populated from app_config
        """
        return cls(
            min_inliers=app_config.min_inliers,
            min_matches=app_config.min_matches,
            sp_max_keypoints=app_config.sp_max_keypoints,
            lg_device=app_config.lg_device,
            lg_confidence=app_config.lg_confidence,
        )


class FeatureVector(TypedDict, total=False):
    """Type definition for feature vector (used for both query and candidate).

    Attributes:
        id: Image identifier
        tree_id: Tree identifier
        dino: DINO global descriptor (1D numpy array)
        keypoints: SuperPoint keypoints (Nx2 numpy array)
        descriptors: SuperPoint descriptors (NxD numpy array)
        scores: Optional SuperPoint keypoint scores (1D numpy array)
        texture_histogram: Optional bark texture histogram (1D numpy array)
        image_size: Optional (W, H) of the original image for LightGlue coordinate normalization
    """
    id: str
    tree_id: str
    dino: np.ndarray
    keypoints: Optional[np.ndarray]
    descriptors: Optional[np.ndarray]
    scores: Optional[np.ndarray]
    texture_histogram: Optional[np.ndarray]
    image_size: Optional[Tuple[int, int]]  # (W, H) of the original image


# Alias for backward compatibility
QueryFeatures = FeatureVector
CandidateFeatures = FeatureVector


@dataclass
class MatchingResult:
    """Result of hierarchical matching process.

    This dataclass stores the complete matching pipeline results across three stages:
    1. Stage 1 (DINO): Global feature similarity for candidate retrieval
    2. Stage 2 (SuperPoint + LightGlue): Local feature matching
    3. Stage 3 (RANSAC): Geometric verification and outlier rejection
    """

    # === Identifiers ===
    query_id: str
    candidate_id: str
    tree_id: str = ""  # Tree ID for the candidate image

    # === Stage 1: DINO Global Features ===
    dino_similarity: float = 0.0  # Cosine similarity [0, 1] from DINO descriptors

    # === Stage 2: SuperPoint Local Features + LightGlue Matching ===
    # LightGlue produces matches between SuperPoint keypoints
    superpoint_matches: int = 0  # Total number of matches from LightGlue (before RANSAC filtering)
    superpoint_match_ratio: float = 0.0  # Match ratio = superpoint_matches / min(query_kp, candidate_kp)

    # === Stage 3: RANSAC Geometric Verification ===
    # RANSAC filters LightGlue matches to keep only geometrically consistent ones
    superpoint_inliers: int = 0  # Number of inliers after RANSAC filtering (subset of superpoint_matches)
    homography: Optional[np.ndarray] = None  # 3x3 homography matrix from RANSAC (planar scenes)
    fundamental: Optional[np.ndarray] = None  # 3x3 fundamental matrix from RANSAC (general scenes)
    reprojection_error: float = float('inf')  # Average reprojection error of inliers (pixels) - measures RANSAC accuracy
    ransac_inlier_ratio: float = 0.0  # Inlier ratio = superpoint_inliers / superpoint_matches (RANSAC filtering quality)

    # === Bark Texture Matching ===
    # Complementary matching using HSV histogram comparison
    bark_texture_similarity: float = 0.0  # Cosine similarity [0, 1] of HSV histograms

    # === Final Score ===
    # Weighted combination of all three matching stages
    final_score: float = 0.0  # Combined confidence score [0, 1] = w1*dino + w2*match_ratio + w3*inlier_ratio

    # === Timing Information ===
    dino_search_time: float = 0.0  # Time for Stage 1 (DINO similarity search)
    superpoint_match_time: float = 0.0  # Time for Stage 2 (LightGlue matching)
    geometric_verify_time: float = 0.0  # Time for Stage 3 (RANSAC verification)
    texture_match_time: float = 0.0  # Time for bark texture comparison
    total_time: float = 0.0  # Total pipeline time

    # === Visualization Data ===
    match_coordinates: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None  # Matched keypoint pairs for visualization

    # === Timestamp ===
    captured_at: Optional[int] = None  # Unix epoch timestamp when the image was captured

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        import math

        # Handle inf/nan values for JSON serialization
        reprojection_error = self.reprojection_error
        if reprojection_error is None or (isinstance(reprojection_error, float) and (math.isinf(reprojection_error) or math.isnan(reprojection_error))):
            reprojection_error = 999.99  # Use a large but finite value

        # Ensure final_score is valid
        final_score = self.final_score
        if isinstance(final_score, float) and (math.isinf(final_score) or math.isnan(final_score)):
            final_score = 0.0

        return {
            'query_id': self.query_id,
            'candidate_id': self.candidate_id,
            'tree_id': self.tree_id,
            'dino_similarity': float(self.dino_similarity),
            'superpoint_matches': int(self.superpoint_matches),
            'superpoint_inliers': int(self.superpoint_inliers),
            'superpoint_match_ratio': float(self.superpoint_match_ratio),
            'ransac_inlier_ratio': float(self.ransac_inlier_ratio),
            'bark_texture_similarity': float(self.bark_texture_similarity),
            'homography': self.homography.tolist() if self.homography is not None else None,
            'fundamental': self.fundamental.tolist() if self.fundamental is not None else None,
            'reprojection_error': float(reprojection_error),
            'final_score': float(final_score),
            'timing': {
                'dino_search': float(self.dino_search_time),
                'superpoint_match': float(self.superpoint_match_time),
                'geometric_verify': float(self.geometric_verify_time),
                'texture_match': float(self.texture_match_time),
                'total': float(self.total_time)
            },
            'captured_at': self.captured_at
        }


class NeighborGroup:
    """
    Represents a candidate image together with its neighboring nodes.

    This class implements the neighbor-based scoring hypothesis:
    If a candidate image has high cosine similarity with a query image,
    then the neighbors of that candidate should also have good similarity
    with the query image.
    """

    def __init__(
        self,
        candidate_id: str,
        tree_id: str,
        candidate_dino: np.ndarray,
        neighbor_ids: List[str],
        neighbor_dinos: List[np.ndarray],
        neighbor_sim_to_candidate: List[float],
    ):
        """
        Initialize a neighbor group.

        Args:
            candidate_id: ID of the central candidate image
            tree_id: Tree ID for all images in the group
            candidate_dino: DINO descriptor of the candidate
            neighbor_ids: List of neighbor image IDs
            neighbor_dinos: List of neighbor DINO descriptors
            neighbor_sim_to_candidate: Similarity of each neighbor to the candidate
        """
        self.candidate_id = candidate_id
        self.tree_id = tree_id
        self.candidate_dino = candidate_dino
        self.neighbor_ids = neighbor_ids
        self.neighbor_dinos = neighbor_dinos
        self.neighbor_sim_to_candidate = neighbor_sim_to_candidate

    def compute_enhanced_similarity(
        self,
        query_dino: np.ndarray,
        query_weight: float = 0.6,
        neighbor_weight: float = 0.4
    ) -> float:
        """
        Compute enhanced similarity score considering neighbors.

        Score = query_weight * sim(query, candidate) +
                neighbor_weight * avg(sim(query, neighbors))

        Args:
            query_dino: DINO descriptor of query image
            query_weight: Weight for direct query-candidate similarity
            neighbor_weight: Weight for average query-neighbor similarity

        Returns:
            Enhanced similarity score [0, 1]
        """
        # Direct query-candidate similarity — clamped to [0, 1] so a negative raw
        # cosine value does not distort the weighted combination below.
        direct_sim = max(0.0, cosine_similarity(query_dino, self.candidate_dino))

        # Average query-neighbor similarity — each term clamped individually so
        # a single very-negative neighbor cannot drag down an otherwise good mean.
        if self.neighbor_dinos:
            neighbor_sims = [
                max(0.0, cosine_similarity(query_dino, n_dino))
                for n_dino in self.neighbor_dinos
            ]
            avg_neighbor_sim = float(np.mean(neighbor_sims))
        else:
            avg_neighbor_sim = 0.0

        # Combined score — final clamp is a safety net for floating-point noise.
        enhanced_score = query_weight * direct_sim + neighbor_weight * avg_neighbor_sim
        return min(1.0, enhanced_score)

class HierarchicalMatcher:
    """
    Hierarchical feature matcher combining DINO and SuperPoint.

    Matching Strategy:
    1. Stage 1 - Global Search: DINO cosine similarity for fast candidate filtering
    2. Stage 2 - Local Matching: SuperPoint descriptor matching with LightGlue
    3. Stage 3 - Geometric Verification: RANSAC for outlier rejection

    This approach is inspired by state-of-the-art methods:
    - Image Retrieval: DINOv3 for global descriptors
    - Feature Matching: LightGlue for sparse feature matching
    - Geometric Verification: RANSAC for robust estimation

    Neighbor-Enhanced Scoring:
    - Candidates are evaluated with their neighbors for more robust scoring
    - If a candidate has high similarity, its neighbors should also have good similarity
    - This approach replaces diversity-based selection with neighbor-group selection
    
    Uses dependency injection for configuration. Wire with:
        container.wire(modules=["src.processor.hierarchicalMatcher"])
    
    Or create with AppConfig:
        config = HierarchicalMatcherConfig.from_app_config(app_config)
        matcher = HierarchicalMatcher(config=config)
    """
    
    @inject
    def __init__(
        self,
        config: HierarchicalMatcherConfig = Provide["hierarchical_matcher_config"],
    ) -> None:
        """
        Initialize hierarchical matcher with dependency injection.

        Args:
            config: HierarchicalMatcherConfig instance with all matching parameters.
                   Defaults to appConfig values when injected via Provide.
                   
        DI Provider:
            Provide["hierarchical_matcher_config"] - HierarchicalMatcherConfig
        """
        # Extract config values
        self.dino_threshold = config.dino_threshold
        self.superpoint_match_ratio = config.superpoint_match_ratio
        self.ransac_threshold = config.ransac_threshold
        self.min_inliers = config.min_inliers
        self.min_matches = config.min_matches
        self.use_fundamental = config.use_fundamental
        self.sp_max_keypoints = config.sp_max_keypoints
        self.lg_device = config.lg_device
        self.lg_confidence = config.lg_confidence
        self.stream_count = config.stream_count
        self.enable_cuda_streams = config.enable_cuda_streams
        self.weights = config.weights

        # Performance metrics
        self.cuda_streams_time = 0.0
        self.sequential_time = 0.0
        self.total_matches_with_streams = 0
        self.total_matches_sequential = 0

        # LightGlue is required - try multiple import paths
        self.LightGlue = LightGlue
        self.SuperPoint = SuperPoint_LG

        # Deferred model initialization — loaded on first match() call
        self.lg_matcher = None
        self.sp_extractor = None
        self._models_loaded = False

        # Initialize CUDA streams for parallel processing (works for both NVIDIA and AMD GPUs)
        self.cuda_streams = []
        if self.enable_cuda_streams and self.lg_device == "cuda" and torch.cuda.is_available():
            for _ in range(self.stream_count):
                stream = torch.cuda.Stream()
                self.cuda_streams.append(stream)
            logger.info(f"[HierarchicalMatcher] CUDA streams enabled: {self.stream_count} streams")

    def _ensure_models_loaded(self) -> None:
        """Lazy-load LightGlue matcher and SuperPoint extractor on first use."""
        if self._models_loaded:
            return

        # Initialize LightGlue matcher with configurable confidence
        self.lg_matcher = self.LightGlue(features='superpoint', confidence=self.lg_confidence)
        if self.lg_device == "cuda" and torch.cuda.is_available():
            self.lg_matcher = self.lg_matcher.cuda()

        # Initialize SuperPoint extractor with configurable max keypoints
        self.sp_extractor = self.SuperPoint(max_num_keypoints=self.sp_max_keypoints)
        if self.lg_device == "cuda" and torch.cuda.is_available():
            self.sp_extractor = self.sp_extractor.cuda()

        self._models_loaded = True
        print(f"✅ LightGlue matcher initialized (device={self.lg_device}, confidence={self.lg_confidence})")
        print(f"✅ SuperPoint extractor initialized (max_keypoints={self.sp_max_keypoints})")

    def search_candidates(
        self,
        query_dino: np.ndarray,
        candidate_dinos: List[np.ndarray],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Stage 1: Global search using DINO features with GPU-accelerated batch processing.

        Uses GPU-accelerated cosine similarity for fast candidate retrieval.
        Falls back to CPU if GPU is unavailable.

        Args:
            query_dino: Query DINO descriptor (1024-dim)
            candidate_dinos: List of candidate DINO descriptors
            top_k: Number of top candidates to return

        Returns:
            List of (candidate_idx, similarity) tuples
        """
        start_time = time.time()

        if len(candidate_dinos) == 0:
            logger.warning("[DINO Search] No candidates to search")
            return []

        # Convert to numpy arrays for batch processing
        candidate_vecs = np.array(candidate_dinos)  # (N, 1024)

        # GPU-accelerated batch cosine similarity with automatic CPU fallback
        similarities_np = cosine_similarity_batch_auto(
            np.array(query_dino), candidate_vecs, device=self.lg_device
        )

        # Filter by threshold
        valid_mask = similarities_np >= self.dino_threshold
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            logger.warning(f"[DINO Search] No candidates passed threshold {self.dino_threshold}")
            return []

        # Get top-k
        top_indices = np.argsort(similarities_np[valid_indices])[-top_k:][::-1]
        results = [(int(valid_indices[i]), float(similarities_np[valid_indices[i]])) for i in top_indices]

        elapsed = time.time() - start_time
        self._dino_search_time = elapsed

        logger.info(f"[DINO Search] Found {len(results)} candidates in {elapsed:.3f}s (GPU: {torch.cuda.is_available() and self.lg_device == 'cuda'})")
        if results:
            logger.info(f"  Top similarity: {results[0][1]:.4f}")
            logger.info(f"  Bottom similarity: {results[-1][1]:.4f}")

        return results

    def find_neighbors(
        self,
        target_dino: np.ndarray,
        candidates: List[CandidateFeatures],
        tree_id: str,
        top_k: int = 3,
        exclude_target_id: Optional[str] = None,
        similarity_threshold: float = 0.89
    ) -> List[Tuple[CandidateFeatures, float]]:
        """
        Find neighbors of a target DINO descriptor within a specific tree.

        Neighbors are defined as images with cosine similarity > threshold to the target
        within the same tree. Returns up to top_k neighbors sorted by similarity.

        Args:
            target_dino: Target DINO descriptor to find neighbors for
            candidates: List of all candidate features
            tree_id: Tree ID to restrict search
            top_k: Maximum number of neighbors to return
            exclude_target_id: ID to exclude from results (e.g., the target itself)
            similarity_threshold: Minimum cosine similarity to be considered a neighbor (default: 0.85)

        Returns:
            List of (candidate, similarity) tuples sorted by similarity (descending)
        """
        # Filter by tree_id and exclusion
        tree_candidates = [
            c for c in candidates
            if c.get('tree_id') == tree_id and
            (exclude_target_id is None or c.get('id') != exclude_target_id)
        ]

        if not tree_candidates:
            return []

        # Compute similarities in batch for efficiency
        candidate_vecs = np.array([c['dino'] for c in tree_candidates])
        similarities = cosine_similarity_batch(np.array(target_dino), candidate_vecs)

        # Filter by threshold and sort by similarity
        valid_indices = np.where(similarities >= similarity_threshold)[0]

        # Handle case when no candidates pass the similarity threshold
        if len(valid_indices) == 0:
            # Fall back to lower threshold (0.5) to ensure we return at least some neighbors
            # This handles edge cases where similarity_threshold is too high
            logger.warning(f"[find_neighbors] No candidates passed threshold {similarity_threshold}, using fallback threshold 0.5")
            valid_indices = np.where(similarities >= 0.5)[0]
            if len(valid_indices) == 0:
                # If still no results, return top-k by similarity (no threshold)
                logger.warning(f"[find_neighbors] No candidates passed fallback threshold, returning top-{top_k} by similarity")
                valid_indices = np.argsort(-similarities)[:top_k]

        valid_indices = valid_indices[np.argsort(-similarities[valid_indices])][:top_k]

        results = [
            (tree_candidates[i], float(similarities[i]))
            for i in valid_indices
        ]

        return results

    def build_neighbor_groups(
        self,
        query_dino: np.ndarray,
        candidates: List[CandidateFeatures],
        neighbors_per_candidate: int = 5,
        query_weight: float = 0.6,
        neighbor_weight: float = 0.4
    ) -> List[Tuple[NeighborGroup, float]]:
        """
        Build neighbor groups for candidates and compute enhanced scores.

        For each candidate, finds its neighbors within the same tree and
        computes an enhanced score that considers both the candidate's
        direct similarity and the average similarity of its neighbors.

        Args:
            query_dino: Query DINO descriptor
            candidates: List of candidate features (must have tree_id)
            neighbors_per_candidate: Number of neighbors to find per candidate
            query_weight: Weight for direct query-candidate similarity
            neighbor_weight: Weight for average query-neighbor similarity

        Returns:
            List of (NeighborGroup, enhanced_score) tuples sorted by enhanced_score
        """
        groups = []

        # Group candidates by tree_id
        tree_to_candidates: Dict[str, List[CandidateFeatures]] = {}
        for cand in candidates:
            tree_id = cand.get('tree_id', '')
            if tree_id:
                if tree_id not in tree_to_candidates:
                    tree_to_candidates[tree_id] = []
                tree_to_candidates[tree_id].append(cand)

        # Build neighbor group for each candidate
        for candidate in candidates:
            candidate_id = candidate.get('id', '')
            tree_id = candidate.get('tree_id', '')
            candidate_dino = candidate.get('dino')

            if not tree_id or not candidate_id or candidate_dino is None:
                continue

            # Find neighbors within the same tree
            tree_candidates = tree_to_candidates.get(tree_id, [])
            neighbors = self.find_neighbors(
                target_dino=candidate_dino,
                candidates=tree_candidates,
                tree_id=tree_id,
                top_k=neighbors_per_candidate,
                exclude_target_id=candidate_id
            )

            # Create NeighborGroup
            neighbor_ids = [n.get('id', '') for n, _ in neighbors]
            neighbor_dinos = [n.get('dino') for n, _ in neighbors]
            neighbor_sims = [sim for _, sim in neighbors]

            group = NeighborGroup(
                candidate_id=candidate_id,
                tree_id=tree_id,
                candidate_dino=candidate_dino,
                neighbor_ids=neighbor_ids,
                neighbor_dinos=neighbor_dinos,
                neighbor_sim_to_candidate=neighbor_sims
            )

            # Compute enhanced score
            enhanced_score = group.compute_enhanced_similarity(
                query_dino=query_dino,
                query_weight=query_weight,
                neighbor_weight=neighbor_weight
            )

            groups.append((group, enhanced_score))

        # Sort by enhanced score (descending)
        groups.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"[Neighbor Groups] Built {len(groups)} groups")
        if groups:
            logger.info(f"  Top enhanced score: {groups[0][1]:.4f}")
            logger.info(f"  Bottom enhanced score: {groups[-1][1]:.4f}")

        return groups

    def select_candidate_groups_for_superpoint(
        self,
        neighbor_groups: List[Tuple[NeighborGroup, float]],
        candidates_dict: Dict[str, CandidateFeatures],
        top_k_groups: int = 5,
        max_candidates_per_tree: int = 8
    ) -> List[CandidateFeatures]:
        """
        Select candidates for SuperPoint matching based on neighbor group scores.

        Instead of selecting diverse candidates, this method selects:
        1. Top candidates by enhanced score
        2. Their neighbors (to form groups for robust matching)

        This ensures SuperPoint works on candidate+neighbor groups that have
        the best aggregate similarity to the query.

        Args:
            neighbor_groups: List of (NeighborGroup, enhanced_score) from build_neighbor_groups
            candidates_dict: Dictionary mapping candidate_id to CandidateFeatures
            top_k_groups: Number of top groups to select
            max_candidates_per_tree: Max candidates to select per tree

        Returns:
            List of selected candidates for SuperPoint matching
        """
        selected = []
        seen_ids = set()

        # Track per-tree count
        tree_counts: Dict[str, int] = {}

        # Select top-K groups
        for group, _ in neighbor_groups[:top_k_groups]:
            tree_id = group.tree_id

            # Check per-tree limit
            current_count = tree_counts.get(tree_id, 0)
            if current_count >= max_candidates_per_tree:
                continue

            # Add the candidate itself
            if group.candidate_id not in seen_ids and group.candidate_id in candidates_dict:
                selected.append(candidates_dict[group.candidate_id])
                seen_ids.add(group.candidate_id)
                tree_counts[tree_id] = current_count + 1

            # Add neighbors (if under per-tree limit)
            for neighbor_id in group.neighbor_ids:
                if tree_counts.get(tree_id, 0) >= max_candidates_per_tree:
                    break
                if neighbor_id not in seen_ids and neighbor_id in candidates_dict:
                    selected.append(candidates_dict[neighbor_id])
                    seen_ids.add(neighbor_id)
                    tree_counts[tree_id] = tree_counts.get(tree_id, 0) + 1

        logger.info(f"[Group Selection] Selected {len(selected)} candidates from {top_k_groups} groups")
        logger.info(f"  Unique candidates: {len(seen_ids)}")

        # Log per-tree breakdown
        for tree_id, count in tree_counts.items():
            logger.info(f"  Tree {tree_id}: {count} candidates")

        return selected

    def match_superpoint(
        self,
        query_kp: np.ndarray,
        query_desc: np.ndarray,
        candidate_kp: np.ndarray,
        candidate_desc: np.ndarray,
        query_scores: Optional[np.ndarray] = None,
        candidate_scores: Optional[np.ndarray] = None,
        query_image_size: Optional[Tuple[int, int]] = None,
        candidate_image_size: Optional[Tuple[int, int]] = None,
        stream: Optional[Any] = None,
    ) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Stage 2: Local feature matching with SuperPoint using LightGlue with score awareness.

        LightGlue provides state-of-the-art matching performance with
        adaptive pruning and context-aware feature selection.

        Now includes keypoint quality scores for each match.

        Args:
            query_kp: Query keypoints (N, 2)
            query_desc: Query descriptors (N, 256)
            candidate_kp: Candidate keypoints (M, 2)
            candidate_desc: Candidate descriptors (M, 256)
            query_scores: Query keypoint quality scores (N,) - optional
            candidate_scores: Candidate keypoint quality scores (M,) - optional
            query_image_size: Real image (W, H) for correct LightGlue coordinate normalization - optional
            candidate_image_size: Real image (W, H) for correct LightGlue coordinate normalization - optional

        Returns:
            Tuple of (matches, query_kp_matched, candidate_kp_matched, q_scores_matched, c_scores_matched, elapsed_time)
        """
        # Validate inputs
        is_valid, error_msg = self._validate_match_inputs(
            query_kp, query_desc, candidate_kp, candidate_desc
        )
        if not is_valid:
            logger.warning(f"[match_superpoint] Validation failed: {error_msg}")
            return [], np.array([]), np.array([]), np.array([]), np.array([]), 0.0

        # Use LightGlue with score-aware matching
        # import time
        # start_time = time.time()

        matches, q_kp, c_kp, q_scores, c_scores, lg_elapsed = self._match_with_lightglue(
            query_kp, query_desc, candidate_kp, candidate_desc,
            query_scores=query_scores, candidate_scores=candidate_scores,
            query_image_size=query_image_size, candidate_image_size=candidate_image_size,
            stream=stream,
        )

        return matches, q_kp, c_kp, q_scores, c_scores, lg_elapsed

    def _match_with_lightglue(
        self,
        query_kp: np.ndarray,
        query_desc: np.ndarray,
        candidate_kp: np.ndarray,
        candidate_desc: np.ndarray,
        query_scores: Optional[np.ndarray] = None,
        candidate_scores: Optional[np.ndarray] = None,
        stream: Optional[torch.cuda.Stream] = None,
        query_image_size: Optional[Tuple[int, int]] = None,
        candidate_image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Match features using LightGlue with score-aware matching and optional CUDA stream.

        Returns matched keypoints WITH quality scores for both query and candidate.

        LightGlue advantages:
        - Adaptive pruning: Early rejection of unlikely matches
        - Context-aware: Uses spatial context to improve matching
        - End-to-end: Differentiable matching for learning

        Args:
            query_kp: Query keypoints (N, 2)
            query_desc: Query descriptors (N, 256)
            candidate_kp: Candidate keypoints (M, 2)
            candidate_desc: Candidate descriptors (M, 256)
            query_scores: Query keypoint scores (N,) - optional
            candidate_scores: Candidate keypoint scores (M,) - optional
            stream: CUDA stream for async execution - optional
            query_image_size: Real image (W, H) for correct LightGlue coordinate normalization - optional
            candidate_image_size: Real image (W, H) for correct LightGlue coordinate normalization - optional

        Returns:
            Tuple of (matches_list, q_kp_matched, c_kp_matched, q_scores_matched, c_scores_matched, elapsed_time)
        """
        self._ensure_models_loaded()

        # Check if keypoints are empty
        if len(query_kp) == 0 or len(candidate_kp) == 0:
            logger.warning(f"Empty keypoints: query_kp={len(query_kp)}, candidate_kp={len(candidate_kp)}")
            return [], np.array([]), np.array([]), np.array([]), np.array([]), 0.0

        # Validate inputs including scores
        is_valid, error_msg = self._validate_lightglue_inputs(
            query_kp, query_desc, candidate_kp, candidate_desc,
            query_scores, candidate_scores
        )
        if not is_valid:
            logger.error(f"[_match_with_lightglue] Validation failed: {error_msg}")
            return [], np.array([]), np.array([]), np.array([]), np.array([]), 0.0

        # Convert to torch tensors
        # LightGlue expects descriptors as (N, D) not (D, N)
        q_kp_tensor = torch.from_numpy(query_kp).unsqueeze(0).float()  # (1, N, 2)
        q_desc_tensor = torch.from_numpy(query_desc).unsqueeze(0).float()  # (1, N, 256)
        c_kp_tensor = torch.from_numpy(candidate_kp).unsqueeze(0).float()  # (1, M, 2)
        c_desc_tensor = torch.from_numpy(candidate_desc).unsqueeze(0).float()  # (1, M, 256)

        # Move tensors to configured device (with optional stream context)
        if self.lg_device == "cuda" and torch.cuda.is_available():
            if stream is not None:
                with torch.cuda.stream(stream):
                    q_kp_tensor = q_kp_tensor.cuda(non_blocking=True)
                    q_desc_tensor = q_desc_tensor.cuda(non_blocking=True)
                    c_kp_tensor = c_kp_tensor.cuda(non_blocking=True)
                    c_desc_tensor = c_desc_tensor.cuda(non_blocking=True)
            else:
                # Sequential processing (no stream)
                q_kp_tensor = q_kp_tensor.cuda()
                q_desc_tensor = q_desc_tensor.cuda()
                c_kp_tensor = c_kp_tensor.cuda()
                c_desc_tensor = c_desc_tensor.cuda()

        # Create feature dictionaries
        # Use real image (W, H) for LightGlue coordinate normalization when available.
        # LightGlue normalizes keypoints to [-1, 1] using image_size, so passing the
        # actual image dimensions instead of keypoint-derived shapes is critical for
        # Build LightGlue feature dicts. Omit image_size when unavailable so
        # LightGlue skips normalization rather than receiving wrong values or None.
        feats0 = {'keypoints': q_kp_tensor, 'descriptors': q_desc_tensor}
        if query_image_size is not None:
            feats0['image_size'] = torch.tensor(
                [[query_image_size[0], query_image_size[1]]], dtype=torch.float32
            )
        feats1 = {'keypoints': c_kp_tensor, 'descriptors': c_desc_tensor}
        if candidate_image_size is not None:
            feats1['image_size'] = torch.tensor(
                [[candidate_image_size[0], candidate_image_size[1]]], dtype=torch.float32
            )

        # Run LightGlue matcher (with optional stream context)
        import time
        start_time = time.time()
        if stream is not None:
            with torch.cuda.stream(stream):
                with torch.no_grad():
                    matches01 = self.lg_matcher({'image0': feats0, 'image1': feats1})
        else:
            with torch.no_grad():
                matches01 = self.lg_matcher({'image0': feats0, 'image1': feats1})
        elapsed = time.time() - start_time

        # Extract matches
        matches = matches01['matches0']
        matches_cpu = matches[0].cpu().numpy()

        # Convert to DMatch format with descriptor distance
        matches_list = []

        for i, j in enumerate(matches_cpu):
            if j != -1:
                # Compute descriptor distance (L2 norm)
                desc_dist = float(np.linalg.norm(query_desc[i] - candidate_desc[j]))
                matches_list.append({
                    'queryIdx': int(i),
                    'trainIdx': int(j),
                    'distance': desc_dist  # Now includes actual descriptor distance!
                })

        # Get matched keypoints
        valid_mask = matches_cpu != -1
        q_kp_matched = query_kp[valid_mask]
        c_kp_matched = candidate_kp[matches_cpu[valid_mask]]

        # Get matched scores
        q_scores_matched = query_scores[valid_mask] if query_scores is not None else np.ones(len(q_kp_matched))
        c_scores_matched = candidate_scores[matches_cpu[valid_mask]] if candidate_scores is not None else np.ones(len(c_kp_matched))

        # Explicit tensor cleanup to prevent GPU memory leaks
        del q_kp_tensor, q_desc_tensor, c_kp_tensor, c_desc_tensor
        del feats0, feats1, matches01, matches, matches_cpu
        if self.lg_device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return matches_list, q_kp_matched, c_kp_matched, q_scores_matched, c_scores_matched, elapsed

    def _validate_lightglue_inputs(
        self,
        query_kp: np.ndarray,
        query_desc: np.ndarray,
        candidate_kp: np.ndarray,
        candidate_desc: np.ndarray,
        query_scores: Optional[np.ndarray] = None,
        candidate_scores: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate inputs for LightGlue matching including scores.

        Comprehensive validation for keypoints, descriptors, and optional scores.
        Returns (is_valid, error_message) tuple.

        Args:
            query_kp: Query keypoints (N, 2)
            query_desc: Query descriptors (N, 256)
            candidate_kp: Candidate keypoints (M, 2)
            candidate_desc: Candidate descriptors (M, 256)
            query_scores: Optional query keypoint scores (N,)
            candidate_scores: Optional candidate keypoint scores (M,)

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        # Validate descriptor dimension (SuperPoint = 256)
        if query_desc.shape[1] != 256:
            return False, f"Invalid query descriptor dimension: {query_desc.shape[1]}, expected 256"

        if candidate_desc.shape[1] != 256:
            return False, f"Invalid candidate descriptor dimension: {candidate_desc.shape[1]}, expected 256"

        # Validate keypoints and descriptors have same number of features
        if query_kp.shape[0] != query_desc.shape[0]:
            return False, f"Query keypoints/descriptors count mismatch: {query_kp.shape[0]} vs {query_desc.shape[0]}"

        if candidate_kp.shape[0] != candidate_desc.shape[0]:
            return False, f"Candidate keypoints/descriptors count mismatch: {candidate_kp.shape[0]} vs {candidate_desc.shape[0]}"

        # Validate scores if provided
        if query_scores is not None and query_scores.shape[0] != query_kp.shape[0]:
            return False, f"Query scores/keypoints count mismatch: {query_scores.shape[0]} vs {query_kp.shape[0]}"

        if candidate_scores is not None and candidate_scores.shape[0] != candidate_kp.shape[0]:
            return False, f"Candidate scores/keypoints count mismatch: {candidate_scores.shape[0]} vs {candidate_kp.shape[0]}"

        # Check for NaN/Inf in descriptors
        if np.isnan(query_desc).any() or np.isinf(query_desc).any():
            return False, "Query descriptors contain NaN or Inf values"

        if np.isnan(candidate_desc).any() or np.isinf(candidate_desc).any():
            return False, "Candidate descriptors contain NaN or Inf values"

        return True, None

    def _validate_match_inputs(
        self,
        query_kp: np.ndarray,
        query_desc: np.ndarray,
        candidate_kp: np.ndarray,
        candidate_desc: np.ndarray
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate inputs for matching functions.

        Performs comprehensive validation of keypoints and descriptors.
        Returns (is_valid, error_message) tuple.

        Args:
            query_kp: Query keypoints (N, 2)
            query_desc: Query descriptors (N, D)
            candidate_kp: Candidate keypoints (M, 2)
            candidate_desc: Candidate descriptors (M, D)

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        # Validate numpy array types
        if not isinstance(query_kp, np.ndarray):
            return False, f"Query keypoints must be numpy array, got {type(query_kp)}"
        if not isinstance(query_desc, np.ndarray):
            return False, f"Query descriptors must be numpy array, got {type(query_desc)}"
        if not isinstance(candidate_kp, np.ndarray):
            return False, f"Candidate keypoints must be numpy array, got {type(candidate_kp)}"
        if not isinstance(candidate_desc, np.ndarray):
            return False, f"Candidate descriptors must be numpy array, got {type(candidate_desc)}"

        # Validate arrays are not empty
        if query_kp.size == 0:
            return False, "Query keypoints are empty"
        if query_desc.size == 0:
            return False, "Query descriptors are empty"
        if candidate_kp.size == 0:
            return False, "Candidate keypoints are empty"
        if candidate_desc.size == 0:
            return False, "Candidate descriptors are empty"

        # Validate array dimensions
        if query_kp.ndim != 2:
            return False, f"Query keypoints must be 2D array (N, 2), got shape {query_kp.shape}"
        if candidate_kp.ndim != 2:
            return False, f"Candidate keypoints must be 2D array (M, 2), got shape {candidate_kp.shape}"
        if query_desc.ndim != 2:
            return False, f"Query descriptors must be 2D array (N, D), got shape {query_desc.shape}"
        if candidate_desc.ndim != 2:
            return False, f"Candidate descriptors must be 2D array (M, D), got shape {candidate_desc.shape}"

        # Validate keypoints have 2 coordinates (x, y)
        if query_kp.shape[1] != 2:
            return False, f"Query keypoints must have shape (N, 2), got {query_kp.shape}"
        if candidate_kp.shape[1] != 2:
            return False, f"Candidate keypoints must have shape (M, 2), got {candidate_kp.shape}"

        # Validate keypoints and descriptors have matching counts
        if query_kp.shape[0] != query_desc.shape[0]:
            return False, f"Query keypoints/descriptors count mismatch: {query_kp.shape[0]} vs {query_desc.shape[0]}"
        if candidate_kp.shape[0] != candidate_desc.shape[0]:
            return False, f"Candidate keypoints/descriptors count mismatch: {candidate_kp.shape[0]} vs {candidate_desc.shape[0]}"

        # Validate descriptor dimension (SuperPoint = 256)
        if query_desc.shape[1] != 256:
            return False, f"Query descriptors must be 256-dimensional (SuperPoint), got {query_desc.shape[1]}"
        if candidate_desc.shape[1] != 256:
            return False, f"Candidate descriptors must be 256-dimensional (SuperPoint), got {candidate_desc.shape[1]}"

        # Validate for NaN/Inf values
        if np.isnan(query_kp).any() or np.isinf(query_kp).any():
            return False, "Query keypoints contain NaN or Inf values"
        if np.isnan(candidate_kp).any() or np.isinf(candidate_kp).any():
            return False, "Candidate keypoints contain NaN or Inf values"
        if np.isnan(query_desc).any() or np.isinf(query_desc).any():
            return False, "Query descriptors contain NaN or Inf values"
        if np.isnan(candidate_desc).any() or np.isinf(candidate_desc).any():
            return False, "Candidate descriptors contain NaN or Inf values"

        return True, None

    def _validate_match_pipeline_inputs(
        self,
        query: QueryFeatures,
        candidates: List[CandidateFeatures]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate inputs for the match() pipeline.

        Performs comprehensive validation of query and candidate feature dictionaries.
        Returns (is_valid, error_message) tuple.

        Args:
            query: Query feature dict with dino, keypoints, descriptors, scores
            candidates: List of candidate feature dicts (same structure)

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        # Validate query is a dict
        if not isinstance(query, dict):
            return False, f"Query must be a dict, got {type(query)}"

        # Validate candidates is a list
        if not isinstance(candidates, list):
            return False, f"Candidates must be a list, got {type(candidates)}"

        # Validate candidates is not empty
        if len(candidates) == 0:
            return False, "Candidates list is empty"

        # Validate required query fields
        required_query_fields = ['dino', 'keypoints', 'descriptors']
        for field in required_query_fields:
            if field not in query:
                return False, f"Query missing required field: {field}"

        # Validate query DINO descriptor
        query_dino = query['dino']
        if not isinstance(query_dino, np.ndarray):
            return False, f"Query DINO must be numpy array, got {type(query_dino)}"
        if query_dino.size == 0:
            return False, "Query DINO descriptor is empty"
        if query_dino.ndim != 1:
            return False, f"Query DINO must be 1D array, got shape {query_dino.shape}"
        # Get expected dimension from config (384 for dinov3-vitb16)
        from src.config.appConfig import get_config
        expected_dims = (get_config().milvus_vector_dim,)
        if query_dino.shape[0] not in expected_dims:
            return False, f"Query DINO must be {expected_dims[0]}-dimensional, got {query_dino.shape[0]}"
        if np.isnan(query_dino).any() or np.isinf(query_dino).any():
            return False, "Query DINO descriptor contains NaN or Inf values"

        # Validate query keypoints and descriptors
        query_kp = query['keypoints']
        query_desc = query['descriptors']

        is_valid, error_msg = self._validate_match_inputs(query_kp, query_desc, query_kp, query_desc)
        if not is_valid:
            return False, f"Query validation failed: {error_msg}"

        # Validate optional query scores
        if 'scores' in query and query['scores'] is not None:
            query_scores = query['scores']
            if not isinstance(query_scores, np.ndarray):
                return False, f"Query scores must be numpy array, got {type(query_scores)}"
            if query_scores.shape[0] != query_kp.shape[0]:
                return False, f"Query scores/keypoints count mismatch: {query_scores.shape[0]} vs {query_kp.shape[0]}"
            if np.isnan(query_scores).any() or np.isinf(query_scores).any():
                return False, "Query scores contain NaN or Inf values"

        # Validate each candidate
        for i, candidate in enumerate(candidates):
            # Validate candidate is a dict
            if not isinstance(candidate, dict):
                return False, f"Candidate {i} must be a dict, got {type(candidate)}"

            # Validate required candidate fields
            for field in required_query_fields:
                if field not in candidate:
                    return False, f"Candidate {i} missing required field: {field}"

            # Validate candidate DINO descriptor
            candidate_dino = candidate['dino']
            if not isinstance(candidate_dino, np.ndarray):
                return False, f"Candidate {i} DINO must be numpy array, got {type(candidate_dino)}"
            if candidate_dino.size == 0:
                return False, f"Candidate {i} DINO descriptor is empty"
            if candidate_dino.ndim != 1:
                return False, f"Candidate {i} DINO must be 1D array, got shape {candidate_dino.shape}"
            # Validate dimension matches config (384 for dinov3-vitb16)
            from src.config.appConfig import get_config
            expected_dim = get_config().milvus_vector_dim
            if candidate_dino.shape[0] != expected_dim:
                return False, f"Candidate {i} DINO must be {expected_dim}-dimensional, got {candidate_dino.shape[0]}"
            if np.isnan(candidate_dino).any() or np.isinf(candidate_dino).any():
                return False, f"Candidate {i} DINO descriptor contains NaN or Inf values"

            # Validate candidate keypoints and descriptors
            candidate_kp = candidate['keypoints']
            candidate_desc = candidate['descriptors']

            is_valid, error_msg = self._validate_match_inputs(candidate_kp, candidate_desc, candidate_kp, candidate_desc)
            if not is_valid:
                return False, f"Candidate {i} validation failed: {error_msg}"

            # Validate candidate has tree_id
            if 'tree_id' not in candidate:
                return False, f"Candidate {i} missing 'tree_id' field"
            candidate_tree_id = candidate.get('tree_id')
            if candidate_tree_id is None or (isinstance(candidate_tree_id, str) and not candidate_tree_id.strip()):
                return False, f"Candidate {i} has invalid tree_id: '{candidate_tree_id}'"

            # Validate optional candidate scores
            if 'scores' in candidate and candidate['scores'] is not None:
                candidate_scores = candidate['scores']
                if not isinstance(candidate_scores, np.ndarray):
                    return False, f"Candidate {i} scores must be numpy array, got {type(candidate_scores)}"
                if candidate_scores.shape[0] != candidate_kp.shape[0]:
                    return False, f"Candidate {i} scores/keypoints count mismatch: {candidate_scores.shape[0]} vs {candidate_kp.shape[0]}"
                if np.isnan(candidate_scores).any() or np.isinf(candidate_scores).any():
                    return False, f"Candidate {i} scores contain NaN or Inf values"

        return True, None

    def geometric_verification(
        self,
        query_kp: np.ndarray,
        candidate_kp: np.ndarray
    ) -> Tuple[np.ndarray, int, float, float]:
        """
        Stage 3: Geometric verification with RANSAC.

        Uses RANSAC to estimate transformation matrix and reject outliers.
        This ensures that matches are geometrically consistent.

        Args:
            query_kp: Matched query keypoints (N, 2)
            candidate_kp: Matched candidate keypoints (N, 2)

        Returns:
            Tuple of (transformation_matrix, inlier_count, reprojection_error, elapsed_time)
        """
        if len(query_kp) < self.min_inliers:
            logger.debug(f"[RANSAC] Insufficient keypoints: {len(query_kp)} < {self.min_inliers}")
            return None, 0, float('inf'), 0.0

        import time
        start_time = time.time()

        logger.info(f"[RANSAC] Starting geometric verification...")
        logger.info(f"  Input keypoints: {len(query_kp)}")
        logger.info(f"  RANSAC threshold: {self.ransac_threshold}px")
        logger.info(f"  Min inliers: {self.min_inliers}")

        # Convert to float32
        src_pts = query_kp.astype(np.float32).reshape(-1, 1, 2)
        dst_pts = candidate_kp.astype(np.float32).reshape(-1, 1, 2)

        if self.use_fundamental:
            logger.info(f"[RANSAC] Using Fundamental matrix estimation...")
            # Use Fundamental matrix (more general, handles perspective)
            F, mask = cv2.findFundamentalMat(
                src_pts, dst_pts,
                cv2.FM_RANSAC,
                r1=self.ransac_threshold,
                r2=0.99
            )
            transformation = F
        else:
            logger.info(f"[RANSAC] Using Homography estimation...")
            # Use Homography (assumes planar scene)
            H, mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold
            )
            transformation = H

        if transformation is None:
            elapsed = time.time() - start_time
            logger.warning(f"[RANSAC] ✗ Transformation estimation failed")
            return None, 0, float('inf'), elapsed

        # Count inliers
        inlier_count = int(mask.sum())
        logger.info(f"[RANSAC] Inliers found: {inlier_count}/{len(query_kp)} ({100*inlier_count/len(query_kp):.1f}%)")

        # Calculate reprojection error
        if self.use_fundamental:
            logger.info(f"[RANSAC] Calculating symmetric epipolar error...")
            # For fundamental matrix, use symmetric epipolar error
            reprojection_error = self._calculate_epipolar_error(
                src_pts, dst_pts, transformation
            )
        else:
            logger.info(f"[RANSAC] Calculating reprojection error...")
            # For homography, use direct reprojection error
            projected = cv2.perspectiveTransform(src_pts, transformation)
            error = np.linalg.norm(dst_pts - projected, axis=2)
            reprojection_error = float(error[mask].mean()) if inlier_count > 0 else float('inf')

        elapsed = time.time() - start_time

        logger.info(f"[RANSAC] ✓ Geometric verification complete")
        logger.info(f"  Inliers: {inlier_count}")
        logger.info(f"  Reprojection error: {reprojection_error:.4f}px")
        logger.info(f"  Time: {elapsed:.3f}s")

        return transformation, inlier_count, reprojection_error, elapsed
    
    def _calculate_epipolar_error(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        F: np.ndarray
    ) -> float:
        """Calculate symmetric epipolar error for fundamental matrix."""
        # Convert to homogeneous coordinates
        pts1_h = cv2.convertPointsToHomogeneous(pts1).squeeze()
        pts2_h = cv2.convertPointsToHomogeneous(pts2).squeeze()
        
        # Calculate epipolar lines
        l2 = (F @ pts1_h.T).T  # Epipolar lines in image 2
        l1 = (F.T @ pts2_h.T).T  # Epipolar lines in image 1
        
        # Calculate distances
        d2 = np.abs(np.sum(pts2_h * l2, axis=1)) / np.linalg.norm(l2[:, :2], axis=1)
        d1 = np.abs(np.sum(pts1_h * l1, axis=1)) / np.linalg.norm(l1[:, :2], axis=1)
        
        # Symmetric error
        error = float(np.mean(d1 + d2))
        
        return error
    
    def compute_final_score(
        self,
        dino_similarity: float,
        match_ratio: float,
        inlier_ratio: float,
        reprojection_error: float = 0.0,
        inlier_count: int = 0,
        match_count: int = 0,
    ) -> float:
        """
        Compute final matching score.

        Formula:
          score = 0.65 * dino
                + 0.15 * normalized_match_count * geo_penalty
                + 0.20 * inlier_ratio * geo_penalty

        ``normalized_match_count`` = min(match_count / 100, 1.0) instead of
        the raw match_ratio (matches/total_kp) which is diluted to near-zero
        with 1400+ keypoints.  This makes 23 matches vs 14 actually matter.

        ``geo_penalty`` scales down SP/inlier when reproj error > 20 px.

        Args:
            dino_similarity: DINO cosine similarity [0, 1]
            match_ratio: SuperPoint match ratio (fallback if match_count=0)
            inlier_ratio: Geometric inlier ratio [0, 1]
            reprojection_error: Average RANSAC reprojection error (pixels)
            inlier_count: Absolute number of RANSAC inliers
            match_count: Absolute number of LightGlue matches

        Returns:
            Final score
        """
        # Penalise the local-feature contributions when geometry is poor
        geo_penalty: float = 1.0
        if reprojection_error > 20.0 and not np.isinf(reprojection_error):
            geo_penalty = min(1.0, 20.0 / reprojection_error)

        # Use normalized match count: min(matches / 100, 1.0)
        # This gives meaningful differentiation between e.g. 23 and 14 matches,
        # unlike match_ratio which divides by 1400+ total keypoints.
        sp_signal: float = min(match_count / 100.0, 1.0) if match_count > 0 else match_ratio

        score: float = (
            self.weights['dino_similarity'] * dino_similarity
            + self.weights['superpoint_match_ratio'] * sp_signal * geo_penalty
            + self.weights['inlier_ratio'] * inlier_ratio * geo_penalty
        )

        return score
    
    def _compute_texture_similarity(
        self,
        _query: QueryFeatures,
        _candidate: CandidateFeatures
    ) -> float:
        """Compute bark texture similarity between query and candidate.

        Note: Bark texture processing has been removed from the pipeline.
        Returns 0.0.
        """
        return 0.0

    def match(
        self,
        query: QueryFeatures,
        candidates: List[CandidateFeatures],
        query_id: str = "query",
        use_neighbor_aggregation: bool = True,
        neighbors_per_candidate: int = 5
    ) -> List[MatchingResult]:
        """
        Perform complete hierarchical matching pipeline with neighbor aggregation.

        Args:
            query: Query feature dict with keys:
                - 'dino': DINO descriptor (1024-dim)
                - 'keypoints': SuperPoint keypoints (N, 2)
                - 'descriptors': SuperPoint descriptors (N, 256)
                - 'scores': SuperPoint keypoint scores (N,)
                - 'texture_histogram': Optional bark texture histogram
            candidates: List of candidate feature dicts (same structure)
            query_id: Query identifier
            use_neighbor_aggregation: If True, aggregate scores with neighbors
            neighbors_per_candidate: Number of neighbors to use per candidate

        Returns:
            List of MatchingResult sorted by final_score (descending)
        """
        import time  # Ensure time is available in this method
        # Validate inputs
        is_valid, error_msg = self._validate_match_pipeline_inputs(query, candidates)
        if not is_valid:
            logger.error(f"[match] Validation failed: {error_msg}")
            raise ValueError(error_msg)

        total_start = time.time()

        # Stage 1: Global search with DINO - Adaptive filtering
        query_dino = query['dino']
        candidate_dinos = [c['dino'] for c in candidates]

        # Adaptive top_k based on DINO similarity threshold
        # Start with smaller candidate set for efficiency
        initial_top_k = 100  # Reduced from 20
        candidate_indices = self.search_candidates(
            query_dino, candidate_dinos, top_k=initial_top_k
        )

        if candidate_indices:
            logger.info(f"[match] Top candidate similarity: {candidate_indices[0][1]:.4f}")

        if not candidate_indices:
            logger.warning(f"No candidates found after DINO search")
            return []

        # Early filtering: only keep candidates with DINO similarity >= threshold
        # Use instance threshold (already applied in search_candidates, but apply again for safety)
        filtered_indices = [(idx, sim) for idx, sim in candidate_indices if sim >= self.dino_threshold]

        if not filtered_indices:
            logger.warning(f"No candidates passed DINO threshold {self.dino_threshold}")
            # Keep top 5 even if below threshold for NO_MATCH response
            filtered_indices = candidate_indices[:5]

        candidate_indices = filtered_indices

        # Build neighbor groups for aggregation if enabled
        neighbor_groups = {}
        if use_neighbor_aggregation:
            logger.info(f"[match] Building neighbor groups for {len(candidate_indices)} candidates...")
            for idx, _ in candidate_indices:
                candidate = candidates[idx]
                candidate_id = candidate.get('id', f"candidate_{idx}")
                candidate_dino = candidate.get('dino')

                if candidate_dino is None:
                    continue

                # Find neighbors within same tree with cosine similarity > 0.85
                tree_id = candidate.get('tree_id')
                neighbors = self.find_neighbors(
                    target_dino=candidate_dino,
                    candidates=candidates,
                    tree_id=tree_id,
                    top_k=neighbors_per_candidate,
                    exclude_target_id=candidate_id,
                    similarity_threshold=0.9
                )

                neighbor_ids = [n.get('id', '') for n, _ in neighbors]
                neighbor_indices = [
                    next((i for i, c in enumerate(candidates) if c.get('id') == nid), None)
                    for nid in neighbor_ids
                ]
                neighbor_indices = [i for i in neighbor_indices if i is not None]

                neighbor_groups[idx] = neighbor_indices
                logger.info(f"  Candidate {candidate_id}: {len(neighbor_indices)} neighbors found (cosine > 0.85)")

        # Extract query scores and image size for score-aware matching
        q_kp = query['keypoints']
        q_desc = query['descriptors']
        q_scores = query.get('scores', None)
        q_image_size = query.get('image_size', None)

        # Stage 2 & 3: Local matching + geometric verification
        # IMPORTANT: We process ALL candidates and track their scores,
        # even if they don't pass inlier threshold.
        # This allows confidence calculation for NO_MATCH cases.
        all_candidate_results = []

        # Early termination flag
        found_high_confidence_match = False

        # Use CUDA streams for parallel processing if enabled
        if self.enable_cuda_streams and len(self.cuda_streams) > 0:
            try:
                # Start timing for CUDA streams
                cuda_streams_start = time.time()

                # Parallel processing with CUDA streams
                logger.info(f"[MATCH] Using CUDA streams for parallel processing ({len(self.cuda_streams)} streams)")

                # Monitor GPU memory before processing
                if torch.cuda.is_available():
                    try:
                        gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                        gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                        logger.info(f"[GPU] Memory before processing: Allocated={gpu_mem_allocated:.2f}GB, Reserved={gpu_mem_reserved:.2f}GB")
                    except Exception as e:
                        logger.warning(f"[GPU] Could not get memory info: {e}")
                stream_results = []

                # Phase 1: Launch all SuperPoint matching operations in parallel across streams
                for stream_idx, (idx, dino_sim) in enumerate(candidate_indices):
                    candidate = candidates[idx]
                    candidate_id = candidate.get('id', f"candidate_{idx}")
                    candidate_tree_id = candidate.get('tree_id')
                    if candidate_tree_id is None or candidate_tree_id == "":
                        logger.error(f"[match] Invalid tree_id: {candidate_tree_id}")
                        continue

                    # Extract features
                    c_kp = candidate['keypoints']
                    c_desc = candidate['descriptors']
                    c_scores = candidate.get('scores', None)
                    c_image_size = candidate.get('image_size', None)

                    # Select stream in round-robin fashion
                    stream = self.cuda_streams[stream_idx % len(self.cuda_streams)]

                    # Launch async matching on stream
                    with torch.cuda.stream(stream):
                        # Stage 2: SuperPoint matching with CUDA stream (async)
                        matched_results = self.match_superpoint(
                            q_kp, q_desc, c_kp, c_desc,
                            query_scores=q_scores, candidate_scores=c_scores,
                            query_image_size=q_image_size, candidate_image_size=c_image_size,
                            stream=stream,
                        )

                        # Store results for later processing
                        stream_results.append({
                            'stream': stream,
                            'idx': idx,
                            'dino_sim': dino_sim,
                            'candidate': candidate,
                            'candidate_id': candidate_id,
                            'candidate_tree_id': candidate_tree_id,
                            'c_kp': c_kp,
                            'c_desc': c_desc,
                            'c_scores': c_scores,
                            'matched_results': matched_results
                        })

                # Phase 2: Synchronize streams and run geometric verification in parallel
                geometric_tasks = []
                for stream_data in stream_results:
                    stream = stream_data['stream']

                    # Check GPU memory before processing each candidate
                    try:
                        if torch.cuda.is_available():
                            gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
                            if gpu_mem_allocated > 6.0:  # Warning if using more than 6GB
                                logger.warning(f"[GPU] High memory usage: {gpu_mem_allocated:.2f}GB")
                    except Exception:
                        pass

                    stream.synchronize()  # Wait for SuperPoint matching to complete

                    # Extract matched results
                    matched_results = stream_data['matched_results']
                    _, q_kp_matched, c_kp_matched, q_scores_matched, c_scores_matched, superpoint_time = matched_results

                    # Log score-based match quality
                    if len(q_scores_matched) > 0:
                        match_quality = np.minimum(q_scores_matched, c_scores_matched)
                        logger.info(f"[MATCH] Score-weighted match quality: min={match_quality.min():.3f}, max={match_quality.max():.3f}, mean={match_quality.mean():.3f}")

                    # Store for geometric verification
                    geometric_tasks.append({
                        'stream': stream,
                        'stream_data': stream_data,
                        'q_kp_matched': q_kp_matched,
                        'c_kp_matched': c_kp_matched,
                        'q_scores_matched': q_scores_matched,
                        'c_scores_matched': c_scores_matched,
                        'superpoint_time': superpoint_time
                    })

                # Phase 3: Run geometric verification in parallel across streams
                for geo_task in geometric_tasks:
                    stream = geo_task['stream']
                    stream_data = geo_task['stream_data']
                    q_kp_matched = geo_task['q_kp_matched']
                    c_kp_matched = geo_task['c_kp_matched']

                    idx = stream_data['idx']
                    dino_sim = stream_data['dino_sim']
                    candidate = stream_data['candidate']
                    candidate_id = stream_data['candidate_id']
                    candidate_tree_id = stream_data['candidate_tree_id']
                    c_kp = stream_data['c_kp']

                    # Skip entirely if insufficient LightGlue matches (below min_matches threshold)
                    if len(q_kp_matched) < self.min_matches:
                        logger.info(f"[MATCH] ✗ Skipping {candidate_id}: only {len(q_kp_matched)} matches < {self.min_matches} min_matches")
                        continue

                    # Calculate match ratio
                    min_kp = min(len(q_kp), len(c_kp))
                    match_ratio = len(q_kp_matched) / min_kp if min_kp > 0 else 0.0

                    # Stage 3: Geometric verification (RANSAC) - run in parallel with stream context
                    with torch.cuda.stream(stream):
                        transformation, inliers, reproj_error, geo_time = self.geometric_verification(
                            q_kp_matched, c_kp_matched
                        )

                    # Store geometric results for final processing
                    geo_task['transformation'] = transformation
                    geo_task['inliers'] = inliers
                    geo_task['reproj_error'] = reproj_error
                    geo_task['match_ratio'] = match_ratio
                    geo_task['geometric_time'] = geo_time

                # Phase 4: Synchronize all geometric verification and create results
                for geo_task in geometric_tasks:
                    stream = geo_task['stream']
                    stream.synchronize()  # Wait for geometric verification to complete

                    stream_data = geo_task['stream_data']
                    q_kp_matched = geo_task['q_kp_matched']
                    c_kp_matched = geo_task['c_kp_matched']

                    # Skip if this was already processed (insufficient matches)
                    if 'transformation' not in geo_task:
                        continue

                    transformation = geo_task['transformation']
                    inliers = geo_task['inliers']
                    reproj_error = geo_task['reproj_error']
                    match_ratio = geo_task['match_ratio']

                    idx = stream_data['idx']
                    dino_sim = stream_data['dino_sim']
                    candidate = stream_data['candidate']
                    candidate_id = stream_data['candidate_id']
                    candidate_tree_id = stream_data['candidate_tree_id']

                    # Calculate inlier ratio
                    inlier_ratio = inliers / len(q_kp_matched) if len(q_kp_matched) > 0 else 0.0

                    # Filter out candidates with inliers below threshold after RANSAC
                    if inliers < self.min_inliers:
                        logger.info(f"[MATCH] ✗ Skipping {candidate_id}: only {inliers} RANSAC inliers < {self.min_inliers} min_inliers")
                        continue

                    # Compute final score
                    final_score = self.compute_final_score(
                        dino_sim, match_ratio, inlier_ratio,
                        reprojection_error=reproj_error,
                        inlier_count=inliers,
                        match_count=len(q_kp_matched)
                    )
                    logger.info(f"[MATCH] ✓ Final score: {final_score:.4f}")

                    # Compute bark texture similarity
                    import time
                    texture_start = time.time()
                    texture_sim = self._compute_texture_similarity(query, candidate)
                    texture_elapsed = time.time() - texture_start

                    # Create result
                    result = MatchingResult(
                        query_id=query_id,
                        candidate_id=candidate_id,
                        tree_id=candidate_tree_id,
                        dino_similarity=dino_sim,
                        superpoint_matches=len(q_kp_matched),
                        superpoint_inliers=inliers,
                        superpoint_match_ratio=match_ratio,
                        homography=transformation if not self.use_fundamental else None,
                        fundamental=transformation if self.use_fundamental else None,
                        reprojection_error=reproj_error,
                        ransac_inlier_ratio=float(inlier_ratio),
                        bark_texture_similarity=texture_sim,
                        final_score=final_score,
                        dino_search_time=getattr(self, '_dino_search_time', 0.0),
                        superpoint_match_time=geo_task.get('superpoint_time', 0.0),
                        geometric_verify_time=geo_task.get('geometric_time', 0.0),
                        texture_match_time=texture_elapsed,
                        captured_at=candidate.get('captured_at'),  # Include timestamp from candidate
                        # total_time=time.time() - total_start,
                        match_coordinates=list(zip(
                            [(int(p[0]), int(p[1])) for p in q_kp_matched],
                            [(int(p[0]), int(p[1])) for p in c_kp_matched]
                        ))
                    )

                    all_candidate_results.append(result)

                    # Early termination: if we found a high-confidence match, stop processing
                    # High confidence = DINO > 0.80, inliers > 50, final_score > 0.75
                    if dino_sim > 0.80 and inliers > 50 and final_score > 0.75:
                        logger.info(f"[EARLY TERMINATION] Found high-confidence match: {candidate_id} (score={final_score:.3f}, inliers={inliers})")
                        found_high_confidence_match = True
                        break

                # Record CUDA streams timing
                cuda_streams_end = time.time()
                self.cuda_streams_time = cuda_streams_end - cuda_streams_start
                self.total_matches_with_streams = len(all_candidate_results)
                logger.info(f"[PERF] CUDA streams processing time: {self.cuda_streams_time:.3f}s for {self.total_matches_with_streams} candidates")

                if found_high_confidence_match:
                    logger.info(f"[PERF] Early termination saved processing {len(geometric_tasks) - len(all_candidate_results)} candidates")

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                # Fallback to sequential processing on CUDA error
                logger.error(f"[MATCH] CUDA streams error: {e}")
                logger.warning("[MATCH] Falling back to sequential processing...")

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Sequential fallback
                all_candidate_results = self._match_sequential(
                    query, candidates, candidate_indices,
                    q_kp, q_desc, q_scores, q_image_size, None, total_start
                )

        # Sequential processing (fallback when CUDA streams disabled)
        else:
            logger.info("[MATCH] Using sequential processing (CUDA streams disabled)")
            all_candidate_results = self._match_sequential(
                query, candidates, candidate_indices,
                q_kp, q_desc, q_scores, q_image_size, None, total_start
            )

        # Neighbor aggregation: enhance candidate scores with neighbor results
        if use_neighbor_aggregation and neighbor_groups:
            logger.info(f"[match] Aggregating neighbor scores for {len(all_candidate_results)} candidates...")

            # Create a map of candidate_id to result for quick lookup
            result_map = {r.candidate_id: r for r in all_candidate_results}

            # For each candidate, aggregate with its neighbors
            for candidate_result in all_candidate_results:
                candidate_id = candidate_result.candidate_id

                # Find the original index of this candidate
                orig_idx = next(
                    (idx for idx, _ in candidate_indices
                     if candidates[idx].get('id') == candidate_id),
                    None
                )

                if orig_idx is None or orig_idx not in neighbor_groups:
                    continue

                neighbor_indices = neighbor_groups[orig_idx]
                if not neighbor_indices:
                    continue

                # Collect neighbor results
                neighbor_scores = []
                for n_idx in neighbor_indices:
                    n_candidate = candidates[n_idx]
                    n_id = n_candidate.get('id', f"candidate_{n_idx}")

                    # Find result for this neighbor
                    n_result = result_map.get(n_id)
                    if n_result:
                        neighbor_scores.append(n_result.final_score)

                if neighbor_scores:
                    # Aggregate: 60% candidate score + 40% average neighbor score
                    avg_neighbor_score = float(np.mean(neighbor_scores))
                    aggregated_score = 0.6 * candidate_result.final_score + 0.4 * avg_neighbor_score

                    logger.info(
                        f"  {candidate_id}: "
                        f"original={candidate_result.final_score:.4f}, "
                        f"neighbors_avg={avg_neighbor_score:.4f}, "
                        f"aggregated={aggregated_score:.4f}"
                    )

                    # Update the result with aggregated score
                    candidate_result.final_score = aggregated_score

        # Sort by final score and return
        all_candidate_results.sort(key=lambda r: r.final_score, reverse=True)

        if self.cuda_streams_time > 0:
            logger.info(f"[PERF] CUDA streams total time: {self.cuda_streams_time:.3f}s")

        return all_candidate_results

    def _match_sequential(
        self,
        query: QueryFeatures,
        candidates: List[CandidateFeatures],
        candidate_indices: List[Tuple[int, float]],
        q_kp: np.ndarray,
        q_desc: np.ndarray,
        q_scores: Optional[np.ndarray],
        q_image_size: Optional[Tuple[int, int]],
        stream: Optional[Any],
        total_start: float
    ) -> List[MatchingResult]:
        """Sequential fallback for candidate matching."""
        all_results = []

        for idx, dino_sim in candidate_indices:
            candidate = candidates[idx]
            candidate_id = candidate.get('id', f"candidate_{idx}")
            candidate_tree_id = candidate.get('tree_id')
            if not candidate_tree_id:
                continue

            c_kp = candidate['keypoints']
            c_desc = candidate['descriptors']
            c_scores = candidate.get('scores', None)
            c_image_size = candidate.get('image_size', None)

            # SuperPoint matching
            matched_results = self.match_superpoint(
                q_kp, q_desc, c_kp, c_desc,
                query_scores=q_scores, candidate_scores=c_scores,
                query_image_size=q_image_size, candidate_image_size=c_image_size,
                stream=stream,
            )
            _, q_kp_matched, c_kp_matched, _, _, superpoint_time = matched_results

            # Skip entirely if insufficient LightGlue matches (below min_matches threshold)
            if len(q_kp_matched) < self.min_matches:
                logger.info(f"[MATCH] ✗ Skipping {candidate_id}: only {len(q_kp_matched)} matches < {self.min_matches} min_matches")
                continue

            # Geometric verification
            min_kp = min(len(q_kp), len(c_kp))
            match_ratio = len(q_kp_matched) / min_kp if min_kp > 0 else 0.0
            transformation, inliers, reproj_error, geo_time = self.geometric_verification(
                q_kp_matched, c_kp_matched
            )

            inlier_ratio = inliers / len(q_kp_matched) if len(q_kp_matched) > 0 else 0.0

            # Filter out candidates with inliers below threshold after RANSAC
            if inliers < self.min_inliers:
                logger.info(f"[MATCH] ✗ Skipping {candidate_id}: only {inliers} RANSAC inliers < {self.min_inliers} min_inliers")
                continue

            final_score = self.compute_final_score(dino_sim, match_ratio, inlier_ratio, reprojection_error=reproj_error, inlier_count=inliers, match_count=len(q_kp_matched))

            texture_start = time.time()
            texture_sim = self._compute_texture_similarity(query, candidate)
            texture_elapsed = time.time() - texture_start

            result = MatchingResult(
                query_id="query",
                candidate_id=candidate_id,
                tree_id=candidate_tree_id,
                dino_similarity=dino_sim,
                superpoint_matches=len(q_kp_matched),
                superpoint_inliers=inliers,
                superpoint_match_ratio=match_ratio,
                homography=transformation if not self.use_fundamental else None,
                fundamental=transformation if self.use_fundamental else None,
                reprojection_error=reproj_error,
                ransac_inlier_ratio=float(inlier_ratio),
                bark_texture_similarity=texture_sim,
                final_score=final_score,
                dino_search_time=getattr(self, '_dino_search_time', 0.0),
                superpoint_match_time=superpoint_time,
                geometric_verify_time=geo_time,
                texture_match_time=texture_elapsed,
                total_time=time.time() - total_start,
                match_coordinates=list(zip(
                    [(int(p[0]), int(p[1])) for p in q_kp_matched],
                    [(int(p[0]), int(p[1])) for p in c_kp_matched]
                ))
            )
            all_results.append(result)

        return all_results


def create_hierarchical_matcher(
    dino_threshold: float = 0.2,  # Very permissive - SuperPoint/RANSAC will filter
    superpoint_match_ratio: float = 0.1,
    ransac_threshold: float = 5.0,
    min_inliers: int = 5,  # Minimum RANSAC inliers (lowered from 20 for test images)
    min_matches: int = 10,  # Minimum LightGlue matched keypoints (lowered from 90 for test images)
    use_fundamental: bool = False,
    # SuperPoint configuration
    sp_max_keypoints: int = 4096,
    # LightGlue configuration
    lg_device: str = "cuda",
    lg_confidence: float = 0.1,
) -> HierarchicalMatcher:
    """
    Factory function to create hierarchical matcher.

    Args:
        dino_threshold: Minimum DINO similarity for candidates
        superpoint_match_ratio: Minimum SuperPoint match ratio
        ransac_threshold: RANSAC threshold in pixels
        min_inliers: Minimum RANSAC inliers
        min_matches: Minimum LightGlue matched keypoints
        use_fundamental: Use fundamental matrix instead of homography
        sp_max_keypoints: Max keypoints for SuperPoint (default: from appConfig)
        lg_device: Device for LightGlue ('cuda' or 'cpu', default: from appConfig)
        lg_confidence: Confidence threshold for LightGlue (default: from appConfig)

    Returns:
        Configured HierarchicalMatcher instance
    """
    return HierarchicalMatcher(
        dino_threshold=dino_threshold,
        superpoint_match_ratio=superpoint_match_ratio,
        ransac_threshold=ransac_threshold,
        min_inliers=min_inliers,
        min_matches=min_matches,
        use_fundamental=use_fundamental,
        sp_max_keypoints=sp_max_keypoints,
        lg_device=lg_device,
        lg_confidence=lg_confidence,
    )


# Convenience functions for backward compatibility
def match_with_geometric_verification(
    query_kp: np.ndarray,
    query_desc: np.ndarray,
    candidate_kp: np.ndarray,
    candidate_desc: np.ndarray,
    ransac_threshold: float = 5.0,
    min_inliers: int = 20
) -> Tuple[int, float, Optional[np.ndarray]]:
    """
    Match features with geometric verification.
    
    Legacy function for backward compatibility.
    
    Returns:
        (inlier_count, reprojection_error, transformation_matrix)
    """
    matcher = create_hierarchical_matcher(
        ransac_threshold=ransac_threshold,
        min_inliers=min_inliers
    )
    
    # Match features
    _, q_kp, c_kp, _, _, _ = matcher.match_superpoint(
        query_kp, query_desc, candidate_kp, candidate_desc
    )

    # Geometric verification
    transformation, inliers, error, _ = matcher.geometric_verification(q_kp, c_kp)

    return inliers, error, transformation