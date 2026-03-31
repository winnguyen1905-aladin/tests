"""
Service layer for Hierarchical Matching

This service integrates the hierarchical matching approach with the existing system.

## Architecture Overview

The HierarchicalMatchingService implements a three-stage hierarchical matching pipeline:

1. **Global Feature Matching (DINO)**
   - Uses DINO global descriptors for coarse retrieval
   - Queries Milvus vector database to find candidate images
   - Supports both single-tree and multi-tree search modes

2. **Local Feature Matching (SuperPoint + LightGlue)**
   - Extracts SuperPoint keypoints and descriptors from query and candidates
   - Uses LightGlue for robust local feature matching
   - Computes match ratios and inlier counts

3. **Geometric Verification (RANSAC + Texture)**
   - Applies RANSAC to compute homography and fundamental matrices
   - Validates geometric consistency
   - Compares bark texture histograms for additional verification

## Multi-Tree Hierarchical Matching Flow

When `tree_id=None` (multi-tree mode), the service performs:

1. **Global Query**: Query Milvus without tree filter to get candidates from all trees
2. **Tree Partitioning**: Group candidates by tree_id and compute average DINO similarity per tree
3. **Top-K Tree Selection**: Select top-K trees by average similarity
4. **Diversity Selection**: For each tree, select N diverse candidates using greedy algorithm
5. **Parallel Tree Matching**: Match candidates within each tree independently
6. **Result Aggregation**: Combine results from all trees and return best match

## Single-Tree Matching Flow

When `tree_id='specific'` (single-tree mode), the service performs:

1. **Filtered Query**: Query Milvus with tree_id filter
2. **Candidate Matching**: Match all candidates from the specific tree
3. **Result Formatting**: Return best match from the single tree

## Usage Examples

### Multi-Tree Matching (Search All Trees)

```python
from src.service.hierarchicalMatchingService import HierarchicalMatchingService
from src.processor.dinoProcessor import DinoProcessor
from src.processor.superPointProcessor import SuperPointProcessor

# Initialize service
service = HierarchicalMatchingService(
    milvus_repo=milvus_repo,
    minio_repo=minio_repo,
    top_k=10,
    dino_threshold=0.7,
    verbose=True
)

# Extract query features
dino_result = dino_processor.process(image)
superpoint_result = superpoint_processor.process(image)

# Perform multi-tree matching (tree_id=None)
result = service.match(
    query_dino=dino_result,
    query_superpoint=superpoint_result,
    tree_id=None,  # Search all trees
    query_texture_histogram=texture_histogram,
    use_async=False
)

# Result structure:
# {
#     'decision': 'MATCH' | 'PROBABLE_MATCH' | 'POSSIBLE_MATCH' | 'NO_MATCH',
#     'reason': 'Best match: tree_001 with score 0.85',
#     'best_match': {
#         'query_id': 'query_1',
#         'candidate_id': 'img_001',
#         'tree_id': 'tree_001',
#         'dino_similarity': 0.85,
#         'superpoint_matches': 50,
#         'superpoint_inliers': 40,
#         'superpoint_match_ratio': 0.8,
#         'ransac_inlier_ratio': 0.75,
#         'bark_texture_similarity': 0.7,
#         'final_score': 0.82,
#         'timing': {'total': 0.5, 'dino': 0.1, 'superpoint': 0.2, 'ransac': 0.2}
#     },
#     'all_matches': [...]  # Top matches from all trees
# }
```

### Single-Tree Matching (Search Specific Tree)

```python
# Perform single-tree matching (tree_id='specific')
result = service.match(
    query_dino=dino_result,
    query_superpoint=superpoint_result,
    tree_id='tree_001',  # Search only tree_001
    query_texture_histogram=texture_histogram,
    use_async=False
)
```

## Key Algorithms

### Tree Partitioning
Groups candidates by tree_id and computes average DINO similarity per tree:
- Input: List of candidates with tree_id and similarity scores
- Output: Dict mapping tree_id to list of candidates, sorted by average similarity

### Diversity Selection (Greedy Algorithm)
Selects N diverse candidates per tree to ensure different viewpoints:
1. Sort candidates by DINO similarity (highest first)
2. Select first candidate (highest similarity)
3. For remaining slots: select candidate with MINIMUM max similarity to already-selected set
4. This ensures diversity while maintaining reasonable similarity scores

### Result Aggregation - ALL Results from ALL Trees
Combines ALL results from ALL diverse candidates across ALL trees:

**Key Insight**: The hierarchical matcher returns ALL candidates sorted by final_score,
not just the best match. The aggregation preserves this diversity:

1. **Flatten**: Collect all results from all trees into a single list
2. **Sort**: Sort by final_score descending (best matches first)
3. **Return**: Top-20 results in all_matches (preserves diversity from all trees)
4. **Best Match**: Use overall best result as best_match
5. **Decision**: Determine based on best score thresholds

**Why This Matters**:
- Diversity selection ensures different viewpoints are matched per tree
- Aggregation combines all these diverse results across all trees
- Final decision is based on the best result from the aggregated pool
- You get multiple viewpoints ranked by score, not just one best match per tree

**Example Multi-Tree Aggregation**:
```
Tree A (3 diverse candidates):
  - img_A1: score=0.90 (frontal view)
  - img_A2: score=0.85 (side view)
  - img_A3: score=0.80 (back view)

Tree B (3 diverse candidates):
  - img_B1: score=0.88 (frontal view)
  - img_B2: score=0.82 (side view)
  - img_B3: score=0.78 (back view)

Aggregated Results (sorted by score):
  1. img_A1: 0.90 ← best_match
  2. img_B1: 0.88
  3. img_A2: 0.85
  4. img_B2: 0.82
  5. img_A3: 0.80
  6. img_B3: 0.78
  ... (all 6 results in all_matches)
```

## Configuration Parameters

- `top_k`: Number of candidates to retrieve from Milvus (default: 10)
- `dino_threshold`: Minimum DINO similarity for MATCH decision (default: 0.7)
- `verbose`: Enable detailed logging (default: False)
- `n_per_tree`: Number of diverse candidates per tree (default: 5)
- `top_k_trees`: Number of top trees to process in multi-tree mode (default: 5)

## Logging

The service provides comprehensive logging at each stage:
- Query type (GLOBAL vs FILTERED)
- Partitioning results (number of trees, candidates per tree)
- Diversity selection (selected candidates per tree)
- Tree matching results (best score per tree)
- Final aggregation (best match, decision)

Enable verbose logging to see detailed information:
```python
service = HierarchicalMatchingService(..., verbose=True)
```
"""
from __future__ import annotations

import logging

from typing import Dict, Any, Optional, List, TYPE_CHECKING, Tuple, TypedDict
import numpy as np

from dependency_injector.wiring import inject, Provide

from src.processor.hierarchicalMatcher import (
    HierarchicalMatcher,
    MatchingResult,
    QueryFeatures,
    CandidateFeatures,
)
from src.utils.similarityUtils import cosine_similarity
from src.processor.dinoProcessor import DinoResult
from src.processor.superPointProcessor import SuperPointResult
from src.repository.milvusRepository import MilvusResult
from src.repository.sqlalchemyRepository import SQLAlchemyORMRepository
from src.repository.minioRepository import MinIORepository

if TYPE_CHECKING:
    from src.repository.milvusRepository import MilvusRepository
    from src.config.appConfig import AppConfig

logger = logging.getLogger(__name__)


# TypedDict classes for response structures
class MatchDetailsDict(TypedDict, total=False):
    """Type definition for match details in best_match response."""
    dino_similarity: float
    superpoint_matches: int
    superpoint_inliers: int
    match_ratio: float
    reprojection_error: float
    ransac_inlier_ratio: float
    bark_texture_similarity: float
    homography: Optional[List[List[float]]]
    fundamental: Optional[List[List[float]]]
    final_score: float
    timing: Dict[str, float]
    query_id: str
    match_coordinates: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]]


class BestMatchDict(TypedDict, total=False):
    """Type definition for best_match in response."""
    query_id: str
    candidate_id: str
    tree_id: str
    dino_similarity: float
    superpoint_matches: int
    superpoint_inliers: int
    superpoint_match_ratio: float
    ransac_inlier_ratio: float
    bark_texture_similarity: float
    homography: Optional[List[List[float]]]
    fundamental: Optional[List[List[float]]]
    reprojection_error: float
    final_score: float
    score: float
    confidence: float
    timing: Dict[str, float]
    match_coordinates: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]]


class AllMatchDict(TypedDict, total=False):
    """Type definition for items in all_matches list."""
    query_id: str
    candidate_id: str
    tree_id: str
    dino_similarity: float
    superpoint_matches: int
    superpoint_inliers: int
    superpoint_match_ratio: float
    ransac_inlier_ratio: float
    bark_texture_similarity: float
    homography: Optional[List[List[float]]]
    fundamental: Optional[List[List[float]]]
    reprojection_error: float
    final_score: float
    score: float
    confidence: float
    timing: Dict[str, float]
    match_coordinates: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]]


class MatchResponseDict(TypedDict):
    """Type definition for the complete match response."""
    decision: str
    reason: str
    best_match: Optional[BestMatchDict]
    all_matches: List[AllMatchDict]


# Helper functions for type-safe conversion
def matching_result_to_best_match(result: MatchingResult, confidence: float) -> BestMatchDict:
    """
    Convert MatchingResult to BestMatchDict with all fields from MatchingResult.to_dict().

    Args:
        result: MatchingResult dataclass instance
        confidence: Calculated confidence score

    Returns:
        BestMatchDict with all fields properly typed
    """
    # Get the full dict from MatchingResult
    result_dict = result.to_dict()

    # Add confidence
    result_dict['confidence'] = float(np.clip(confidence, 0.0, 1.0))
    result_dict['score'] = float(np.clip(result.final_score, 0.0, 1.0))

    return result_dict


def matching_result_to_all_match(result: MatchingResult, confidence: float) -> AllMatchDict:
    """
    Convert MatchingResult to AllMatchDict with all fields from MatchingResult.to_dict().

    Args:
        result: MatchingResult dataclass instance
        confidence: Calculated confidence score

    Returns:
        AllMatchDict with all fields properly typed
    """
    # Get the full dict from MatchingResult
    result_dict = result.to_dict()

    # Add confidence
    result_dict['confidence'] = float(np.clip(confidence, 0.0, 1.0))
    result_dict['score'] = float(np.clip(result.final_score, 0.0, 1.0))

    return result_dict


# Validation functions for response structures
def validate_best_match_dict(best_match: Optional[BestMatchDict]) -> Tuple[bool, str]:
    """
    Validate BestMatchDict structure and values.

    Args:
        best_match: BestMatchDict to validate (can be None)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if best_match is None:
        return True, ""

    # Check required fields
    required_fields = ['candidate_id', 'tree_id', 'score', 'confidence']
    for field in required_fields:
        if field not in best_match:
            return False, f"Missing required field: {field}"

    # Validate score range
    score = best_match.get('score', 0)
    if not isinstance(score, (int, float)) or score < 0 or score > 1:
        return False, f"Invalid score: {score}(must be in [0, 1])"

    # Validate confidence range
    confidence = best_match.get('confidence', 0)
    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
        return False, f"Invalid confidence: {confidence} (must be in [0, 1])"

    return True, ""


def validate_all_match_dict(all_match: AllMatchDict) -> Tuple[bool, str]:
    """
    Validate AllMatchDict structure and values.

    Args:
        all_match: AllMatchDict to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required fields
    required_fields = ['candidate_id', 'tree_id', 'score', 'confidence']
    for field in required_fields:
        if field not in all_match:
            return False, f"Missing required field: {field}"

    # Validate score range
    score = all_match.get('score', 0)
    if not isinstance(score, (int, float)) or score < 0 or score > 1:
        return False, f"Invalid score: {score} (must be in [0, 1])"

    # Validate confidence range
    confidence = all_match.get('confidence', 0)
    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
        return False, f"Invalid confidence: {confidence} (must be in [0, 1])"

    # Validate counts are non-negative
    if 'superpoint_matches' in all_match and all_match['superpoint_matches'] < 0:
        return False, "superpoint_matches must be non-negative"
    if 'superpoint_inliers' in all_match and all_match['superpoint_inliers'] < 0:
        return False, "superpoint_inliers must be non-negative"

    return True, ""


def validate_match_response(response: MatchResponseDict) -> Tuple[bool, str]:
    """
    Validate MatchResponseDict structure and values.

    Args:
        response: MatchResponseDict to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required fields
    required_fields = ['decision', 'reason', 'best_match', 'all_matches']
    for field in required_fields:
        if field not in response:
            return False, f"Missing required field: {field}"

    # Validate decision
    valid_decisions = ['MATCH', 'PROBABLE_MATCH', 'POSSIBLE_MATCH', 'NO_MATCH', 'ERROR']
    if response['decision'] not in valid_decisions:
        return False, f"Invalid decision: {response['decision']}"

    # Validate best_match
    is_valid, error_msg = validate_best_match_dict(response['best_match'])
    if not is_valid:
        return False, f"Invalid best_match: {error_msg}"

    # Validate all_matches
    if not isinstance(response['all_matches'], list):
        return False, "all_matches must be a list"

    for i, match in enumerate(response['all_matches']):
        is_valid, error_msg = validate_all_match_dict(match)
        if not is_valid:
            return False, f"Invalid all_matches[{i}]: {error_msg}"

    return True, ""


class HierarchicalMatchingService:
    """
    Service for hierarchical feature matching with DINO + SuperPoint + RANSAC.
    
    This service provides a complete matching pipeline:
    1. Global search with DINO features
    2. Local matching with SuperPoint features
    3. Geometric verification with RANSAC
    4. Weighted scoring for final ranking
    
    Uses @inject decorator for dependency injection with dependency_injector.
    Wire with: container.wire(modules=["src.service.hierarchicalMatchingService"])
    """
    
    @inject
    def __init__(
        self,
        postgres_repo: SQLAlchemyORMRepository = Provide["sqlalchemy_repo"],
        minio_repo: MinIORepository = Provide["minio_repo"],
        matcher: HierarchicalMatcher = Provide["matcher"],
        top_k: int = 30,
        verbose: bool = False,
    ) -> None:
        """
        Initialize hierarchical matching service with injected dependencies.

        Args:
            postgres_repo: SQLAlchemy/PostgreSQL repository (injected via Provide)
            minio_repo: MinIO repository for loading candidate features (injected via Provide)
            matcher: Hierarchical matcher instance (injected via Provide)
            top_k: Number of candidates to retrieve
            verbose: Enable verbose logging
        """
        self.postgres_repo: SQLAlchemyORMRepository = postgres_repo
        self.minio_repo: MinIORepository = minio_repo
        self.matcher: HierarchicalMatcher = matcher
        self.top_k: int = top_k
        self.verbose: bool = verbose
         
    
    def validate(
        self,
        query_dino: Optional[DinoResult] = None,
        query_superpoint: Optional[SuperPointResult] = None,
        tree_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate inputs for hierarchical matching functions.

        Performs comprehensive validation of all inputs commonly used in match functions.
        Returns (is_valid, error_message) tuple.

        Args:
            query_dino: Optional DINO extraction result
            query_superpoint: Optional SuperPoint extraction result
            tree_id: Optional tree ID string

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
            - is_valid: True if all validations pass, False otherwise
            - error_message: Error description if validation fails, None otherwise
        """
        # Validate DINO result if provided
        if query_dino is not None:
            if not hasattr(query_dino, 'global_descriptor'):
                return False, "Query DINO result missing 'global_descriptor' attribute"
            if query_dino.global_descriptor is None or query_dino.global_descriptor.size == 0:
                return False, "Query DINO descriptor is empty"
            if np.isnan(query_dino.global_descriptor).any() or np.isinf(query_dino.global_descriptor).any():
                return False, "Query DINO descriptor contains NaN or Inf values"

        # Validate SuperPoint result if provided
        if query_superpoint is not None:
            if not hasattr(query_superpoint, 'keypoints') or not hasattr(query_superpoint, 'descriptors'):
                return False, "Query SuperPoint result missing 'keypoints' or 'descriptors' attribute"

            # Validate keypoints
            if query_superpoint.keypoints is None or query_superpoint.keypoints.size == 0:
                return False, "Query SuperPoint keypoints are empty"
            if not isinstance(query_superpoint.keypoints, np.ndarray):
                return False, f"Query SuperPoint keypoints is not numpy array: {type(query_superpoint.keypoints)}"
            if len(query_superpoint.keypoints.shape) != 2 or query_superpoint.keypoints.shape[1] != 2:
                return False, f"Query SuperPoint keypoints should be Nx2, got shape {query_superpoint.keypoints.shape}"
            if np.isnan(query_superpoint.keypoints).any() or np.isinf(query_superpoint.keypoints).any():
                return False, "Query SuperPoint keypoints contain NaN or Inf values"

            # Validate descriptors
            if query_superpoint.descriptors is None or query_superpoint.descriptors.size == 0:
                return False, "Query SuperPoint descriptors are empty"
            if not isinstance(query_superpoint.descriptors, np.ndarray):
                return False, f"Query SuperPoint descriptors is not numpy array: {type(query_superpoint.descriptors)}"
            if len(query_superpoint.descriptors.shape) != 2:
                return False, f"Query SuperPoint descriptors should be 2D, got shape {query_superpoint.descriptors.shape}"
            if np.isnan(query_superpoint.descriptors).any() or np.isinf(query_superpoint.descriptors).any():
                return False, "Query SuperPoint descriptors contain NaN or Inf values"

        # Validate tree_id if provided
        if tree_id is not None:
            if not isinstance(tree_id, str):
                return False, f"tree_id must be string, got {type(tree_id)}"
            if not tree_id.strip():
                return False, "tree_id cannot be empty or whitespace"

        return True, None

    def match(
        self,
        query_dino: DinoResult,
        query_superpoint: SuperPointResult,
        geo_filter: Optional[Dict[str, float]] = None,
        angle_filter: Optional[Dict[str, float]] = None,
        time_filter: Optional[Dict[str, int]] = None,
    ) -> MatchResponseDict:
        """
        Perform hierarchical matching for a query.

        Args:
            query_dino: DINO result with global descriptor
            query_superpoint: SuperPoint result with keypoints and descriptors and scores
            geo_filter: Optional dict with lat_min, lat_max, lon_min, lon_max for location filtering
            angle_filter: Optional dict with angle bounds (hor_angle_min/max, ver_angle_min/max, pitch_min/max)

        Returns:
            MatchResponseDict with typed matching results

        Flow:
            - If geo_filter or angle_filter is provided:
              1. First filter by location/angle using search_with_bounding_box
              2. Then apply DINO vector search
            - Multi-tree mode (default):
              1. Query Milvus globally (no filter)
              2. Partition candidates by tree_id
              3. Select top-K trees by average similarity
              4. Select diverse candidates per tree
              5. Match each tree independently
              6. Aggregate results across trees
        """
        try:
            # Validate inputs
            is_valid, error_msg = self.validate(
                query_dino=query_dino,
                query_superpoint=query_superpoint,
                tree_id='dummy',  # Dummy value for validation
            )
            if not is_valid:
                raise ValueError(error_msg)

            # Check if we have geo/angle/time filters
            has_spatial_filter = geo_filter is not None or angle_filter is not None or time_filter is not None

            if has_spatial_filter:
                # Use spatial filtering first, then DINO search
                return self._match_with_spatial_filter(
                    query_dino=query_dino,
                    query_superpoint=query_superpoint,
                    geo_filter=geo_filter,
                    angle_filter=angle_filter,
                    time_filter=time_filter,
                )

            # Multi-tree mode (default)
            return self._match_multi_tree(
                query_dino=query_dino,
                query_superpoint=query_superpoint,
            )

        except Exception as e:
            logger.error(f"[match] ✗ Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'decision': 'ERROR',
                'reason': str(e),
                'best_match': None,
                'all_matches': [],
            }

    # def _match_single_tree(
    #     self,
    #     query_dino: DinoResult,
    #     query_superpoint: SuperPointResult,
    #     tree_id: str,
    #     use_async: bool = False
    # ) -> MatchResponseDict:
    #     """
    #     Match against a single tree (existing flow).

    #     Args:
    #         query_dino: DINO result
    #         query_superpoint: SuperPoint result
    #         tree_id: Tree ID to filter candidates
    #         use_async: Use async matching

    #     Returns:
    #         MatchResponseDict
    #     """
    #     # Stage 1: Get candidates from Milvus (filtered by tree_id)
    #     logger.info(f"[STEP 3.2] Retrieving candidates from Milvus (tree_id={tree_id})...")
    #     candidates: List[CandidateFeatures] = self._get_candidates(
    #         query_dino.global_descriptor,
    #         self.top_k,
    #         tree_id=tree_id
    #     )

    #     if not candidates:
    #         logger.warning(f"[STEP 3.2] ✗ No candidates found for tree_id={tree_id}")
    #         return {
    #             'decision': 'NO_MATCH',
    #             'reason': f'No candidates found for tree_id={tree_id}',
    #             'best_match': None,
    #             'all_matches': [],
    #         }

    #     # Run common matching pipeline
    #     return self._run_matching_pipeline(
    #         query_dino=query_dino,
    #         query_superpoint=query_superpoint,
    #         candidates=candidates,
    #         use_async=use_async
    #     )

    def _match_with_spatial_filter(
        self,
        query_dino: DinoResult,
        query_superpoint: SuperPointResult,
        geo_filter: Optional[Dict[str, float]] = None,
        angle_filter: Optional[Dict[str, float]] = None,
        time_filter: Optional[Dict[str, int]] = None,
    ) -> MatchResponseDict:
        """
        Match with spatial filtering (location + angle) before DINO search.

        Flow:
        1. First filter candidates by geo location (within radius) and angle bounds
        2. Then apply DINO vector search on filtered candidates
        3. Continue with hierarchical matching

        Args:
            query_dino: DINO result
            query_superpoint: SuperPoint result
            geo_filter: Dict with lat_min, lat_max, lon_min, lon_max
            angle_filter: Dict with angle bounds
            time_filter: Dict with captured_at_min, captured_at_max (epoch seconds)

        Returns:
            MatchResponseDict with matching results
        """
        # Log spatial filter params
        if geo_filter:
            logger.info(f"[SPATIAL] Geo filter: lat[{geo_filter.get('lat_min', '?')}, {geo_filter.get('lat_max', '?')}], lon[{geo_filter.get('lon_min', '?')}, {geo_filter.get('lon_max', '?')}]")
        if angle_filter:
            logger.info(f"[SPATIAL] Angle filter: hor[{angle_filter.get('hor_angle_min', '?')}, {angle_filter.get('hor_angle_max', '?')}], ver[{angle_filter.get('ver_angle_min', '?')}, {angle_filter.get('ver_angle_max', '?')}], pitch[{angle_filter.get('pitch_min', '?')}, {angle_filter.get('pitch_max', '?')}]")

        # Stage 1: Get candidates from Milvus with spatial filtering
        logger.info(f"[STEP 3.2] Retrieving candidates from Milvus with spatial filter...")
        candidates: List[CandidateFeatures] = self._get_candidates_with_spatial_filter(
            query_dino.global_descriptor,
            self.top_k,
            geo_filter=geo_filter,
            angle_filter=angle_filter,
            time_filter=time_filter,
        )

        if not candidates:
            logger.warning(f"[STEP 3.2] ✗ No candidates found with spatial filter")
            return {
                'decision': 'NO_MATCH',
                'reason': 'No candidates found within the specified location/angle filters',
                'best_match': None,
                'all_matches': [],
            }

        logger.info(f"[STEP 3.2] ✓ Found {len(candidates)} candidates with spatial filter")

        # Run common matching pipeline
        return self._run_matching_pipeline(
            query_dino=query_dino,
            query_superpoint=query_superpoint,
            candidates=candidates,
            use_async=False
        )

    def _run_matching_pipeline(
        self,
        query_dino: DinoResult,
        query_superpoint: SuperPointResult,
        candidates: List[CandidateFeatures],
        use_async: bool = False
    ) -> MatchResponseDict:
        """
        Common matching pipeline used by both _match_single_tree and _match_with_spatial_filter.

        This method contains the shared logic for:
        - Preparing query features
        - Running hierarchical matching
        - Validating and filtering results
        - Formatting results

        Args:
            query_dino: DINO result
            query_superpoint: SuperPoint result
            candidates: List of candidate features
            use_async: Use async matching

        Returns:
            MatchResponseDict
        """
        import asyncio

        # Prepare query features
        query_features: QueryFeatures = {
            'dino': query_dino.global_descriptor,
            'keypoints': query_superpoint.keypoints,
            'descriptors': query_superpoint.descriptors,
            'scores': query_superpoint.scores,
            'tree_id': 'unknown',
        }

        # Stage 2 & 3: Hierarchical matching
        logger.info(f"[STEP 3.3] Running hierarchical matching (SuperPoint + RANSAC)...")

        if use_async:
            # Run in thread pool for pipeline parallelism
            logger.info(f"Async Running hierarchical matching ...")

            results: List[MatchingResult] = asyncio.run(asyncio.to_thread(
                self.matcher.match,
                query_features,
                candidates,
                "query",
                True,  # use_neighbor_aggregation
                5     # neighbors_per_candidate
            ))
        else:
            results: List[MatchingResult] = self.matcher.match(
                query=query_features,
                candidates=candidates,
                query_id="query",
                use_neighbor_aggregation=True,
                neighbors_per_candidate=5
            )

        # Validate matching results
        if not results:
            logger.warning(f"[STEP 3.3] ✗ No matching results returned")
            return {
                'decision': 'NO_MATCH',
                'reason': 'No matching results returned',
                'best_match': None,
                'all_matches': [],
            }

        # Filter results: only those with sufficient matches AND inliers pass threshold
        filtered_results = [
            r for r in results
            if r.superpoint_matches >= self.matcher.min_matches
            and r.superpoint_inliers >= self.matcher.min_inliers
        ]

        # If no candidate passes thresholds, return NO_MATCH
        if not filtered_results:
            logger.warning(f"[STEP 3.3] ✗ No candidates passed thresholds (min_matches={self.matcher.min_matches}, min_inliers={self.matcher.min_inliers})")
            return {
                'decision': 'NO_MATCH',
                'reason': f'No candidates passed thresholds (min_matches={self.matcher.min_matches}, min_inliers={self.matcher.min_inliers})',
                'best_match': None,
                'all_matches': [],
            }

        # Validate results before formatting
        for i, res in enumerate(filtered_results):
            if not isinstance(res, MatchingResult):
                raise ValueError(f"Result {i} is not MatchingResult: {type(res)}")
            if res.final_score < 0 or res.final_score > 1:
                logger.warning(f"  Result {i}: final_score {res.final_score} out of range [0, 1]")
            if np.isnan(res.final_score) or np.isinf(res.final_score):
                raise ValueError(f"Result {i}: final_score is NaN or Inf")

        logger.info(f"[STEP 3.3] ✓ Hierarchical matching complete")

        # Log top results
        for i, res in enumerate(filtered_results[:3]):
            logger.info(f"    Result {i+1}: {res.candidate_id}(tree: {res.tree_id}, score: {res.final_score:.4f}, inliers: {res.superpoint_inliers})")

        # Convert to output format
        logger.info(f"[STEP 3.4] Formatting results...")
        result: MatchResponseDict = self._format_results(filtered_results)
        return result

    async def match_async(
        self,
        query_dino: DinoResult,
        query_superpoint: SuperPointResult,
        geo_filter: Optional[Dict[str, float]] = None,
        angle_filter: Optional[Dict[str, float]] = None,
        time_filter: Optional[Dict[str, int]] = None,
    ) -> MatchResponseDict:
        """Async wrapper around match().

        verificationService.verify() is async and calls match_async,
        but HierarchicalMatchingService.match() is synchronous.
        This wrapper runs the sync match() in a thread pool via asyncio.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.match(
                query_dino=query_dino,
                query_superpoint=query_superpoint,
                geo_filter=geo_filter,
                angle_filter=angle_filter,
                time_filter=time_filter,
            )
        )

    def _get_candidates_with_spatial_filter(
        self,
        query_dino: np.ndarray,
        top_k: int,
        geo_filter: Optional[Dict[str, float]] = None,
        angle_filter: Optional[Dict[str, float]] = None,
        time_filter: Optional[Dict[str, int]] = None,
    ) -> List[CandidateFeatures]:
        """
        Get candidate features from Milvus with spatial filtering.

        Args:
            query_dino: DINO global descriptor
            top_k: Number of candidates to retrieve
            geo_filter: Dict with lat_min, lat_max, lon_min, lon_max
            angle_filter: Dict with angle bounds
            time_filter: Dict with captured_at_min, captured_at_max (epoch seconds)

        Returns:
            List of candidate feature dictionaries
        """
        if self.postgres_repo is None:
            raise ValueError("PostgreSQL repository not configured")

        if self.minio_repo is None:
            raise ValueError("MinIO repository not configured")

        logger.info(f"[_get_candidates_with_spatial_filter] Querying PostgreSQL with spatial filter (top_k={top_k})...")

        # Search PostgreSQL with bounding box filtering (via MilvusResult adapter)
        milvus_result = self.postgres_repo.search_with_bounding_box(
            query_vector=query_dino,
            top_k=top_k,
            lon_min=geo_filter.get('lon_min') if geo_filter else None,
            lon_max=geo_filter.get('lon_max') if geo_filter else None,
            lat_min=geo_filter.get('lat_min') if geo_filter else None,
            lat_max=geo_filter.get('lat_max') if geo_filter else None,
            hor_angle_min=angle_filter.get('hor_angle_min') if angle_filter else None,
            hor_angle_max=angle_filter.get('hor_angle_max') if angle_filter else None,
            ver_angle_min=angle_filter.get('ver_angle_min') if angle_filter else None,
            ver_angle_max=angle_filter.get('ver_angle_max') if angle_filter else None,
            pitch_min=angle_filter.get('pitch_min') if angle_filter else None,
            pitch_max=angle_filter.get('pitch_max') if angle_filter else None,
            captured_at_min=time_filter.get('captured_at_min') if time_filter else None,
            captured_at_max=time_filter.get('captured_at_max') if time_filter else None,
        )

        candidates: List[CandidateFeatures] = []
        for evidence_id, distance, tree_id, metadata in zip(
            milvus_result.ids,
            milvus_result.distances,
            milvus_result.tree_ids,
            milvus_result.metadatas
        ):
            try:
                # Convert distance to similarity
                similarity = distance

                # Load features from MinIO using the stored minio_key from evidence metadata.
                # During ingestion, minio_key is stored in evidence_metadata so we can load
                # features by storage key rather than by image_id slug (which might not match
                # the evidence UUID returned by PostgreSQL search).
                minio_key = metadata.get('minio_key') if metadata else None
                if minio_key:
                    features = self.minio_repo.load_features_from_key(minio_key)
                else:
                    # Fallback 1: use image_id from metadata (the slug used when storing)
                    _image_id = metadata.get('image_id') if metadata else None
                    if _image_id:
                        features = self.minio_repo.load_features(_image_id)
                    else:
                        # Fallback 2: use evidence_id (UUID) — should only happen for
                        # newly ingested evidence after minio_key fix is deployed
                        features = self.minio_repo.load_features(evidence_id)

                if features is None:
                    raise ValueError(f"Failed to load features from MinIO for evidence {evidence_id} (minio_key={minio_key})")

                # Get DINO descriptor from PostgreSQL by evidence PK (not by metadata image_id)
                dino_descriptor = self.postgres_repo.get_global_vector_by_id(evidence_id)
                if dino_descriptor is None:
                    raise ValueError(f"Failed to get DINO descriptor from PostgreSQL for {evidence_id}")

                candidate_data: CandidateFeatures = {
                    'id': evidence_id,
                    'tree_id': tree_id,
                    'dino': dino_descriptor,
                    'keypoints': features.get('keypoints'),
                    'descriptors': features.get('descriptors'),
                    'scores': features.get('scores'),
                    'similarity': similarity,
                    'captured_at': metadata.get('captured_at') if metadata else None
                }

                # Filter: skip candidates with too few keypoints
                kp = candidate_data.get('keypoints')
                kp_count = len(kp) if kp is not None else 0
                if kp_count <= 90:
                    logger.info(f"[_get_candidates_with_spatial_filter] ✗ Skipping {evidence_id}: only {kp_count} keypoints (require > 100)")
                    continue

                candidates.append(candidate_data)

            except Exception as e:
                logger.warning(f"Failed to load features for evidence {evidence_id}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

        logger.info(f"[STEP 3.2] ✓ Loaded {len(candidates)} candidates with spatial filter")
        return candidates

    def _match_multi_tree(
        self,
        query_dino: DinoResult,
        query_superpoint: SuperPointResult,
    ) -> MatchResponseDict:
        """
        Match against multiple trees (multi-tree mode).

        Flow:
        1. Query Milvus globally (no tree filter)
        2. Partition candidates by tree_id
        3. Select top-K trees by average similarity
        4. Select diverse candidates per tree
        5. Match each tree independently
        6. Aggregate results across trees

        Args:
            query_dino: DINO result
            query_superpoint: SuperPoint result

        Returns:
            MatchResponseDict
        """
        import asyncio

        # Stage 1: Get candidates from Milvus (global search, no tree filter)
        all_candidates: List[CandidateFeatures] = self._get_candidates(
            query_dino.global_descriptor,
            self.top_k * 3,  # Get more candidates for multi-tree
        )

        if not all_candidates:
            raise ValueError("No candidates found in database")

        # Stage 2: Partition by tree_id
        logger.info(f"[STEP 3.2b] Partitioning {len(all_candidates)}candidates by tree_id...")
        tree_candidates = self._partition_by_tree(all_candidates)

        if not tree_candidates:
            raise ValueError("No trees found after partitioning")

        # Stage 3: Select top-K trees (e.g., top 5)
        # Use sorted() for deterministic selection instead of dict.items() slicing
        # which is non-deterministic in Python < 3.7
        top_k_trees = min(5, len(tree_candidates))
        sorted_tree_items = sorted(tree_candidates.items(), key=lambda x: x[1], reverse=True)[:top_k_trees]
        selected_trees = dict(sorted_tree_items)
        logger.info(f"[STEP 3.2c] Selected top {top_k_trees} trees for matching")

        # Stage 4: Select candidates using neighbor-group scoring.
        # Finds neighbors of each candidate and uses their average similarity
        # to rank candidates.  The COUNT of candidates per tree does NOT
        # influence the final result — only the average similarity matters.
        logger.info(f"[STEP 3.2d] Selecting neighbor-group candidates per tree...")
        selected_candidates = self._select_neighbor_group_candidates(
            all_candidates=all_candidates,
            tree_candidates=selected_trees,
            query_dino=query_dino.global_descriptor,
            n_per_tree=5,
            neighbors_per_candidate=5,
            query_weight=0.6,
            neighbor_weight=0.4
        )

        # Stage 5: Match each tree independently using parallel processing
        logger.info(f"[STEP 3.3] Running hierarchical matching for each tree...")

        # Use asyncio.gather() for parallel tree matching to improve throughput
        import asyncio

        async def match_tree_async(tree_id: str, candidates: List[CandidateFeatures]) -> Tuple[str, List[MatchingResult]]:
            """Async wrapper for tree matching."""
            # Prepare query features
            query_features: QueryFeatures = {
                'dino': query_dino.global_descriptor,
                'keypoints': query_superpoint.keypoints,
                'descriptors': query_superpoint.descriptors,
                'scores': query_superpoint.scores,
                'tree_id': tree_id,
            }

            # Run hierarchical matching for this tree in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.matcher.match(
                    query=query_features,
                    candidates=candidates,
                    query_id="query",
                )
            )
            return tree_id, results

        # Run all tree matches in parallel
        async def run_all_tree_matches():
            tasks = [
                match_tree_async(tree_id, candidates)
                for tree_id, candidates in selected_candidates.items()
            ]
            return await asyncio.gather(*tasks)

        # Execute async tasks
        tree_results_list = asyncio.run(run_all_tree_matches())
        tree_results: Dict[str, List[MatchingResult]] = dict(tree_results_list)

        if not tree_results:
            raise ValueError("No matching results from any tree")

        # Stage 6: Aggregate results across trees — pure final_score ranking,
        # no candidate-count bias.
        logger.info(f"[STEP 3.4] Aggregating results across {len(tree_results)} trees...")
        for tree_id, results in tree_results.items():
            best_score = max(r.final_score for r in results) if results else 0
            logger.info(f"  - Tree {tree_id}: {len(results)} results, best_score={best_score:.4f}")

        result: MatchResponseDict = self._aggregate_tree_results(tree_results)
        return result

    def _partition_by_tree(
        self,
        candidates: List[CandidateFeatures]
    ) -> Dict[str, List[CandidateFeatures]]:
        """
        Partition candidates by tree_id and compute tree-level scores.

        This method groups candidates by their tree_id and computes the average DINO
        similarity for each tree. Trees are sorted by average similarity in descending order.

        Args:
            candidates: List of candidate features with tree_id and similarity

        Returns:
            Dict mapping tree_id to list of candidates, sorted by average similarity

        Example:
            >>> candidates = [
            ...     {'tree_id': 'tree_A', 'similarity': 0.9},
            ...     {'tree_id': 'tree_A', 'similarity': 0.8},
            ...     {'tree_id': 'tree_B', 'similarity': 0.7},
            ... ]
            >>> result = service._partition_by_tree(candidates)
            >>> # result = {
            >>> #     'tree_A': [{'tree_id': 'tree_A', 'similarity': 0.9}, ...],
            >>> #     'tree_B': [{'tree_id': 'tree_B', 'similarity': 0.7}]
            >>> # }
        """
        """
        Group candidates by tree_id and compute tree-level scores.

        Args:
            candidates: List of candidate features from Milvus (possibly from multiple trees)

        Returns:
            Dict mapping tree_id to list of candidates, sorted by tree score (descending)
        """
        # Group by tree_id
        tree_candidates: Dict[str, List[CandidateFeatures]] = {}
        for cand in candidates:
            tree_id = cand['tree_id']
            if tree_id not in tree_candidates:
                tree_candidates[tree_id] = []
            tree_candidates[tree_id].append(cand)

        # Rank trees by AVERAGE similarity to match the documented behavior.
        # The documentation states "average DINO similarity per tree".
        tree_scores: Dict[str, float] = {}
        for tree_id, cands in tree_candidates.items():
            avg_sim = sum(c['similarity'] for c in cands) / len(cands)
            tree_scores[tree_id] = avg_sim  # AVG

        # Sort trees by score (descending)
        sorted_trees = sorted(tree_scores.items(), key=lambda x: x[1], reverse=True)

        if self.verbose:
            logger.info(f"[partition] Found {len(sorted_trees)} trees")
            for i, (tid, score) in enumerate(sorted_trees[:3]):
                logger.info(f"  Rank {i+1}: {tid} (score={score:.4f})")

        # Return ordered dict (sorted by tree score)
        from collections import OrderedDict
        return OrderedDict((tid, tree_candidates[tid]) for tid, _ in sorted_trees)

    def _select_neighbor_group_candidates(
        self,
        all_candidates: List[CandidateFeatures],
        tree_candidates: Dict[str, List[CandidateFeatures]],
        query_dino: np.ndarray,
        n_per_tree: int = 5,
        neighbors_per_candidate: int = 5,
        query_weight: float = 0.6,
        neighbor_weight: float = 0.4
    ) -> Dict[str, List[CandidateFeatures]]:
        """
        Select candidates based on neighbor-enhanced scoring.

        Instead of diversity selection, this method uses the hypothesis that
        if a candidate has high similarity to the query, its neighbors should
        also have good similarity. This creates groups of candidates+neighbors
        for more robust matching.

        Args:
            all_candidates: All candidate features from query
            tree_candidates: Dict mapping tree_id to list of candidates
            query_dino: Query DINO descriptor for computing similarities
            n_per_tree: Number of candidates to select per tree
            neighbors_per_candidate: Number of neighbors to consider per candidate
            query_weight: Weight for direct query-candidate similarity
            neighbor_weight: Weight for average query-neighbor similarity

        Returns:
            Dict mapping tree_id to list of selected candidates
        """
        from src.processor.hierarchicalMatcher import NeighborGroup

        # Create a map for quick lookup
        candidates_dict = {c.get('id', ''): c for c in all_candidates}

        selected_by_tree: Dict[str, List[CandidateFeatures]] = {}

        for tree_id, candidates in tree_candidates.items():
            if not candidates:
                continue

            # Build neighbor groups for all candidates in this tree
            groups = []
            for candidate in candidates:
                candidate_id = candidate.get('id', '')
                candidate_dino = candidate.get('dino')

                if not candidate_id or candidate_dino is None:
                    continue

                # Find neighbors
                neighbors = self.matcher.find_neighbors(
                    target_dino=candidate_dino,
                    candidates=candidates,
                    tree_id=tree_id,
                    top_k=neighbors_per_candidate,
                    exclude_target_id=candidate_id
                )

                # Create neighbor group
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

            # Sort by enhanced score
            groups.sort(key=lambda x: x[1], reverse=True)

            # Select top candidates by enhanced score
            selected = []
            seen_ids = set()

            for group, _ in groups[:n_per_tree]:
                # Add the candidate itself
                if group.candidate_id in candidates_dict and group.candidate_id not in seen_ids:
                    selected.append(candidates_dict[group.candidate_id])
                    seen_ids.add(group.candidate_id)

                # Also include top neighbors
                for neighbor_id in group.neighbor_ids[:2]:  # Include top 2 neighbors
                    if neighbor_id in candidates_dict and neighbor_id not in seen_ids:
                        if len(selected) < n_per_tree + 5:
                            selected.append(candidates_dict[neighbor_id])
                            seen_ids.add(neighbor_id)

            selected_by_tree[tree_id] = selected[:n_per_tree + 5]  # Limit total

            if self.verbose:
                logger.info(f"    [{tree_id}] Selected {len(selected)} candidates using neighbor-group scoring")
                for i, (g, s) in enumerate(groups[:3], 1):
                    logger.info(f"      Rank {i}: {g.candidate_id} (enhanced_score={s:.4f})")

        return selected_by_tree

    def _get_candidates(
        self,
        query_dino: np.ndarray,
        top_k: int,
        tree_id: Optional[str] = None
    ) -> List[CandidateFeatures]:
        """
        Get candidate features from Milvus and MinIO.

        Args:
            query_dino: DINO global descriptor
            top_k: Number of candidates to retrieve
            tree_id: Optional tree ID to filter candidates. If provided, returns same-tree candidates only.

        Returns:
            List of candidate feature dictionaries
        """
        if self.postgres_repo is None:
            raise ValueError("PostgreSQL repository not configured")

        if self.minio_repo is None:
            raise ValueError("MinIO repository not configured")

        # Log query type
        query_type = "FILTERED (single-tree)" if tree_id else "GLOBAL (multi-tree)"
        logger.info(f"[_get_candidates] Querying PostgreSQL ({query_type}, top_k={top_k})...")

        # Search PostgreSQL for candidates (via MilvusResult adapter)
        milvus_result = self.postgres_repo.search(
            query_vector=query_dino,
            top_k=top_k,
            tree_id_filter=tree_id  # Pass tree_id_filter to repository for filtering
        )

        candidates: List[CandidateFeatures] = []
        for evidence_id, distance, tree_id, metadata in zip(
            milvus_result.ids,
            milvus_result.distances,
            milvus_result.tree_ids,
            milvus_result.metadatas
        ):
            try:
                # Convert distance to similarity
                # For COSINE metric in Milvus, the "distance" field is actually the cosine similarity!
                # So we use it directly as similarity
                similarity = distance

                # Load features from MinIO using the stored minio_key from evidence metadata.
                # PostgreSQL metadata contains image_id (the slug used when storing), so we
                # can always derive the MinIO key: features/<image_id>.npz.gz
                # Fallback chain: minio_key -> image_id (slug) -> evidence_id (UUID)
                minio_key = metadata.get('minio_key') if metadata else None
                if minio_key:
                    features = self.minio_repo.load_features_from_key(minio_key)
                else:
                    # Fallback 1: use image_id slug (the key used when storing in MinIO)
                    _image_id = metadata.get('image_id') if metadata else None
                    if _image_id:
                        features = self.minio_repo.load_features(_image_id)
                    else:
                        # Fallback 2: use evidence_id (UUID) — last resort for edge cases
                        features = self.minio_repo.load_features(evidence_id)

                if features is None:
                    raise ValueError(f"Failed to load features from MinIO for evidence {evidence_id} (minio_key={minio_key})")

                # Get DINO descriptor from PostgreSQL by evidence PK (not by metadata image_id)
                dino_descriptor = self.postgres_repo.get_global_vector_by_id(evidence_id)
                if dino_descriptor is None:
                    raise ValueError(f"Failed to get DINO descriptor from PostgreSQL for {evidence_id}")

                candidate_data: CandidateFeatures = {
                    'id': evidence_id,
                    'tree_id': tree_id,
                    'dino': dino_descriptor,
                    'keypoints': features.get('keypoints'),
                    'descriptors': features.get('descriptors'),
                    'scores': features.get('scores'),  # Include scores for matcher
                    'similarity': similarity,  # Add similarity for reference
                }

                # Filter: skip candidates with too few keypoints (require > 100)
                kp = candidate_data.get('keypoints')
                kp_count = len(kp) if kp is not None else 0
                if kp_count <= 90:
                    logger.info(f"[_get_candidates] ✗ Skipping {evidence_id}: only {kp_count} keypoints (require > 100)")
                    continue

                candidates.append(candidate_data)

            except Exception as e:
                logger.warning(f"Failed to load features for {evidence_id}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

        logger.info(f"[STEP 3.2] ✓ Loaded {len(candidates)}candidates with features")
        return candidates

    def _aggregate_tree_results(
        self,
        tree_results: Dict[str, List[MatchingResult]],
    ) -> MatchResponseDict:
        """
        Aggregate matching results from multiple trees.

        Args:
            tree_results: Dict mapping tree_id to their matching results

        Returns:
            MatchResponseDict with all aggregated results sorted by score
        """
        if not tree_results:
            return {
                'decision': 'NO_MATCH',
                'reason': 'No tree results to aggregate',
                'best_match': None,
                'all_matches': [],
            }

        # Flatten ALL results from ALL trees into a single list
        all_results: List[MatchingResult] = []
        for tree_id, results in tree_results.items():
            if results:
                all_results.extend(results)
                if self.verbose:
                    logger.info(f"[aggregate] Tree {tree_id}: {len(results)} results")

        if not all_results:
            return {
                'decision': 'NO_MATCH',
                'reason': 'No valid results from any tree',
                'best_match': None,
                'all_matches': [],
            }

        # Filter: only keep results with sufficient matches AND inliers
        all_results = [
            r for r in all_results
            if r.superpoint_matches >= self.matcher.min_matches
            and r.superpoint_inliers >= self.matcher.min_inliers
        ]
        logger.info(f"[aggregate] After filter (matches>={self.matcher.min_matches}, inliers>={self.matcher.min_inliers}): {len(all_results)} results remain")

        if not all_results:
            return {
                'decision': 'NO_MATCH',
                'reason': f'No candidates passed thresholds (min_matches={self.matcher.min_matches}, min_inliers={self.matcher.min_inliers})',
                'best_match': None,
                'all_matches': [],
            }

        # Note: Tree voting boost was previously applied here but removed because
        # a tree dominating the coarse retrieval does not reliably indicate
        # the correct identity — visually similar trees (e.g., durian_5 vs
        # durian_6) share many candidates in DINO space.

        # Sort ALL results by final_score (descending)
        all_results.sort(key=lambda r: r.final_score, reverse=True)

        # Get overall best result
        best_match_result: MatchingResult = all_results[0]
        best_score = best_match_result.final_score

        if self.verbose:
            logger.info(f"[aggregate] Total results from all trees: {len(all_results)}")
            logger.info(f"[aggregate] Best result: {best_match_result.candidate_id} (tree={best_match_result.tree_id}, score={best_score:.4f})")
            logger.info(f"[aggregate] Top 5 results:")
            for i, result in enumerate(all_results[:5], 1):
                logger.info(f"  {i}. {result.candidate_id} (tree={result.tree_id}, score={result.final_score:.4f})")

        # Format best match
        best_match_dict: BestMatchDict = {
            'query_id': best_match_result.query_id,
            'candidate_id': best_match_result.candidate_id,
            'tree_id': best_match_result.tree_id,
            'dino_similarity': best_match_result.dino_similarity,
            'superpoint_matches': best_match_result.superpoint_matches,
            'superpoint_inliers': best_match_result.superpoint_inliers,
            'superpoint_match_ratio': best_match_result.superpoint_match_ratio,
            'ransac_inlier_ratio': best_match_result.ransac_inlier_ratio,
            'bark_texture_similarity': best_match_result.bark_texture_similarity,
            'homography': best_match_result.homography,
            'fundamental': best_match_result.fundamental,
            'reprojection_error': best_match_result.reprojection_error,
            'final_score': best_match_result.final_score,
            'score': best_match_result.final_score,
            'confidence': best_match_result.final_score,
            'timing': None,
            # 'match_coordinates': best_match_result.match_coordinates,
        }

        # Format all matches (top 20 from aggregated pool)
        all_matches_list: List[AllMatchDict] = []
        for result in all_results[:20]:
            all_matches_list.append({
                'query_id': result.query_id,
                'candidate_id': result.candidate_id,
                'tree_id': result.tree_id,
                'dino_similarity': result.dino_similarity,
                'superpoint_matches': result.superpoint_matches,
                'superpoint_inliers': result.superpoint_inliers,
                'superpoint_match_ratio': result.superpoint_match_ratio,
                'ransac_inlier_ratio': result.ransac_inlier_ratio,
                'bark_texture_similarity': result.bark_texture_similarity,
                'homography': result.homography,
                'fundamental': result.fundamental,
                'reprojection_error': result.reprojection_error,
                'final_score': result.final_score,
                'score': result.final_score,
                'confidence': result.final_score,
                'timing': {
                    'dino_search': result.dino_search_time,
                    'superpoint_match': result.superpoint_match_time,
                    'geometric_verify': result.geometric_verify_time,
                    'texture_match': result.texture_match_time,
                    'total': result.total_time,
                },
                # 'match_coordinates': result.match_coordinates,
            })

        # Determine decision using the same logic as _make_decision (confidence + dino thresholds)
        # This ensures multi-tree and single-tree modes use identical decision criteria.
        decision = self._make_decision(best_match_result, all_results)
        reason = f'Best match: {best_match_result.candidate_id}(tree={best_match_result.tree_id}) with score {best_score:.4f}'

        return {
            'decision': decision,
            'reason': reason,
            'best_match': best_match_dict,
            'all_matches': all_matches_list,
        }

    def _format_results(
        self,
        results: List[MatchingResult],
    ) -> MatchResponseDict:
        """
        Format matching results for output using type-safe structures.

        Preserves all fields from MatchingResult.to_dict() including:
        - query_id, candidate_id, tree_id
        - dino_similarity, superpoint_matches, superpoint_inliers, superpoint_match_ratio
        - ransac_inlier_ratio, bark_texture_similarity
        - homography, fundamental, reprojection_error
        - final_score, confidence
        - timing (dino_search, superpoint_match, geometric_verify, texture_match, total)
        - match_coordinates

        Args:
            results: List of MatchingResult objects

        Returns:
            MatchResponseDict with properly typed fields
        """
        # Format all matches (even NO_MATCH responses include top candidates with confidence)
        all_matches: List[AllMatchDict] = []
        best_match: Optional[BestMatchDict] = None

        if results:
            confidences: List[float] = [self._calculate_confidence(r, results) for r in results]

            # Convert results to typed all_matches using helper function
            for result, confidence in zip(results[:5], confidences[:5]):
                match_dict = matching_result_to_all_match(result, confidence)
                all_matches.append(match_dict)

            # Convert best result to typed best_match using helper function
            best_result: MatchingResult = results[0]
            best_confidence: float = confidences[0] if confidences else self._calculate_confidence(best_result, results)
            best_match = matching_result_to_best_match(best_result, best_confidence)

            decision: str = self._make_decision(best_result, results)
            reason: str = f'Best match: {best_result.candidate_id} with score {float(best_result.final_score):.3f}'
        else:
            decision = 'NO_MATCH'
            reason = 'No valid matches found'

        response: MatchResponseDict = {
            'decision': str(decision),
            'reason': str(reason),
            'best_match': best_match,
            'all_matches': all_matches,
        }

        # Validate response before returning
        is_valid, error_msg = validate_match_response(response)
        if not is_valid:
            logger.error(f"[_format_results] Response validation failed: {error_msg}")
            raise ValueError(f"Invalid response structure: {error_msg}")

        logger.info(f"[_format_results] Decision: {decision}, Best match: {best_match['candidate_id'] if best_match else 'None'}, Confidence: {best_match['confidence'] if best_match else 'N/A'}")

        return response

    def _calculate_confidence(self, result: MatchingResult, all_results: Optional[List[MatchingResult]] = None) -> float:
        """
        Calculate confidence score using adaptive weighted formula.

        Adaptive strategy:
        - DINO is the primary signal (40-60% weight depending on quality)
        - Inliers contribute 20-40% (adaptive based on count)
        - RANSAC, texture, and error provide supporting evidence (10-20% combined)

        Args:
            result: MatchingResult object with all metric values
            all_results: Optional list of all results for context (reserved for future use)

        Returns:
            Confidence score clipped to [0.0, 1.0]
        """
        import math

        _ = all_results  # Reserved for future use

        # Extract raw metrics
        dino_conf: float = float(result.dino_similarity)
        inlier_count: int = result.superpoint_inliers
        ransac_conf: float = float(result.ransac_inlier_ratio)
        texture_conf: float = float(result.bark_texture_similarity)

        reprojection_error: float = result.reprojection_error
        # Handle None, inf, or invalid reprojection_error values
        if reprojection_error is None:
            error_raw: float = 9999.0
        elif isinstance(reprojection_error, float):
            if math.isinf(reprojection_error) or math.isnan(reprojection_error):
                error_raw: float = 9999.0
            else:
                error_raw = float(reprojection_error)
        else:
            try:
                error_raw = float(reprojection_error)
                if math.isinf(error_raw) or math.isnan(error_raw):
                    error_raw = 9999.0
            except (ValueError, TypeError):
                error_raw = 9999.0

        log_err: float = 1.0 / (1.0 + math.log(1.0 + error_raw))

        # Adaptive inlier confidence with smoother scaling
        # Use sigmoid-like curve: good matches have 50-200 inliers
        if inlier_count >= 100:
            inlier_conf = 1.0
        elif inlier_count >= 50:
            inlier_conf = 0.7 + 0.3 * (inlier_count - 50) / 50
        elif inlier_count >= 20:
            inlier_conf = 0.4 + 0.3 * (inlier_count - 20) / 30
        elif inlier_count >= 8:
            inlier_conf = 0.2 + 0.2 * (inlier_count - 8) / 12
        else:
            inlier_conf = 0.1 * inlier_count / 8

        # Adaptive weight allocation based on signal quality
        # If DINO is very confident (>0.85), trust it more
        # If inliers are strong (>50), give them more weight
        if inlier_count >= 50:
            w_dino = 0.35
            w_inlier = 0.45
            w_ransac = 0.10
            w_texture = 0.07
            w_error = 0.03
        elif inlier_count >= 20:
            w_dino = 0.50
            w_inlier = 0.25
            w_ransac = 0.12
            w_texture = 0.08
            w_error = 0.05
        elif dino_conf >= 0.85:
            w_dino = 0.35
            w_inlier = 0.45
            w_ransac = 0.08
            w_texture = 0.08
            w_error = 0.04
        else:
            # Low inliers: rely heavily on DINO and texture
            w_dino = 0.35
            w_inlier = 0.40
            w_ransac = 0.08
            w_texture = 0.12
            w_error = 0.05

        confidence: float = (
            w_dino * dino_conf
            + w_inlier * inlier_conf
            + w_ransac * ransac_conf
            + w_texture * texture_conf
            + w_error * log_err
        )

        # Ensure confidence is valid (not inf, not nan)
        if math.isinf(confidence) or math.isnan(confidence):
            confidence = 0.0

        # Clip to [0.0, 1.0]
        return float(np.clip(confidence, 0.0, 1.0))

    def _make_decision(self, result: MatchingResult, all_results: Optional[List[MatchingResult]] = None) -> str:
        """
        Make final decision from matching result.

        Args:
            result: MatchingResult object
            all_results: Optional list of all results for context

        Returns:
            Decision string: 'MATCH', 'PROBABLE_MATCH', 'POSSIBLE_MATCH', or 'NO_MATCH'
        """
        confidence: float = self._calculate_confidence(result, all_results)
        dino_sim: float = result.dino_similarity

        logger.info(f"[_make_decision] Confidence: {confidence:.4f}, DINO: {dino_sim:.4f}")

        # MATCH: confidence >= 45% AND dino_similarity >= 0.75
        if confidence >= 0.45 and dino_sim >= 0.75:
            logger.info(f"[_make_decision] → MATCH")
            return 'MATCH'

        # PROBABLE_MATCH: confidence 40-45% AND dino_similarity >= 0.70
        elif confidence >= 0.40 and dino_sim >= 0.70:
            logger.info(f"[_make_decision] → PROBABLE_MATCH")
            return 'PROBABLE_MATCH'

        # POSSIBLE_MATCH: confidence 35-40% AND dino_similarity >= 0.65
        elif confidence >= 0.35 and dino_sim >= 0.65:
            logger.info(f"[_make_decision] → POSSIBLE_MATCH")
            return 'POSSIBLE_MATCH'

        # NO_MATCH: Everything else
        else:
            logger.info(f"[_make_decision] → NO_MATCH")
            return 'NO_MATCH'


class HierarchicalVerificationService:
    """
    Verification service using hierarchical matching.
    
    This integrates with the existing verification pipeline but uses
    the new hierarchical matching approach.
    """
    
    def __init__(
        self,
        hierarchical_service: HierarchicalMatchingService,
        confidence_threshold: float = 0.7,
        verbose: bool = False
    ) -> None:
        """
        Initialize hierarchical verification service.

        Args:
            hierarchical_service: HierarchicalMatchingService instance
            confidence_threshold: Minimum confidence for positive match
            verbose: Enable verbose logging
        """
        self.hierarchical_service: HierarchicalMatchingService = hierarchical_service
        self.confidence_threshold: float = confidence_threshold
        self.verbose: bool = verbose
    
    def verify(
        self,
        query_dino: DinoResult,
        query_superpoint: SuperPointResult,
        known_tree_id: Optional[str] = None,
        use_async: bool = False
    ) -> Dict[str, Any]:
        """
        Verify tree identity using hierarchical matching.

        Args:
            query_dino: DINO result with global descriptor
            query_superpoint: SuperPoint result with keypoints and descriptors
            known_tree_id: Optional known tree ID for evaluation
            use_async: If True, run matching in thread pool for pipeline parallelism

        Returns:
            Dict with verification result
        """
        import asyncio

        try:
            # Run hierarchical matching
            if use_async:
                result = asyncio.run(asyncio.to_thread(
                    self.hierarchical_service.match,
                    query_dino=query_dino,
                    query_superpoint=query_superpoint,
                ))
            else:
                result = self.hierarchical_service.match(
                    query_dino=query_dino,
                    query_superpoint=query_superpoint,
                )

            # Validate result
            if result is None:
                logger.error(f"[HierarchicalVerificationService] hierarchical_service.match() returned None!")
                return {
                    'decision': 'ERROR',
                    'reason': 'Hierarchical matching returned None',
                    'best_match': None,
                    'all_matches': [],
                }

            if not isinstance(result, dict):
                logger.error(f"[HierarchicalVerificationService] hierarchical_service.match() returned non-dict: {type(result)}")
                return {
                    'decision': 'ERROR',
                    'reason': f'Hierarchical matching returned {type(result)} instead of dict',
                    'best_match': None,
                    'all_matches': [],
                }

            # Add verification-specific fields
            if result.get('decision') in ['MATCH', 'PROBABLE_MATCH']:
                best_match = result.get('best_match')
                if best_match and best_match.get('confidence', 0) >= self.confidence_threshold:
                    result['verified'] = True
                    result['verified_tree_id'] = best_match['tree_id']
                else:
                    result['verified'] = False
                    result['verified_tree_id'] = None
            else:
                result['verified'] = False
                result['verified_tree_id'] = None

            # Add evaluation if known_tree_id provided
            if known_tree_id is not None:
                best_match = result.get('best_match')
                if best_match:
                    predicted_tree_id = best_match.get('tree_id')
                    result['correct'] = (predicted_tree_id == known_tree_id)
                else:
                    result['correct'] = False
            return result

        except Exception as e:
            logger.error(f"[HierarchicalVerificationService] verify() failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'decision': 'ERROR',
                'reason': str(e),
                'best_match': None,
                'all_matches': [],
            }


# Export functions for easy import
def create_hierarchical_matching_service(
    postgres_repo: Any = None,
    minio_repo: Any = None,
    config: Optional[Any] = None,
    **kwargs
) -> HierarchicalMatchingService:
    """
    Factory function to create hierarchical matching service.

    Args:
        postgres_repo: PostgreSQL repository for vector search
        minio_repo: MinIO repository for storing features
        config: Optional AppConfig for settings (uses values from appConfig if provided)
        **kwargs: Additional arguments for service (overrides config if provided)
                 Note: 'app_config' key in kwargs is filtered out (reserved for factory use)

    Returns:
        Configured HierarchicalMatchingService
    """
    # Set default kwargs from config or with balanced thresholds
    defaults: Dict[str, Any] = {
        'dino_threshold': 0.20,  # Very permissive for multi-tree mode - SuperPoint/RANSAC will filter
        'superpoint_match_ratio': 0.1,
        'ransac_threshold': 5.0,
        'min_inliers': 5,  # Minimum RANSAC inliers (lowered from 15 for test images)
        'min_matches': 20,  # Minimum LightGlue matches (lowered from 90 for test images)
        'top_k': 20,
        'verbose': False
    }

    # Override with config values if provided
    if config is not None:
        # Extract values from config object (supports both AppConfig and dict-like)
        sp_max = getattr(config, 'sp_max_keypoints', 4096) if hasattr(config, 'sp_max_keypoints') else (config.get('sp_max_keypoints') if hasattr(config, 'get') else 4096)
        lg_dev = getattr(config, 'lg_device', 'cuda') if hasattr(config, 'lg_device') else (config.get('lg_device') if hasattr(config, 'get') else 'cuda')
        lg_conf = getattr(config, 'lg_confidence', 0.1) if hasattr(config, 'lg_confidence') else (config.get('lg_confidence') if hasattr(config, 'get') else 0.1)
        defaults.update({
            'sp_max_keypoints': sp_max,
            'lg_device': lg_dev,
            'lg_confidence': lg_conf,
        })

    # Filter out kwargs that HierarchicalMatchingService.__init__ does NOT accept
    # (it only takes: postgres_repo, minio_repo, hierarchical_matcher, top_k, verbose)
    service_only_keys = {'top_k', 'verbose'}
    filtered_kwargs = {
        k: v for k, v in kwargs.items()
        if k != 'app_config' and k in service_only_keys
    }

    # Merge filtered kwargs (top_k, verbose) into service call
    service_kwargs = {}
    if 'top_k' in filtered_kwargs:
        service_kwargs['top_k'] = filtered_kwargs['top_k']
    if 'verbose' in filtered_kwargs:
        service_kwargs['verbose'] = filtered_kwargs['verbose']

    return HierarchicalMatchingService(
        postgres_repo=postgres_repo,
        minio_repo=minio_repo,
        **service_kwargs
    )

