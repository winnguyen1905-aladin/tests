#!/usr/bin/env python3
"""
LightGlue Processor - Fine-grained Feature Matching

Matches SuperPoint features between query and database images.
Returns inlier count for geometric verification.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..processor.superPointProcessor import SuperPointResult

logger = logging.getLogger(__name__)


@dataclass
class LightGlueConfig:
    """Configuration for LightGlue processor."""
    device: str = "cuda"
    match_threshold: float = 0.0  # Lower = more matches
    filter_threshold: float = 0.7  # Confidence filter
    verbose: bool = False


@dataclass
class LightGlueResult:
    """Result from LightGlue matching."""
    n_inliers: int  # Number of inlier matches after geometric verification
    matches: np.ndarray  # (N, 2) match indices
    confidence: float  # Match confidence score
    match_score: float  # Ratio of matches to keypoints


class LightGlueProcessor:
    """Processor for LightGlue feature matching."""

    def __init__(self, config: Optional[LightGlueConfig] = None) -> None:
        """Initialize LightGlue processor.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or LightGlueConfig()
        self.model = None
        self.using_fallback = False
        self._model_initialized = False

    def _init_model(self) -> None:
        """Initialize LightGlue model for SuperPoint features (256-dim)."""
        if self.config.verbose:
            print("Initializing LightGlue model for SuperPoint...")
            print(f"Device: {self.config.device}")

        # Try multiple import paths for LightGlue
        try:
            # First try direct import
            try:
                from lightglue import LightGlue
            except ImportError:
                # Try deep_image_matching thirdparty
                from lightglue import LightGlue

            # Use SuperPoint matcher for 256-dim descriptors (native SuperPoint)
            self.model = LightGlue(features='superpoint')
            self.using_fallback = False
            logger.info("LightGlue SuperPoint matcher loaded successfully!")
            if self.config.verbose:
                print(f"✓ LightGlue SuperPoint matcher loaded!")
                print(f"  Input dimension: {self.model.conf.input_dim}")
                print(f"  Expected: 256-dim descriptors")
        except ImportError as e:
            # Fallback to BFMatcher
            if self.config.verbose:
                print(f"⚠ LightGlue not available: {e}")
                print("Falling back to BFMatcher for feature matching...")
            logger.warning(f"LightGlue not available, using BFMatcher fallback: {e}")

            self.model = None  # Will use BFMatcher instead
            self.using_fallback = True

            if self.config.verbose:
                print("✓ BFMatcher loaded as fallback")
        except Exception as e:
            error_msg = f"❌ CRITICAL: Error loading LightGlue: {e}"
            logger.error(error_msg)
            raise RuntimeError(
                "Failed to initialize LightGlue matcher. "
                "Please check your installation and configuration."
            ) from e
    
    def match(
        self,
        query_features,
        candidate_features
    ) -> LightGlueResult:
        """Match features between query and candidate images.

        Args:
            query_features: Dict or SuperPointResult with 'keypoints', 'descriptors', 'scores'
            candidate_features: Dict or SuperPointResult with features

        Returns:
            LightGlueResult with match statistics
        """
        # Lazy load model on first use
        if not self._model_initialized:
            self._init_model()
            self._model_initialized = True

        # Log device info for debugging consistency issues
        if self.model is not None:
            model_device = next(self.model.parameters()).device
            logger.debug(f"LightGlue model device: {model_device}, config device: {self.config.device}")

        # Try LightGlue first, fall back to BFMatcher if it fails
        if self.model is not None:
            try:
                return self._match_with_lightglue(query_features, candidate_features)
            except Exception as e:
                logger.warning(f"LightGlue matching failed: {e}, falling back to BFMatcher")

        # Try BFMatcher fallback
        try:
            return self._match_with_bfmatcher(query_features, candidate_features)
        except Exception as e:
            logger.error(f"Both LightGlue and BFMatcher failed: {e}")
            # Return empty result instead of raising
            return LightGlueResult(
                n_inliers=0,
                matches=np.zeros((0, 2), dtype=np.int64),
                confidence=0.0,
                match_score=0.0
            )

    def _match_with_lightglue(self, query_features, candidate_features) -> LightGlueResult:
        if hasattr(query_features, 'keypoints'):
            # SuperPointResult object
            query_kp = torch.from_numpy(query_features.keypoints).float()
            query_desc = torch.from_numpy(query_features.descriptors).float()
            query_scores = torch.from_numpy(query_features.scores).float()
        else:
            # Dict input
            query_kp = torch.from_numpy(query_features['keypoints']).float()
            query_desc = torch.from_numpy(query_features['descriptors']).float()
            query_scores = torch.from_numpy(query_features['scores']).float()

        # Handle both dict and SuperPointResult input for candidate
        if hasattr(candidate_features, 'keypoints'):
            # SuperPointResult object
            cand_kp = torch.from_numpy(candidate_features.keypoints).float()
            cand_desc = torch.from_numpy(candidate_features.descriptors).float()
            cand_scores = torch.from_numpy(candidate_features.scores).float()
        else:
            # Dict input
            cand_kp = torch.from_numpy(candidate_features['keypoints']).float()
            cand_desc = torch.from_numpy(candidate_features['descriptors']).float()
            cand_scores = torch.from_numpy(candidate_features['scores']).float()

        # Create feature dictionaries for LightGlue (new API)
        # LightGlue expects descriptors in (1, N, dim) format
        feats0 = {
            'keypoints': query_kp.unsqueeze(0),  # Add batch dim: (1, N, 2)
            'descriptors': query_desc.unsqueeze(0),  # (1, N, 256) - NO transpose
            'keypoint_scores': query_scores.unsqueeze(0),  # (1, N)
        }

        feats1 = {
            'keypoints': cand_kp.unsqueeze(0),  # (1, N, 2)
            'descriptors': cand_desc.unsqueeze(0),  # (1, N, 256) - NO transpose
            'keypoint_scores': cand_scores.unsqueeze(0),  # (1, N)
        }

        # Move to device
        feats0 = {k: v.to(self.config.device) for k, v in feats0.items()}
        feats1 = {k: v.to(self.config.device) for k, v in feats1.items()}

        # Debug logging
        if self.config.verbose:
            print(f"feats0 descriptors shape: {feats0['descriptors'].shape}")
            print(f"feats1 descriptors shape: {feats1['descriptors'].shape}")
            if not self.using_fallback:
                print(f"LightGlue input_dim: {self.model.conf.input_dim}")

        # Check if dimensions match (should always be 256 for native SuperPoint)
        expected_dim = 256  # SuperPoint native dimension
        if self.model is not None:
            expected_dim = self.model.conf.input_dim

        if feats0['descriptors'].shape[2] != expected_dim:
            raise ValueError(
                f"Descriptor dimension mismatch! "
                f"Expected {expected_dim}-dim descriptors, "
                f"got {feats0['descriptors'].shape[2]}-dim. "
                f"Ensure you're using the correct feature type."
            )

        # LightGlue matching (required)
        with torch.no_grad():
            # Ensure features are on the same device as the model
            device = next(self.model.parameters()).device
            feats0_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in feats0.items()}
            feats1_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in feats1.items()}
            matches01 = self.model({'image0': feats0_device, 'image1': feats1_device})

        # Process matches - new API returns list of matches
        matches = matches01['matches']
        matching_scores = matches01['scores']  # This is a list of confidence scores

        if matches is None or len(matches) == 0:
            return LightGlueResult(
                n_inliers=0,
                matches=np.zeros((0, 2), dtype=np.int64),
                confidence=0.0,
                match_score=0.0
            )

        # Convert matches to numpy
        matches_array = matches[0].cpu().numpy() if isinstance(matches, list) else matches.cpu().numpy()

        # Explicitly clean up GPU tensors to prevent memory leaks
        del matches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Count valid matches (inliers) - those with high confidence
        # matching_scores could be a list or a tensor on CUDA
        try:
            if isinstance(matching_scores, list):
                scores_array = np.array(matching_scores)
            elif isinstance(matching_scores, torch.Tensor):
                # Move to CPU before converting to numpy
                if matching_scores.is_cuda:
                    scores_array = matching_scores.detach().cpu().numpy()
                else:
                    scores_array = matching_scores.detach().numpy()
            else:
                scores_array = np.array([matching_scores])
        except (TypeError, AttributeError) as e:
            # Fallback: if scores are on CUDA, use matches as score
            print(f"Warning: Could not convert matching_scores to numpy: {e}. Using matches count.")
            scores_array = np.array([1.0] * len(matches_array))
        if len(scores_array.shape) == 0:
            # Single score
            n_inliers = len(matches_array) if scores_array > self.config.filter_threshold else 0
        else:
            # Array of scores
            n_inliers = (scores_array > self.config.filter_threshold).sum()

        # Calculate match score
        match_score = len(matches_array) / len(feats0['keypoints'][0]) if len(feats0['keypoints'][0]) > 0 else 0

        # Final confidence
        confidence = n_inliers / len(matches_array) if len(matches_array) > 0 else 0.0

        # Calculate match score - handle both dict and SuperPointResult
        if isinstance(query_features, dict):
            n_query_kp = len(query_features['keypoints'])
        elif hasattr(query_features, 'keypoints'):
            n_query_kp = len(query_features.keypoints)
        else:
            n_query_kp = 1
        match_score = n_inliers / max(n_query_kp, 1)

        # Get max confidence
        # Initialize scores_array if not set (for FLANN fallback)
        if 'scores_array' not in locals():
            scores_array = np.array([match_score])
        max_confidence = float(scores_array.max()) if hasattr(scores_array, 'max') else float(scores_array)

        return LightGlueResult(
            n_inliers=n_inliers,
            matches=matches_array,
            confidence=max_confidence,
            match_score=match_score
        )

    def _match_with_bfmatcher(self, query_features, candidate_features) -> LightGlueResult:
        """Match features using OpenCV BFMatcher as fallback."""
        import cv2

        # Extract features
        if hasattr(query_features, 'keypoints'):
            query_kp = query_features.keypoints
            query_desc = query_features.descriptors
        else:
            query_kp = query_features['keypoints']
            query_desc = query_features['descriptors']

        if hasattr(candidate_features, 'keypoints'):
            cand_kp = candidate_features.keypoints
            cand_desc = candidate_features.descriptors
        else:
            cand_kp = candidate_features['keypoints']
            cand_desc = candidate_features['descriptors']

        # Ensure descriptors are float32
        query_desc = query_desc.astype(np.float32)
        cand_desc = cand_desc.astype(np.float32)

        if len(query_desc) == 0 or len(cand_desc) == 0:
            logger.warning("No descriptors to match")
            return LightGlueResult(
                n_inliers=0,
                matches=np.array([], dtype=np.int32).reshape(0, 2),
                confidence=0.0,
                match_score=0.0
            )

        # Use BFMatcher with L2 distance
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(query_desc, cand_desc, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1:
                good_matches.append(match_pair[0])

        if len(good_matches) == 0:
            logger.warning("No good matches found")
            return LightGlueResult(
                n_inliers=0,
                matches=np.array([], dtype=np.int32).reshape(0, 2),
                confidence=0.0,
                match_score=0.0
            )

        # Extract match indices and distances
        match_indices = np.array([[m.queryIdx, m.trainIdx] for m in good_matches], dtype=np.int32)
        match_distances = np.array([m.distance for m in good_matches], dtype=np.float32)

        # Convert distances to confidence scores (inverse relationship)
        # Normalize distances to [0, 1] range
        max_dist = match_distances.max() if len(match_distances) > 0 else 1.0
        confidence_scores = 1.0 - (match_distances / (max_dist + 1e-6))
        confidence_scores = np.clip(confidence_scores, 0.0, 1.0).astype(np.float32)
        avg_confidence = float(confidence_scores.mean())

        # Try RANSAC for geometric verification
        num_inliers = 0
        if len(match_indices) >= 4:
            try:
                query_pts = query_kp[match_indices[:, 0]].astype(np.float32)
                cand_pts = cand_kp[match_indices[:, 1]].astype(np.float32)

                # Compute fundamental matrix with RANSAC
                F, mask = cv2.findFundamentalMat(query_pts, cand_pts, cv2.FM_RANSAC, 1.0, 0.99)
                if F is not None:
                    num_inliers = int(np.sum(mask))
                    logger.info(f"RANSAC found {num_inliers} inliers out of {len(match_indices)} matches")
            except Exception as e:
                logger.warning(f"RANSAC failed: {e}")
                num_inliers = len(match_indices)
        else:
            num_inliers = len(match_indices)

        match_score = len(good_matches) / max(len(query_desc), 1)

        logger.info(f"BFMatcher: {len(good_matches)}matches, score={match_score:.3f}, confidence={avg_confidence:.3f}, inliers={num_inliers}")

        return LightGlueResult(
            n_inliers=num_inliers,
            matches=match_indices,
            confidence=avg_confidence,
            match_score=match_score
        )

    def match_from_results(
        self,
        query_result: 'SuperPointResult',
        candidate_result: 'SuperPointResult'
    ) -> LightGlueResult:
        """Match using SuperPointResult objects directly.
        
        Args:
            query_result: SuperPointResult from query image
            candidate_result: SuperPointResult from candidate image
        
        Returns:
            LightGlueResult with match statistics
        """
        query_features = {
            'keypoints': query_result.keypoints,
            'descriptors': query_result.descriptors,
            'scores': query_result.scores,
        }
        candidate_features = {
            'keypoints': candidate_result.keypoints,
            'descriptors': candidate_result.descriptors,
            'scores': candidate_result.scores,
        }
        return self.match(query_features, candidate_features)
    
    def match_batch(
        self,
        query_features: Dict[str, np.ndarray],
        candidates: list,
        stream_count: int = 8
    ) -> list:
        """Match query against multiple candidates using CUDA streams for parallelism.

        Args:
            query_features: Query image features
            candidates: List of candidate feature dicts
            stream_count: Number of CUDA streams for parallel GPU dispatch

        Returns:
            List of LightGlueResult for each candidate
        """
        if not candidates:
            return []

        use_streams = (
            self.config.device == "cuda"
            and torch.cuda.is_available()
            and self.model is not None
            and not self.using_fallback
        )

        if not use_streams:
            return [self.match(query_features, cand) for cand in candidates]

        # Pre-convert query tensors once and pin memory for fast transfers
        def to_feats(f):
            if hasattr(f, 'keypoints'):
                kp = torch.from_numpy(f.keypoints).float()
                desc = torch.from_numpy(f.descriptors).float()
                sc = torch.from_numpy(f.scores).float()
            else:
                kp = torch.from_numpy(f['keypoints']).float()
                desc = torch.from_numpy(f['descriptors']).float()
                sc = torch.from_numpy(f['scores']).float()
            return {
                'keypoints': kp.unsqueeze(0),
                'descriptors': desc.unsqueeze(0),
                'keypoint_scores': sc.unsqueeze(0),
            }

        q_feats_cpu = to_feats(query_features)

        # Create CUDA streams
        streams = [torch.cuda.Stream() for _ in range(min(stream_count, len(candidates)))]
        results = [None] * len(candidates)

        # Phase 1: async GPU transfers + inference across streams
        pending = []
        for i, cand in enumerate(candidates):
            stream = streams[i % len(streams)]
            c_feats_cpu = to_feats(cand)

            with torch.cuda.stream(stream):
                q_gpu = {k: v.to(self.config.device, non_blocking=True) for k, v in q_feats_cpu.items()}
                c_gpu = {k: v.to(self.config.device, non_blocking=True) for k, v in c_feats_cpu.items()}

                with torch.no_grad():
                    out = self.model({'image0': q_gpu, 'image1': c_gpu})

            pending.append((i, stream, out, q_feats_cpu))

        # Phase 2: sync each stream and collect results
        for i, stream, out, q_f in pending:
            stream.synchronize()

            matches = out['matches']
            scores_raw = out['scores']

            if matches is None or len(matches) == 0:
                results[i] = LightGlueResult(n_inliers=0, matches=np.zeros((0, 2), dtype=np.int64), confidence=0.0, match_score=0.0)
                continue

            matches_np = matches[0].cpu().numpy() if isinstance(matches, list) else matches.cpu().numpy()

            # Clean up GPU tensors to prevent memory leaks
            del matches, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if isinstance(scores_raw, torch.Tensor):
                scores_np = scores_raw.detach().cpu().numpy()
            elif isinstance(scores_raw, list):
                scores_np = np.array(scores_raw)
            else:
                scores_np = np.array([1.0] * len(matches_np))

            n_inliers = int((scores_np > self.config.filter_threshold).sum()) if scores_np.ndim > 0 else 0
            n_q_kp = q_f['keypoints'].shape[1]
            match_score = n_inliers / max(n_q_kp, 1)
            confidence = float(scores_np.max()) if scores_np.size > 0 else 0.0

            results[i] = LightGlueResult(
                n_inliers=n_inliers,
                matches=matches_np,
                confidence=confidence,
                match_score=match_score
            )

        return results
    
    def close(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'extractor') and self.extractor is not None:
            del self.extractor
            self.extractor = None


def create_lightglue_processor(
    device: str = "cuda",
    filter_threshold: float = 0.7,
    verbose: bool = False
) -> LightGlueProcessor:
    """Factory function to create LightGlue processor.
    
    Args:
        device: Device to run on ('cuda' or 'cpu')
        filter_threshold: Confidence threshold for matches
        verbose: Enable verbose output
        
    Returns:
        Configured LightGlueProcessor instance
    """
    config = LightGlueConfig(
        device=device,
        filter_threshold=filter_threshold,
        verbose=verbose
    )
    return LightGlueProcessor(config)


if __name__ == "__main__":
    # Example usage
    import argparse
    import sys
    from src.config.appConfig import get_config

    parser = argparse.ArgumentParser(description="LightGlue Feature Matching")
    parser.add_argument("--image1", required=True, help="Path to first image")
    parser.add_argument("--image2", required=True, help="Path to second image")
    parser.add_argument("--features1", help="Path to precomputed features (.npz)")
    parser.add_argument("--features2", help="Path to precomputed features (.npz)")
    parser.add_argument("--max-kp", type=int, help="Max keypoints (required)")
    parser.add_argument("--max-dim", type=int, help="Max dimension (required)")
    args = parser.parse_args()

    # Get config
    app_config = get_config()
    max_kp = args.max_kp if args.max_kp else app_config.sp_max_keypoints
    max_dim = args.max_dim if args.max_dim else app_config.sp_max_dimension

    if max_kp is None or max_dim is None:
        parser.error("--max-kp and --max-dim are required (or configure appConfig)")

    # Import processors
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from superPointProcessor import create_superpoint_processor

    # Load images
    img1 = cv2.imread(args.image1)
    img2 = cv2.imread(args.image2)

    if img1 is None or img2 is None:
        raise ValueError("Failed to load images")

    # Extract features with REQUIRED parameters
    sp = create_superpoint_processor(
        max_keypoints=max_kp,
        max_dimension=max_dim,
        verbose=True
    )
    
    if args.features1 and args.features2:
        # Load precomputed features
        feat1 = np.load(args.features1)
        feat2 = np.load(args.features2)
        query_features = {
            'keypoints': feat1['keypoints'],
            'descriptors': feat1['descriptors'],
            'scores': feat1['scores'],
        }
        cand_features = {
            'keypoints': feat2['keypoints'],
            'descriptors': feat2['descriptors'],
            'scores': feat2['scores'],
        }
    else:
        # Extract from images
        result1 = sp.extract(img1)
        result2 = sp.extract(img2)
        query_features = {
            'keypoints': result1.keypoints,
            'descriptors': result1.descriptors,
            'scores': result1.scores,
        }
        cand_features = {
            'keypoints': result2.keypoints,
            'descriptors': result2.descriptors,
            'scores': result2.scores,
        }
    
    # Match
    lg = create_lightglue_processor(verbose=True)
    result = lg.match(query_features, cand_features)
    
    print(f"\n=== Matching Result ===")
    print(f"Inliers: {result.n_inliers}")
    print(f"Match confidence: {result.confidence:.4f}")
    print(f"Match score: {result.match_score:.4f}")

