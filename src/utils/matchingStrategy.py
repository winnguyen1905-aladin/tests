#!/usr/bin/env python3
"""
Matching Strategy - Decision Logic for Tree Identification

Implements the decision rules for determining if a query image matches
a known tree based on inlier counts from LightGlue matching.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from enum import Enum


class Decision(Enum):
    """Decision outcome for tree identification."""
    MATCH = "MATCH"
    UNKNOWN = "UNKNOWN"
    LOW_SIMILARITY = "LOW_SIMILARITY"
    NO_CANDIDATES = "NO_CANDIDATES"


# TypedDicts for structured data passed into the strategy
class LightGlueResult(TypedDict):
    """Result from LightGlue fine-grained matching."""
    n_inliers: int
    confidence: float


class CoarseResult(TypedDict):
    """Result from coarse retrieval (Milvus)."""
    image_id: str
    tree_id: str
    similarity: float


class MatchResultDict(TypedDict):
    """Dict format for match results used by decide()."""
    image_id: str
    tree_id: str
    coarse_score: float
    coarse_similarity: float
    n_inliers: int
    match_score: float
    confidence: float


class DecisionResult(TypedDict):
    """Return type for decide() method."""
    decision: str
    reason: str
    tree_id: Optional[str]
    best_match: Optional[MatchResultDict]
    all_matches: List[MatchResultDict]


@dataclass
class MatchCandidate:
    """A candidate match from coarse retrieval."""
    image_id: str
    tree_id: str
    coarse_similarity: float
    n_inliers: int = 0
    match_confidence: float = 0.0


@dataclass
class MatchingStrategyResult:
    """Result from matching strategy."""
    decision: Decision
    matched_tree_id: Optional[str]
    confidence: float
    reason: str
    best_candidate: Optional[MatchCandidate]
    all_candidates: List[MatchCandidate] = field(default_factory=list)


@dataclass
class MatchingStrategyConfig:
    """Configuration for matching strategy."""
    inlier_threshold: int = 15  # Very low threshold to test performance
    coarse_similarity_threshold: float = 0.6  # Minimum coarse similarity
    top_k_coarse: int = 10  # Number of coarse candidates to evaluate
    min_inlier_ratio: float = 0.01  # Minimum inlier ratio (inliers / query_kp)


class MatchingStrategy:
    """Strategy for making match/unmatch decisions based on inlier counts."""

    def __init__(self, config: Optional[MatchingStrategyConfig] = None) -> None:
        """Initialize matching strategy.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config: MatchingStrategyConfig = config or MatchingStrategyConfig()

    def evaluate(
        self,
        query_features_count: int,
        coarse_results: List[Tuple[Tuple[str, str, Any], float]],
        fine_grained_results: List[Tuple[LightGlueResult, Any]],
    ) -> MatchingStrategyResult:
        """Evaluate all candidates and make a decision.

        Args:
            query_features_count: Number of query keypoints.
            coarse_results: List of ((image_id, tree_id, metadata), similarity)
                from Milvus coarse retrieval.
            fine_grained_results: List of (LightGlueResult, extra_data) for each
                candidate returned by LightGlue fine-grained matching.

        Returns:
            MatchingStrategyResult with decision and details.
        """
        if not coarse_results:
            return MatchingStrategyResult(
                decision=Decision.NO_CANDIDATES,
                matched_tree_id=None,
                confidence=0.0,
                reason="No candidates found in database",
                best_candidate=None,
                all_candidates=[],
            )

        # Check if top candidate passes coarse threshold
        top_similarity: float = coarse_results[0][1] if coarse_results else 0.0
        if top_similarity < self.config.coarse_similarity_threshold:
            return MatchingStrategyResult(
                decision=Decision.LOW_SIMILARITY,
                matched_tree_id=None,
                confidence=top_similarity,
                reason=f"Top coarse similarity ({top_similarity:.3f}) below threshold ({self.config.coarse_similarity_threshold})",
                best_candidate=None,
                all_candidates=[],
            )

        # Build candidate list with fine-grained results
        candidates: List[MatchCandidate] = []
        for i, ((image_id, tree_id, _), (lg_result, _)) in enumerate[tuple[Tuple[Tuple[str, str, Any], float], Tuple[LightGlueResult, Any]]](zip(coarse_results, fine_grained_results)):
            candidate: MatchCandidate = MatchCandidate(
                image_id=image_id,
                tree_id=tree_id,
                coarse_similarity=coarse_results[i][1] if i < len(coarse_results) else 0.0,
                n_inliers=lg_result["n_inliers"],
                match_confidence=lg_result["confidence"],
            )
            candidates.append(candidate)

        # Sort by inlier count (descending)
        candidates.sort(key=lambda x: x.n_inliers, reverse=True)

        # Get best candidate
        best: Optional[MatchCandidate] = candidates[0] if candidates else None

        if best is None:
            return MatchingStrategyResult(
                decision=Decision.UNKNOWN,
                matched_tree_id=None,
                confidence=0.0,
                reason="No valid candidates after fine-grained matching",
                best_candidate=None,
                all_candidates=[],
            )

        # Apply inlier threshold
        min_inliers: int = self.config.inlier_threshold
        inlier_ratio: float = best.n_inliers / max(query_features_count, 1)

        if best.n_inliers >= min_inliers:
            return MatchingStrategyResult(
                decision=Decision.MATCH,
                matched_tree_id=best.tree_id,
                confidence=self._calculate_confidence(best, query_features_count),
                reason=f"MATCH: {best.n_inliers} inliers (threshold: {min_inliers})",
                best_candidate=best,
                all_candidates=candidates,
            )
        else:
            return MatchingStrategyResult(
                decision=Decision.UNKNOWN,
                matched_tree_id=None,
                confidence=inlier_ratio,
                reason=f"UNKNOWN: Best candidate has {best.n_inliers} inliers (threshold: {min_inliers})",
                best_candidate=best,
                all_candidates=candidates,
            )

    def evaluate_simple(
        self,
        n_inliers: int,
        coarse_similarity: float,
        query_keypoints: int = 0,
    ) -> MatchingStrategyResult:
        """Simple evaluation for a single candidate.

        Args:
            n_inliers: Number of inliers from LightGlue.
            coarse_similarity: Coarse similarity from Milvus.
            query_keypoints: Number of query keypoints (optional).

        Returns:
            MatchingStrategyResult with decision.
        """
        if coarse_similarity < self.config.coarse_similarity_threshold:
            return MatchingStrategyResult(
                decision=Decision.LOW_SIMILARITY,
                matched_tree_id=None,
                confidence=coarse_similarity,
                reason=f"Low coarse similarity: {coarse_similarity:.3f}",
                best_candidate=None,
                all_candidates=[],
            )

        if n_inliers >= self.config.inlier_threshold:
            confidence: float
            if query_keypoints > 0:
                confidence = n_inliers / max(query_keypoints, 1)
            else:
                confidence = 0.5
            return MatchingStrategyResult(
                decision=Decision.MATCH,
                matched_tree_id="unknown",  # Caller should fill this
                confidence=confidence,
                reason=f"MATCH: {n_inliers} inliers >= threshold {self.config.inlier_threshold}",
                best_candidate=None,
                all_candidates=[],
            )

        return MatchingStrategyResult(
            decision=Decision.UNKNOWN,
            matched_tree_id=None,
            confidence=0.0,
            reason=f"UNKNOWN: {n_inliers} inliers < threshold {self.config.inlier_threshold}",
            best_candidate=None,
            all_candidates=[],
        )

    def _calculate_confidence(self, candidate: MatchCandidate, query_kp_count: int) -> float:
        """Calculate confidence score for a match.

        Args:
            candidate: Match candidate.
            query_kp_count: Number of query keypoints.

        Returns:
            Confidence score between 0 and 1.
        """
        # Combine coarse and fine-grained scores
        coarse_weight: float = 0.3
        fine_weight: float = 0.5
        inlier_weight: float = 0.2

        inlier_ratio: float
        if query_kp_count > 0:
            inlier_ratio = candidate.n_inliers / max(query_kp_count, 1)
        else:
            inlier_ratio = 0.5

        confidence: float = (
            coarse_weight * candidate.coarse_similarity
            + fine_weight * candidate.match_confidence
            + inlier_weight * min(inlier_ratio, 1.0)
        )

        return min(confidence, 1.0)

    def decide(
        self,
        match_results: List[MatchResultDict],
        inlier_threshold: Optional[int] = None,
    ) -> DecisionResult:
        """Make decision from match results.

        Args:
            match_results: List of match result dicts with keys:
                - image_id: str
                - tree_id: str
                - coarse_score: float
                - coarse_similarity: float
                - n_inliers: int
                - match_score: float
                - confidence: float
            inlier_threshold: Override inlier threshold.

        Returns:
            DecisionResult dict with decision details.
        """
        threshold: int = inlier_threshold if inlier_threshold is not None else self.config.inlier_threshold

        if not match_results:
            return DecisionResult(
                decision="UNKNOWN",
                reason="No match results",
                tree_id=None,
                best_match=None,
                all_matches=[],
            )

        # Find best match
        best: MatchResultDict = max(match_results, key=lambda x: x.get("n_inliers", 0))  # type: ignore[arg-type]

        # Check if best candidate passes coarse similarity threshold
        best_coarse: float = best.get("coarse_similarity", 0.0)
        if best_coarse < self.config.coarse_similarity_threshold:
            return DecisionResult(
                decision="UNKNOWN",
                tree_id=None,
                reason=f"UNKNOWN: Low coarse similarity ({best_coarse:.3f}) < threshold ({self.config.coarse_similarity_threshold})",
                best_match=best,
                all_matches=match_results,
            )

        if best.get("n_inliers", 0) >= self.config.inlier_threshold:
            return DecisionResult(
                decision="MATCH",
                tree_id=best.get("tree_id"),
                reason=f"MATCH: {best.get('n_inliers')} inliers (threshold: {threshold})",
                best_match=best,
                all_matches=match_results,
            )
        else:
            return DecisionResult(
                decision="UNKNOWN",
                tree_id=None,
                reason=f"UNKNOWN: Best has {best.get('n_inliers')} inliers (threshold: {threshold})",
                best_match=best,
                all_matches=match_results,
            )

    def set_thresholds(
        self,
        inlier_threshold: Optional[int] = None,
        coarse_threshold: Optional[float] = None,
    ) -> None:
        """Update thresholds dynamically.

        Args:
            inlier_threshold: New inlier threshold.
            coarse_threshold: New coarse similarity threshold.
        """
        if inlier_threshold is not None:
            self.config.inlier_threshold = inlier_threshold
        if coarse_threshold is not None:
            self.config.coarse_similarity_threshold = coarse_threshold


def create_matching_strategy(
    inlier_threshold: int = 25,
    coarse_threshold: float = 0.6,
    min_inlier_ratio: float = 0.01,
) -> MatchingStrategy:
    """Factory function to create matching strategy.

    Args:
        inlier_threshold: Minimum inliers for match decision.
        coarse_threshold: Minimum coarse similarity threshold.
        min_inlier_ratio: Minimum inlier ratio threshold.

    Returns:
        Configured MatchingStrategy instance.
    """
    config: MatchingStrategyConfig = MatchingStrategyConfig(
        inlier_threshold=inlier_threshold,
        coarse_similarity_threshold=coarse_threshold,
        min_inlier_ratio=min_inlier_ratio,
    )
    return MatchingStrategy(config)


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create strategy
    strategy: MatchingStrategy = create_matching_strategy(inlier_threshold=50, coarse_threshold=0.6)

    # Test cases
    test_cases: List[Tuple[str, int, float]] = [
        ("Good match", 75, 0.85),
        ("Good match (low coarse)", 80, 0.55),
        ("Low inliers", 15, 0.90),
        ("Low both", 10, 0.45),
        ("Borderline", 50, 0.70),
    ]

    print("=== Matching Strategy Tests ===\n")

    for name, n_inliers, coarse_sim in test_cases:
        result: MatchingStrategyResult = strategy.evaluate_simple(
            n_inliers=n_inliers,
            coarse_similarity=coarse_sim,
            query_keypoints=1000,
        )

        print(f"Test: {name}")
        print(f"  Inliers: {n_inliers}, Coarse: {coarse_sim:.2f}")
        print(f"  Decision: {result.decision.value}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Reason: {result.reason}")
        print()
