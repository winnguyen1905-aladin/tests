"""
Validation utilities for hierarchical matching optimizations.

These functions verify that optimizations maintain accuracy and correctness.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


def validate_geometric_verification_accuracy(
    parallel_results: List[Any],
    sequential_results: List[Any],
    tolerance: float = 1e-3
) -> Tuple[bool, str]:
    """
    Validate that parallel geometric verification produces same results as sequential.
    
    Compares RANSAC results (inliers, reprojection error, transformation matrix)
    between parallel CUDA streams implementation and sequential baseline.
    
    Args:
        parallel_results: Results from parallel CUDA streams processing
        sequential_results: Results from sequential processing (baseline)
        tolerance: Numerical tolerance for floating point comparisons
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if len(parallel_results) != len(sequential_results):
        return False, f"Result count mismatch: {len(parallel_results)}vs {len(sequential_results)}"
    
    mismatches = []
    
    for i, (par, seq) in enumerate[tuple[Any, Any]](zip[tuple[Any, Any]](parallel_results, sequential_results)):
        # Compare candidate IDs
        if par.candidate_id != seq.candidate_id:
            mismatches.append(f"Result {i}: candidate_id mismatch ({par.candidate_id} vs {seq.candidate_id})")
            continue
        
        # Compare inlier counts (should be exact)
        if par.superpoint_inliers != seq.superpoint_inliers:
            mismatches.append(f"Result {i}: inlier count mismatch ({par.superpoint_inliers} vs {seq.superpoint_inliers})")
        
        # Compare reprojection errors (with tolerance)
        if not np.isclose(par.reprojection_error, seq.reprojection_error, rtol=tolerance, atol=tolerance):
            mismatches.append(f"Result {i}: reprojection error mismatch ({par.reprojection_error:.4f} vs {seq.reprojection_error:.4f})")
        
        # Compare final scores (with tolerance)
        if not np.isclose(par.final_score, seq.final_score, rtol=tolerance, atol=tolerance):
            mismatches.append(f"Result {i}: final score mismatch ({par.final_score:.4f} vs {seq.final_score:.4f})")
    
    if mismatches:
        return False, f"Found {len(mismatches)}mismatches:\n" + "\n".join(mismatches[:5])
    
    logger.info(f"✓ Geometric verification accuracy validated: {len(parallel_results)}results match")
    return True, "All results match within tolerance"


def validate_candidate_filtering_recall(
    filtered_results: List[Any],
    full_results: List[Any],
    min_recall: float = 0.95
) -> Tuple[bool, str]:
    """
    Validate that adaptive candidate filtering doesn't miss correct matches.
    
    Ensures that reducing candidate count (top_k) and early termination
    don't significantly reduce recall (ability to find correct matches).
    
    Args:
        filtered_results: Results from adaptive filtering (reduced candidates)
        full_results: Results from full candidate set (baseline)
        min_recall: Minimum acceptable recall ratio (default: 0.95 = 95%)
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not full_results:
        return True, "No baseline results to compare"
    
    if not filtered_results:
        return False, "Filtered results are empty"
    
    # Get best match from each
    filtered_best = filtered_results[0]
    full_best = full_results[0]
    
    # Check if same candidate was selected
    same_candidate = (filtered_best.candidate_id == full_best.candidate_id)
    
    # Check if score is within acceptable range
    score_ratio = filtered_best.final_score / full_best.final_score if full_best.final_score > 0 else 1.0
    
    if same_candidate and score_ratio >= min_recall:
        logger.info(f"✓ Candidate filtering recall validated: same best match, score ratio {score_ratio:.3f}")
        return True, f"Same best match selected, score ratio: {score_ratio:.3f}"
    elif same_candidate:
        return False, f"Same candidate but score degraded: {score_ratio:.3f} < {min_recall}"
    else:
        return False, f"Different best match: {filtered_best.candidate_id} vs {full_best.candidate_id}"


def validate_pipeline_consistency(
    results: List[Any],
    expected_properties: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """
    Validate pipeline consistency and correctness.
    
    Checks that results have expected properties:
    - All required fields present
    - Values within valid ranges
    - No NaN/Inf values
    - Scores are monotonically decreasing
    
    Args:
        results: List of MatchingResult objects
        expected_properties: Optional dict of expected properties to validate
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not results:
        return False, "No results to validate"
    
    issues = []
    
    for i, result in enumerate(results):
        # Check required fields
        required_fields = ['candidate_id', 'tree_id', 'final_score', 'superpoint_inliers']
        for field in required_fields:
            if not hasattr(result, field):
                issues.append(f"Result {i}: missing field '{field}'")
        
        # Check for NaN/Inf in scores
        if np.isnan(result.final_score) or (np.isinf(result.final_score) and result.final_score != float('inf')):
            issues.append(f"Result {i}: final_score is NaN or invalid Inf")
        
        # Check score range [0, 1]
        if result.final_score < 0 or result.final_score > 1:
            issues.append(f"Result {i}: final_score {result.final_score:.4f} out of range [0, 1]")
        
        # Check inliers are non-negative
        if result.superpoint_inliers < 0:
            issues.append(f"Result {i}: negative inliers {result.superpoint_inliers}")
    
    # Check monotonic decreasing scores
    for i in range(len(results) - 1):
        if results[i].final_score < results[i+1].final_score:
            issues.append(f"Results {i}-{i+1}: scores not monotonically decreasing ({results[i].final_score:.4f}< {results[i+1].final_score:.4f})")
    
    if issues:
        return False, f"Found {len(issues)} issues:\n" + "\n".join(issues[:5])
    
    logger.info(f"✓ Pipeline consistency validated: {len(results)} results are consistent")
    return True, "All results are consistent"

