"""
Similarity Utilities

Centralized cosine similarity computations for use across the codebase.

Three levels of abstraction:
- ``cosine_similarity``        — single pair of vectors (scalar result)
- ``cosine_similarity_batch``  — one query vs. many candidates, CPU
- ``cosine_similarity_batch_auto`` — same as above but prefers GPU when available
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector, any shape that ``np.dot`` accepts.
        vec2: Second vector, same shape as *vec1*.

    Returns:
        Cosine similarity in [-1, 1].  Returns 0.0 when either vector is zero.
    """
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def cosine_similarity_batch(
    query: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and a batch of candidates
    using CPU arithmetic.

    Args:
        query:      Query vector, shape ``(D,)``.
        candidates: Candidate matrix, shape ``(N, D)``.

    Returns:
        Similarity scores, shape ``(N,)``.
    """
    query_norm = query / (np.linalg.norm(query) + 1e-7)
    candidates_norm = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-7)
    return np.dot(candidates_norm, query_norm)


def cosine_similarity_batch_auto(
    query: np.ndarray,
    candidates: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and a batch of candidates.

    Uses GPU-accelerated computation when *device* is ``"cuda"`` and a CUDA
    device is available, otherwise falls back to CPU.

    Args:
        query:      Query vector, shape ``(D,)``.
        candidates: Candidate matrix, shape ``(N, D)``.
        device:     Target device string — ``"cuda"`` or ``"cpu"``.

    Returns:
        Similarity scores, shape ``(N,)``.
    """
    import torch

    if device == "cuda" and torch.cuda.is_available():
        try:
            query_t = torch.from_numpy(query.reshape(1, -1)).float().to(device)
            cands_t = torch.from_numpy(candidates).float().to(device)
            query_norm = torch.nn.functional.normalize(query_t, p=2, dim=1)
            cands_norm = torch.nn.functional.normalize(cands_t, p=2, dim=1)
            similarities = torch.mm(query_norm, cands_norm.T).squeeze(0)
            return similarities.cpu().numpy()
        except Exception as exc:
            logger.warning(
                "[cosine_similarity_batch_auto] GPU computation failed, falling back to CPU: %s", exc
            )

    return cosine_similarity_batch(query, candidates)
