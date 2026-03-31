"""
Image Compressor for Evidence Storage

Compresses raw image bytes for storage in the evidence-image bucket.
Converts to grayscale, resizes proportionally, and applies iterative JPEG
compression with SuperPoint keypoint validation to ensure the compressed
output retains sufficient detail for feature extraction.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_DIMENSION = 640
MIN_KEYPOINTS = 50
QUALITY_START = 85
QUALITY_STEP = 5
QUALITY_FLOOR = 40


def _count_superpoint_keypoints(
    image_gray: np.ndarray,
    sp_processor=None,
) -> int:
    """Run SuperPoint extraction and return the keypoint count."""
    if sp_processor is None:
        from src.processor.superPointProcessor import SuperPointProcessor, SuperPointConfig

        sp_processor = SuperPointProcessor(
            SuperPointConfig(max_keypoints=4096, max_dimension=MAX_DIMENSION, device="cpu")
        )

    result = sp_processor.extract(image_gray)
    return len(result.keypoints)


def compress_for_evidence(
    image_bytes: bytes,
    sp_processor=None,
) -> bytes:
    """Compress a raw image for evidence storage.

    Pipeline:
      1. Decode -> grayscale (colour is unnecessary for SuperPoint).
      2. Proportional resize so longest edge <= MAX_DIMENSION.
      3. Iterative JPEG quality search (85 -> 40) picking the lowest quality
         whose compressed output still yields >= MIN_KEYPOINTS SuperPoint
         keypoints.
      4. Falls back to QUALITY_START if no lower quality meets the threshold.

    Args:
        image_bytes: Raw image file bytes (JPEG / PNG / etc.).
        sp_processor: Optional pre-initialised SuperPointProcessor (avoids
            repeated model loading when compressing many images).

    Returns:
        JPEG-compressed bytes of the grayscale, resized image.

    Raises:
        ValueError: If the input bytes cannot be decoded as an image.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image bytes")

    original_size = len(image_bytes)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest > MAX_DIMENSION:
        scale = MAX_DIMENSION / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    logger.info(
        f"[imageCompressor] original={original_size} bytes, "
        f"grayscale resized to {gray.shape[1]}x{gray.shape[0]}"
    )

    # Lazy-init a CPU SuperPoint processor if none provided
    if sp_processor is None:
        from src.processor.superPointProcessor import SuperPointProcessor, SuperPointConfig

        sp_processor = SuperPointProcessor(
            SuperPointConfig(max_keypoints=4096, max_dimension=MAX_DIMENSION, device="cpu")
        )

    best_bytes: Optional[bytes] = None
    best_quality: Optional[int] = None

    quality = QUALITY_START
    while quality >= QUALITY_FLOOR:
        ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            quality -= QUALITY_STEP
            continue

        compressed = buf.tobytes()

        # Decode back for keypoint check
        check_img = cv2.imdecode(np.frombuffer(compressed, np.uint8), cv2.IMREAD_GRAYSCALE)
        kp_count = _count_superpoint_keypoints(check_img, sp_processor)

        logger.debug(
            f"[imageCompressor] quality={quality}, size={len(compressed)} bytes, "
            f"keypoints={kp_count}"
        )

        if kp_count >= MIN_KEYPOINTS:
            best_bytes = compressed
            best_quality = quality
        else:
            # Quality too low — stop searching
            break

        quality -= QUALITY_STEP

    # Fallback: if even QUALITY_START didn't meet threshold, encode at QUALITY_START
    if best_bytes is None:
        ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, QUALITY_START])
        if not ok:
            raise ValueError("JPEG encoding failed at fallback quality")
        best_bytes = buf.tobytes()
        best_quality = QUALITY_START
        logger.warning(
            f"[imageCompressor] No quality met {MIN_KEYPOINTS} keypoints threshold; "
            f"falling back to quality {QUALITY_START}"
        )

    reduction_pct = (1 - len(best_bytes) / original_size) * 100
    logger.info(
        f"[imageCompressor] accepted quality={best_quality}, "
        f"size={len(best_bytes)} bytes, reduction={reduction_pct:.1f}%"
    )

    return best_bytes
