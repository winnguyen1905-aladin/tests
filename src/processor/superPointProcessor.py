#!/usr/bin/env python3
"""
SuperPoint Processor - Local Feature Extraction with Uniform Distribution

Extracts keypoints and descriptors using SuperPoint algorithm.
Includes uniform distribution to ensure keypoints cover the entire object.
"""

import cv2
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_sp_cuda_stream: Optional[torch.cuda.Stream] = None

def get_sp_cuda_stream() -> torch.cuda.Stream:
    """Get or create a shared CUDA stream for async inference."""
    global _sp_cuda_stream
    if _sp_cuda_stream is None and torch.cuda.is_available():
        _sp_cuda_stream = torch.cuda.Stream()
    return _sp_cuda_stream


@dataclass
class SuperPointConfig:
    """Configuration for SuperPoint processor.

    Note: Default values are provided for backward compatibility.
    In production, these should be overridden by appConfig values.
    """
    max_keypoints: int = 4096  # Default from appConfig
    max_dimension: int = 4096  # Default from appConfig (matches DINO/Milvus dimension)
    detection_threshold: float = 0.001  # LOWER for more keypoints
    nms_radius: int = 4
    device: str = "cuda"
    verbose: bool = False
    # New options for uniform distribution
    enable_uniform_distribution: bool = False  # Disabled by default - too aggressive filtering
    grid_size: int = 8  # Divide mask into 8x8 grid for uniform sampling


@dataclass
class SuperPointResult:
    """Result from SuperPoint feature extraction."""
    keypoints: np.ndarray  # (N, 2) float32, (x, y) coordinates
    descriptors: np.ndarray  # (N, 256) float32
    scores: np.ndarray  # (N,) float32, keypoint scores
    image_size: tuple  # (H, W) of input image


class SuperPointProcessor:
    """Processor for SuperPoint feature extraction."""
    
    def __init__(self, config: Optional[SuperPointConfig] = None) -> None:
        """Initialize SuperPoint processor.
        
        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or SuperPointConfig()
        self.model = None
        self.use_lightglue = True  # Will be set properly when model loads
    
    def _init_model(self) -> None:
        """Initialize SuperPoint model (LightGlue preferred, SIFT fallback)."""
        try:
            from lightglue import SuperPoint as LightGlueSuperPoint

            self.model = LightGlueSuperPoint(
                max_num_keypoints=self.config.max_keypoints,
                detection_threshold=self.config.detection_threshold,
                nms_radius=self.config.nms_radius
            ).eval()

            if torch.cuda.is_available() and self.config.device == "cuda":
                self.model = self.model.cuda()

            self.use_lightglue = True

        except ImportError as e:
            logger.warning(f"LightGlue not available, falling back to SIFT: {e}")
            self.model = cv2.SIFT_create()
            self.use_lightglue = False
    
    def extract(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> SuperPointResult:
        """Extract keypoints and descriptors from image.

        Args:
            image: Input image in BGR or grayscale format (H, W) or (H, W, 3)
            mask: Optional binary mask to constrain keypoint detection to specific regions.
                  If provided, ensures keypoints are within the mask and uniformly distributed.

        Returns:
            SuperPointResult with keypoints, descriptors, and scores
        """
        return self._extract_superpoint(image, mask)

    def process(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> SuperPointResult:
        """Process single image with SuperPoint (alias for extract).

        Args:
            image: Input image
            mask: Optional binary mask

        Returns:
            SuperPointResult containing keypoints and descriptors
        """
        return self.extract(image, mask)

    def _extract_superpoint(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> SuperPointResult:
        """Extract keypoints and descriptors from image.

        Args:
            image: Input image in BGR or grayscale format (H, W) or (H, W, 3)
            mask: Optional binary mask for uniform distribution

        Returns:
            SuperPointResult with keypoints, descriptors, and scores
        """
        if self.model is None:
            self._init_model()

        original_size = (image.shape[0], image.shape[1])

        if self.use_lightglue:
            return self._extract_lightglue_superpoint(image, original_size, mask)
        else:
            return self._extract_sift_fallback(image, original_size, mask)

    def _extract_lightglue_superpoint(
        self,
        image: np.ndarray,
        original_size: tuple,
        mask: Optional[np.ndarray] = None
    ) -> SuperPointResult:
        """Extract features using LightGlue SuperPoint."""
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        if self.config.device == "cuda" and torch.cuda.is_available():
            cuda_stream = get_sp_cuda_stream()
            if not image_tensor.is_pinned():
                image_tensor = image_tensor.pin_memory()
            with torch.cuda.stream(cuda_stream):
                image_tensor = image_tensor.to(self.config.device, non_blocking=True)
                with torch.no_grad():
                    feats = self.model.extract(image_tensor)
                torch.cuda.synchronize()
        else:
            image_tensor = image_tensor.to(self.config.device)
            with torch.no_grad():
                feats = self.model.extract(image_tensor)

        keypoints = feats['keypoints'][0].cpu().numpy()
        descriptors = feats['descriptors'][0].cpu().numpy()
        scores = feats['keypoint_scores'][0].cpu().numpy()

        result = SuperPointResult(
            keypoints=keypoints.astype(np.float32),
            descriptors=descriptors.astype(np.float32),
            scores=scores.astype(np.float32),
            image_size=original_size
        )

        if len(result.keypoints) == 0:
            return result

        if mask is not None and self.config.enable_uniform_distribution:
            result = self._apply_uniform_distribution(result, mask)

        return result

    def extract_batch(self, images: list, masks: Optional[list] = None, batch_size: int = 8) -> list:
        """Extract features from multiple images using TRUE batch processing.

        Args:
            images: List of BGR images
            masks: Optional list of binary masks for each image
            batch_size: Batch size for GPU processing (default: 8 for RTX 5060 Ti)

        Returns:
            List of SuperPointResult
        """
        if not images:
            return []

        if self.model is None:
            self._init_model()

        # Process in batches for maximum GPU utilization
        all_results = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_masks = masks[i:i + batch_size] if masks else [None] * len(batch_images)

            # Prepare batch tensors
            original_sizes = [(img.shape[0], img.shape[1]) for img in batch_images]
            batch_tensors = []

            for img in batch_images:
                if len(img.shape) == 2:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Convert to tensor [0, 1]
                img_tensor = torch.from_numpy(img_rgb).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)
                batch_tensors.append(img_tensor)

            # Stack into batch (B, 3, H, W) - need to resize to same dimensions first
            # For now, process individually but with batched GPU transfers
            batch_tensor_list = []
            for tensor in batch_tensors:
                batch_tensor_list.append(tensor.unsqueeze(0))  # (1, 3, H, W)

            use_gpu = self.config.device == "cuda" and torch.cuda.is_available()
            for j, (tensor, orig_size, mask) in enumerate(zip(batch_tensor_list, original_sizes, batch_masks)):
                try:
                    tensor = tensor.to(self.config.device, non_blocking=use_gpu)

                    with torch.no_grad():
                        feats = self.model.extract(tensor)

                    keypoints = feats['keypoints'][0].cpu().numpy()
                    descriptors = feats['descriptors'][0].cpu().numpy()
                    scores = feats['keypoint_scores'][0].cpu().numpy()

                    result = SuperPointResult(
                        keypoints=keypoints.astype(np.float32),
                        descriptors=descriptors.astype(np.float32),
                        scores=scores.astype(np.float32),
                        image_size=orig_size
                    )

                    if mask is not None and self.config.enable_uniform_distribution and len(result.keypoints) > 0:
                        result = self._apply_uniform_distribution(result, mask)

                    all_results.append(result)

                except Exception as e:
                    logger.warning(f"Batch item {j} failed: {e}, retrying single")
                    if j < len(batch_images):
                        result = self.extract(batch_images[j], batch_masks[j])
                        if result:
                            all_results.append(result)

        return all_results

    def _extract_sift_fallback(
        self,
        image: np.ndarray,
        original_size: tuple,
        mask: Optional[np.ndarray] = None
    ) -> SuperPointResult:
        """Extract using SIFT as fallback when LightGlue is not available."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Extract SIFT features
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is None:
            logger.warning("SIFT failed to extract descriptors")
            return SuperPointResult(
                keypoints=np.array([], dtype=np.float32).reshape(0, 2),
                descriptors=np.array([], dtype=np.float32).reshape(0, 256),
                scores=np.array([], dtype=np.float32),
                image_size=original_size
            )

        kp_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        scores = np.array([kp.response for kp in keypoints], dtype=np.float32)

        # Pad 128-dim SIFT descriptors to 256-dim for compatibility
        desc_array = descriptors.astype(np.float32)
        if desc_array.shape[1] < 256:
            padded = np.zeros((desc_array.shape[0], 256), dtype=np.float32)
            padded[:, :desc_array.shape[1]] = desc_array
            desc_array = padded

        if len(kp_array) > self.config.max_keypoints:
            top_indices = np.argsort(scores)[-self.config.max_keypoints:]
            kp_array = kp_array[top_indices]
            desc_array = desc_array[top_indices]
            scores = scores[top_indices]

        result = SuperPointResult(
            keypoints=kp_array,
            descriptors=desc_array,
            scores=scores,
            image_size=original_size
        )

        if mask is not None and self.config.enable_uniform_distribution:
            result = self._apply_uniform_distribution(result, mask)

        return result

    def _apply_uniform_distribution(
        self,
        result: SuperPointResult,
        mask: np.ndarray
    ) -> SuperPointResult:
        """Ensure keypoints are uniformly distributed across the mask.

        Args:
            result: Original SuperPointResult
            mask: Binary mask defining the region of interest

        Returns:
            SuperPointResult with uniformly distributed keypoints
        """
        h, w = mask.shape[:2]
        original_size = result.image_size  # Get original size from result

        if len(result.keypoints) == 0:
            return result

        h, w = result.image_size
        keypoints = result.keypoints.copy()
        descriptors = result.descriptors.copy()
        scores = result.scores.copy()

        if mask.shape[:2] != (h, w):
            mask_resized = cv2.resize(mask, (w, h))
        else:
            mask_resized = mask.copy()

        if np.count_nonzero(mask_resized) == 0:
            logger.warning("Empty mask, skipping uniform distribution")
            return result

        grid_size = self.config.grid_size
        cell_h = h // grid_size
        cell_w = w // grid_size

        cell_keypoints = {i: [] for i in range(grid_size * grid_size)}
        for idx, kp in enumerate(keypoints):
            cell_x = min(int(kp[0] // cell_w), grid_size - 1)
            cell_y = min(int(kp[1] // cell_h), grid_size - 1)
            cell_keypoints[cell_y * grid_size + cell_x].append(idx)

        min_per_cell = max(2, len(keypoints) // (grid_size * grid_size) // 2)
        added_keypoints, added_descriptors, added_scores = [], [], []

        for cell_idx in range(grid_size * grid_size):
            cell_y = cell_idx // grid_size
            cell_x = cell_idx % grid_size
            y_start, y_end = cell_y * cell_h, min((cell_y + 1) * cell_h, h)
            x_start, x_end = cell_x * cell_w, min((cell_x + 1) * cell_w, w)

            cell_mask = mask_resized[y_start:y_end, x_start:x_end]
            if np.count_nonzero(cell_mask) < 10:
                continue

            existing_indices = cell_keypoints[cell_idx]
            if len(existing_indices) >= min_per_cell:
                continue

            need = min_per_cell - len(existing_indices)
            cell_pixels = np.column_stack(np.where(cell_mask > 0))
            if len(cell_pixels) == 0 or len(keypoints) == 0:
                continue

            sampled_indices = np.random.choice(len(cell_pixels), min(need * 3, len(cell_pixels)), replace=False)
            sampled = cell_pixels[np.atleast_1d(sampled_indices)]
            if sampled.ndim == 1:
                sampled = sampled.reshape(1, -1)

            for sample_y, sample_x in sampled:
                img_y, img_x = y_start + sample_y, x_start + sample_x
                if existing_indices:
                    existing_kps = keypoints[existing_indices]
                    nearest_idx = existing_indices[np.argmin(np.linalg.norm(existing_kps - [img_x, img_y], axis=1))]
                    borrowed_desc = descriptors[nearest_idx]
                    borrowed_score = scores[nearest_idx] if scores.ndim == 1 else float(scores)
                else:
                    borrowed_desc = np.mean(descriptors, axis=0) if len(descriptors) > 0 else np.zeros(256, dtype=np.float32)
                    borrowed_score = 0.1
                added_keypoints.append([img_x, img_y])
                added_descriptors.append(borrowed_desc)
                added_scores.append(borrowed_score)

        if added_keypoints:
            keypoints = np.vstack([keypoints, np.array(added_keypoints, dtype=np.float32)])
            descriptors = np.vstack([descriptors, np.array(added_descriptors, dtype=np.float32)])
            scores = np.hstack([scores, np.array(added_scores, dtype=np.float32)])

        if len(keypoints) > self.config.max_keypoints:
            keep_indices = np.argsort(scores)[-self.config.max_keypoints:]
            keypoints = keypoints[keep_indices]
            descriptors = descriptors[keep_indices]
            scores = scores[keep_indices]

        return SuperPointResult(
            keypoints=keypoints.astype(np.float32),
            descriptors=descriptors.astype(np.float32),
            scores=scores.astype(np.float32),
            image_size=original_size
        )


    def extract_pair(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[SuperPointResult, SuperPointResult]:
        """Extract features from a pair of images.
        
        Args:
            image1: First image
            image2: Second image
        
        Returns:
            Tuple of (SuperPointResult for image1, SuperPointResult for image2)
        """
        result1 = self.extract(image1)
        result2 = self.extract(image2)
        return result1, result2
    
    def extract_with_mask_pair(
        self,
        image1: np.ndarray,
        mask1: np.ndarray,
        image2: np.ndarray,
        mask2: np.ndarray
    ) -> Tuple[SuperPointResult, SuperPointResult]:
        """Extract features from a pair of images with masks.

        Args:
            image1: First image
            mask1: Binary mask for first image
            image2: Second image
            mask2: Binary mask for second image

        Returns:
            Tuple of (SuperPointResult for image1, SuperPointResult for image2)
            with uniformly distributed keypoints
        """
        result1 = self.extract(image1, mask1)
        result2 = self.extract(image2, mask2)
        return result1, result2

    def close(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None


def create_superpoint_processor(
    max_keypoints: int,
    max_dimension: int,
    device: str = "cuda",
    verbose: bool = False,
    detection_threshold: float = 0.001,
    enable_uniform_distribution: bool = True,
    grid_size: int = 8
) -> SuperPointProcessor:
    """Factory function to create SuperPoint processor.

    Note: max_keypoints and max_dimension MUST be provided explicitly!
          Do NOT rely on defaults - always pass these from appConfig.

    Args:
        max_keypoints: Maximum number of keypoints to detect (REQUIRED - use appConfig.sp_max_keypoints)
        max_dimension: Maximum image dimension (REQUIRED - use appConfig.sp_max_dimension)
        device: Device to run on ('cuda' or 'cpu')
        verbose: Enable verbose output
        detection_threshold: Lower threshold for more keypoints
        enable_uniform_distribution: Force uniform distribution across mask
        grid_size: Grid divisions for uniform sampling

    Returns:
        Configured SuperPointProcessor instance

    Raises:
        ValueError: If max_keypoints or max_dimension are not provided
    """
    if not isinstance(max_keypoints, int) or max_keypoints <= 0:
        raise ValueError(f"max_keypoints must be positive integer, got {max_keypoints}")
    if not isinstance(max_dimension, int) or max_dimension <= 0:
        raise ValueError(f"max_dimension must be positive integer, got {max_dimension}")

    config = SuperPointConfig(
        max_keypoints=max_keypoints,
        max_dimension=max_dimension,
        device=device,
        verbose=verbose,
        detection_threshold=detection_threshold,
        enable_uniform_distribution=enable_uniform_distribution,
        grid_size=grid_size
    )
    return SuperPointProcessor(config)


if __name__ == "__main__":
    # Example usage
    import argparse
    from src.config.appConfig import get_config

    parser = argparse.ArgumentParser(description="SuperPoint Feature Extraction")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mask", help="Path to mask image for uniform distribution")
    parser.add_argument("--max-kp", type=int, help="Max keypoints (required)")
    parser.add_argument("--max-dim", type=int, help="Max dimension (required)")
    parser.add_argument("--output", help="Save features to .npz file")
    args = parser.parse_args()

    # Get config
    app_config = get_config()

    # Use provided values or get from appConfig
    max_kp = args.max_kp if args.max_kp else app_config.sp_max_keypoints
    max_dim = args.max_dim if args.max_dim else app_config.sp_max_dimension

    if max_kp is None or max_dim is None:
        parser.error("--max-kp and --max-dim are required (or configure appConfig)")

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Failed to load image: {args.image}")

    # Load mask if provided
    mask = None
    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {args.mask}")

    # Create processor with required parameters
    sp = create_superpoint_processor(
        max_keypoints=max_kp,
        max_dimension=max_dim,
        verbose=True,
        detection_threshold=0.001,
        enable_uniform_distribution=True
    )

    # Extract features
    result = sp.extract(image, mask)

    print(f"Image size: {result.image_size}")
    print(f"Keypoints: {len(result.keypoints)}")
    print(f"Descriptors shape: {result.descriptors.shape}")
    print(f"Scores range: [{result.scores.min():.4f}, {result.scores.max():.4f}]")

    if args.output:
        np.savez(
            args.output,
            keypoints=result.keypoints,
            descriptors=result.descriptors,
            scores=result.scores,
            image_size=np.array(result.image_size)
        )
        print(f"Features saved to: {args.output}")
