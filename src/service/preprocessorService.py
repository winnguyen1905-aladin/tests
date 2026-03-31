"""
Preprocessor Service - Image preprocessing utilities

Handles:
1. SAM3 Semantic Segmentation (durian fruit detection)
2. Mask application (remove background)
3. Bounding box extraction
4. Image cropping
5. Format conversion for different models

Uses @inject decorator for dependency injection with dependency_injector.
Wire with: container.wire(modules=["src.service.preprocessorService"])
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging
import torch
import os

from dependency_injector.wiring import inject, Provide

from src.processor.sam3ImageProcessor import SAM3Detector
from src.processor.sam3ImageProcessor import DetectionConfig, DetectionMode
from src.config.appConfig import AppConfig, get_config
import torch

# Set memory optimization environment variables based on platform
# expandable_segments is NOT supported on AMD ROCm
is_rocm = False
try:
    is_rocm = getattr(torch.version, 'hip', None) is not None
except Exception:
    pass

if is_rocm:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
else:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'

logger = logging.getLogger(__name__)

class PreprocessorService:
    """Service for image preprocessing operations.
    
    Uses @inject decorator for dependency injection with dependency_injector.
    Wire with: container.wire(modules=["src.service.preprocessorService"])
    """

    @inject
    def __init__(
        self,
        app_config: AppConfig = Provide["app_config"],
        background_color: str = "black",
        sam3_model_path: Optional[str] = None,
    ) -> None:
        """Initialize preprocessor with injected dependencies.

        Args:
            app_config: Application configuration (injected via Provide)
            background_color: "black" or "white" for background fill
            sam3_model_path: Path to SAM3 model file for semantic segmentation
        """
        self.app_config: AppConfig = app_config
        self.background_color: str = background_color
        self.sam3_model_path: str = sam3_model_path or "sam3.pt"
        self.sam3_detector = None

    def _init_sam3(self) -> None:
        """Initialize SAM3 detector for semantic segmentation."""
        try:
            # Use get_config() directly — self.app_config may be a Provide proxy
            # when this service is instantiated outside the DI container (e.g. in
            # standalone scripts like test_segmentation.py).
            app_config = get_config()

            # Force CUDA if available - no fallback to CPU for maximum performance
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")

            # Use config values directly
            config = DetectionConfig(
                input_image="",  # Will be set dynamically
                mode=DetectionMode.TEXT,
                text_prompts=app_config.sam3_text_prompts,
                confidence_threshold=app_config.sam3_confidence,
                device=device,
                model_path=self.sam3_model_path,
                verbose=False,
                half_precision=True,
                max_detections=100,
                refine_masks=True,
                erosion_kernel_size=3
            )

            self.sam3_detector = SAM3Detector(config)
            if hasattr(self.sam3_detector, 'predictor') and self.sam3_detector.predictor is not None:
                logger.info(f"SAM3 detector initialized (confidence={app_config.sam3_confidence})")
            else:
                logger.warning("SAM3 initialized but not in semantic predictor mode")

        except ImportError as e:
            logger.error(f"Failed to initialize SAM3: {e}")
            self.sam3_detector = None
        except Exception as e:
            logger.error(f"Error initializing SAM3: {e}")
            self.sam3_detector = None
    
    def segment_with_sam3(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use SAM3SemanticPredictor for semantic segmentation of durian fruits.

        Args:
            image: Input image (BGR format)

        Returns:
            Tuple of (segmented_image, mask)
            - segmented_image: Image with background removed (durian fruit only)
            - mask: Binary mask where 255 = durian fruit, 0 = background
        """
        if self.sam3_detector is None:
            self._init_sam3()

        if self.sam3_detector is None:
            logger.warning("SAM3 initialization failed, returning full image with full mask")
            return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

        try:
            # Use get_config() directly — self.app_config may be a Provide proxy
            app_config = get_config()

            # Clear GPU cache before processing (lite mode = aggressive cleanup, ultra mode = minimal)
            if app_config.model_mode == "lite" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Convert BGR to RGB for SAM3
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Set the image in SAM3 detector (TEXT mode uses self.predictor)
            self.sam3_detector.predictor.set_image(image_rgb)

            # Run text-based detection for durian (set_image=False since we already set it)
            masks, _, scores = self.sam3_detector.detect_by_text(image_rgb, set_image=False)

            if len(masks) == 0:
                logger.warning("No durian fruits detected by SAM3, using full image mask")
                return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

            # Combine ALL masks above confidence threshold (multiple durian fruits)
            combined_mask = np.zeros(masks[0].shape, dtype=np.uint8)
            valid_count = 0
            for i, (m, s) in enumerate(zip(masks, scores)):
                if s >= 0.3:
                    combined_mask = np.maximum(combined_mask, (m * 255).astype(np.uint8))
                    valid_count += 1
                    logger.info(f"  Durian detection {i}: confidence={s:.3f}")

            if valid_count == 0:
                # Fallback: use best mask regardless of threshold
                best_idx = np.argmax(scores)
                combined_mask = (masks[best_idx] * 255).astype(np.uint8)
                valid_count = 1
                logger.info(f"  Fallback to best mask: confidence={scores[best_idx]:.3f}")

            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            # Create segmented image
            segmented = self.apply_mask(image, combined_mask)

            # Clear GPU memory only in lite mode
            if app_config.model_mode == "lite" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"SAM3 detected <{valid_count}> durian fruit(s) from {len(masks)} detections")

            return segmented, combined_mask

        except Exception as e:
            logger.error(f"SAM3 segmentation failed: {e}")
            logger.warning("Falling back to full image with full mask")
            return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

    # ─────────────────────────────────────────────────────────────────────────
    # SAM3 Automatic (everything) mode — no CLIP required
    # Uses SAMPredictor in automatic mode to find salient foreground objects.
    # This is the fallback when CLIP is not available.
    # ─────────────────────────────────────────────────────────────────────────

    # def segment_with_sam3_auto(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     """Use SAM3 in automatic mode for foreground segmentation (no CLIP needed).

    #     Automatically finds and segments the most salient foreground object(s) in the
    #     image.  This method does NOT require the `clip` package — it uses
    #     SAMPredictor with automatic everything-mode.

    #     Args:
    #         image: Input image (BGR format)

    #     Returns:
    #         Tuple of (segmented_image, mask)
    #         - segmented_image: Image with background zeroed
    #         - mask: Binary mask where 255 = foreground, 0 = background
    #     """
    #     try:
    #         from ultralytics.models.sam import SAMPredictor

    #         # Check model file
    #         model_path = self.sam3_model_path or "sam3.pt"
    #         if not os.path.exists(model_path):
    #             logger.warning(f"SAM3 model not found at {model_path}, returning full mask")
    #             return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

    #         # Create predictor once and cache it
    #         if not hasattr(self, "_sam_auto_predictor") or self._sam_auto_predictor is None:
    #             self._sam_auto_predictor = SAMPredictor(model=model_path)

    #         # Convert BGR → RGB
    #         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #         # Run automatic everything-mode: SAM3 generates its own prompts internally
    #         results = self._sam_auto_predictor(image_rgb)

    #         if results is None or not hasattr(results[0], "masks") or results[0].masks is None:
    #             logger.warning("SAM3 auto mode returned no masks, using full mask")
    #             return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

    #         # Extract masks
    #         raw_masks = results[0].masks.data.cpu().numpy()  # (N, H, W), float32 in [0,1]

    #         # Combine all masks
    #         combined_mask = np.zeros(raw_masks.shape[1:], dtype=np.uint8)
    #         for m in raw_masks:
    #             combined_mask = np.maximum(combined_mask, (m * 255).astype(np.uint8))

    #         # Morphological cleanup
    #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #         combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    #         combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    #         # Resize mask back to original image size if SAM resized internally
    #         h, w = image.shape[:2]
    #         if combined_mask.shape != (h, w):
    #             combined_mask = cv2.resize(combined_mask, (w, h),
    #                                        interpolation=cv2.INTER_NEAREST)

    #         segmented = self.apply_mask(image, combined_mask)

    #         coverage = (combined_mask > 0).sum() / combined_mask.size * 100
    #         logger.info(f"SAM3 auto mode: {len(raw_masks)} mask(s), coverage={coverage:.1f}%")

    #         return segmented, combined_mask

    #     except ImportError as e:
    #         logger.error(f"Import error in SAM3 auto mode: {e}")
    #         return image, np.ones(image.shape[:2], dtype=np.uint8) * 255
    #     except Exception as e:
    #         logger.error(f"SAM3 auto mode failed: {e}")
    #         return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

    def segment_with_sam3_box(self, image: np.ndarray, bounding_boxes: list) -> Tuple[np.ndarray, np.ndarray]:
        """Use SAM3 box prompt mode for segmentation based on bounding box coordinates.

        Args:
            image: Input image (BGR format)
            bounding_boxes: List of bounding boxes [[x1, y1, x2, y2], ...]

        Returns:
            Tuple of (segmented_image, mask)
            - segmented_image: Image with background removed
            - mask: Binary mask where 255 = object, 0 = background
        """
        if self.sam3_detector is None:
            self._init_sam3()

        if self.sam3_detector is None:
            logger.warning("SAM3 initialization failed, returning full image with full mask")
            return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

        try:
            # Use get_config() directly — self.app_config may be a Provide proxy
            app_config = get_config()

            # Clear GPU cache before processing only in lite mode
            if app_config.model_mode == "lite":
                torch.cuda.empty_cache()

            # Convert BGR to RGB for SAM3
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Temporarily switch to BOX mode and set bounding boxes
            original_mode = self.sam3_detector.config.mode
            self.sam3_detector.config.mode = DetectionMode.BOX
            self.sam3_detector.config.bounding_boxes = bounding_boxes

            # Re-initialize model for BOX mode if needed (BOX mode uses self.model, not self.predictor)
            if not hasattr(self.sam3_detector, 'model') or self.sam3_detector.model is None:
                from ultralytics import SAM
                self.sam3_detector.model = SAM(self.sam3_detector.config.model_path)

            # Run box-based detection (image_rgb is passed directly to detect_by_boxes)
            masks, _, scores = self.sam3_detector.detect_by_boxes(image_rgb)

            # Restore original mode
            self.sam3_detector.config.mode = original_mode

            if len(masks) == 0:
                logger.warning("No objects detected by SAM3 box prompt, using full image mask")
                return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

            # Combine all masks
            combined_mask = np.zeros(masks[0].shape, dtype=np.uint8)
            for i, (m, s) in enumerate(zip(masks, scores)):
                combined_mask = np.maximum(combined_mask, (m * 255).astype(np.uint8))
                logger.info(f"  Box detection {i}: confidence={s:.3f}")

            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            # Create segmented image
            segmented = self.apply_mask(image, combined_mask)

            # Clear GPU memory only in lite mode
            if app_config.model_mode == "lite":
                torch.cuda.empty_cache()

            logger.info(f"SAM3 box prompt detected {len(masks)} object(s)")

            return segmented, combined_mask

        except Exception as e:
            logger.error(f"SAM3 box segmentation failed: {e}")
            logger.warning("Falling back to full image with full mask")
            return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

    # def set_prompts(self, prompts: list) -> None:
    #     """Set text prompts for SAM3 segmentation, keeping appConfig as source of truth.

    #     Args:
    #         prompts: List of text prompts (e.g. ["the large tree in the center"])
    #     """
    #     app_config = get_config()
    #     app_config.sam3_text_prompts = prompts
    #     if self.sam3_detector is not None:
    #         self.sam3_detector.config.text_prompts = prompts
    #         logger.info(f"Set SAM3 prompts to: {prompts}")
    #     else:
    #         logger.warning("SAM3 not initialized, prompts saved to config only")

    # def set_prompt(self, prompt: str) -> None:
    #     """Set a single text prompt for SAM3 segmentation.

    #     Args:
    #         prompt: Text prompt for detection (e.g. 'durian fruit')
    #     """
    #     self.set_prompts([prompt])

    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to image (remove background).

        Args:
            image: Input image (BGR)
            mask: Binary mask (0-255 or 0-1)

        Returns:
            Masked image with background removed
        """
        # Normalize mask to 0-1 range
        mask_float: np.ndarray
        if mask.max() > 1:
            mask_float = (mask > 127).astype(np.float32)
        else:
            mask_float = mask.astype(np.float32)

        # Expand mask to 3 channels if needed
        if len(mask_float.shape) == 2:
            mask_float = np.stack([mask_float] * 3, axis=-1)

        # Create background
        bg: np.ndarray = np.zeros_like(image) if self.background_color == "black" else np.ones_like(image) * 255

        # Apply mask
        return np.where(mask_float > 0, image.astype(np.float32), bg.astype(np.float32)).astype(np.uint8)
    
    def get_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Extract bounding box from mask.

        Args:
            mask: Binary mask

        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        # Normalize mask
        mask_uint8: np.ndarray
        mask_normalized = mask if isinstance(mask, np.ndarray) else np.array(mask)
        if mask_normalized.max() > 1:
            mask_uint8 = (mask_normalized > 127).astype(np.uint8)
        else:
            mask_uint8 = mask_normalized.astype(np.uint8)

        # Convert to single channel if needed
        if len(mask_uint8.shape) == 3:
            mask_uint8 = mask_uint8[:, :, 0]

        # Find contours
        contours: list
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in mask")

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        return x, y, x + w, y + h
    
    def crop_to_bounding_box(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """Crop image and mask to bounding box.

        Args:
            image: Input image
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            mask: Optional mask to crop (if None, creates full mask)

        Returns:
            Tuple of (cropped_image, cropped_mask, bbox_coords)
        """
        x_min, y_min, x_max, y_max = bbox
        cropped_image: np.ndarray = image[y_min:y_max, x_min:x_max]

        # Crop mask if provided, otherwise create full mask
        if mask is not None:
            cropped_mask: np.ndarray = mask[y_min:y_max, x_min:x_max]
        else:
            cropped_mask: np.ndarray = np.ones((y_max - y_min, x_max - x_min), dtype=np.uint8) * 255

        return cropped_image, cropped_mask, (x_min, y_min, x_max, y_max)
    
    def prepare_for_dino(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Prepare image for DinoV2/V3 model.

        Args:
            image: Input image (BGR)
            target_size: Target size for model (defaults to appConfig.dino_image_size)

        Returns:
            Resized image ready for DinoV2/V3
        """
        if target_size is None:
            # Use get_config() directly — self.app_config may be a Provide proxy
            # when this service is instantiated outside the DI container (e.g. in
            # standalone scripts like test_segmentation.py).
            size = get_config().dino_image_size
            target_size = (size, size)

        # Resize to target size
        resized: np.ndarray = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

        # Convert BGR to RGB
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    def prepare_for_superpoint(
        self,
        image: np.ndarray,
        max_dimension: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """Prepare image for SuperPoint (keep high resolution).

        Args:
            image: Input image
            max_dimension: Maximum dimension to keep (defaults to appConfig.sp_max_dimension)

        Returns:
            Tuple of (resized_image, scale_factor)
        """
        if max_dimension is None:
            max_dimension = get_config().sp_max_dimension
        h, w = image.shape[:2]

        if max(h, w) > max_dimension:
            scale: float = max_dimension / max(h, w)
            resized: np.ndarray = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC
            )
            return resized, scale

        return image, 1.0
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale.

        Args:
            image: Input image (BGR or RGB)

        Returns:
            Grayscale image
        """
        grayscale_image: np.ndarray
        if len(image.shape) == 3 and image.shape[2] == 3:
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_image = image
        return grayscale_image
    
    # def normalize_image(self, image: np.ndarray) -> np.ndarray:
    #     """Normalize image to 0-1 range.

    #     Args:
    #         image: Input image

    #     Returns:
    #         Normalized image
    #     """
    #     if image.dtype == np.uint8:
    #         normalized: np.ndarray = image.astype(np.float32) / 255.0
    #         return normalized
    #     return image
