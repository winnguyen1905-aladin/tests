#!/usr/bin/env python3
"""
SAM3 Frame-by-Frame Detection Script
Supports multiple detection modes: text, exemplar, point, box, and mask prompts.
Outputs individual processed images for each frame. For video tracking output, use sam3_video_tracker.py.
"""

import argparse
import cv2
import numpy as np
import os
import sys
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging

from src.config.appConfig import DetectionConfig, DetectionMode
from src.config.appConfig import get_config
# Comment out visualizer import as it's not essential for functionality
# from src.utils.visualizer import DetectionVisualizer

logger = logging.getLogger(__name__)

class SAM3Detector:
    """Main detector class for SAM3."""
    
    def __init__(self, config: DetectionConfig) -> None:
        """Initialize SAM3 detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        self.device = config.device
        # Skip visualizer initialization for now
        self.visualizer = None
        
        # Initialize model based on mode
        self._init_model()

        # Set memory optimization
        self._set_memory_optimization()

        print(f"✅ SAM3 detector initialized (device={self.device})")
    
    def _init_model(self) -> None:
        """Initialize SAM3 model based on detection mode."""
        if self.config.verbose:
            print(f"Initializing SAM3 model ({self.config.mode.value} mode)...")
            print(f"Model path: {self.config.model_path}")
            print(f"Device: {self.device}")
        
        # Check model file exists
        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {self.config.model_path}\n"
                "Please download sam3.pt from https://huggingface.co/facebook/sam3"
            )
        
        # Import appropriate model class based on mode
        if self.config.mode in [DetectionMode.TEXT, DetectionMode.EXEMPLAR]:
            # Enable AMD GPU experimental features if using ROCm (before import)
            try:
                if torch.cuda.is_available() and getattr(torch.version, 'hip', None) is not None:
                    os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')
            except Exception:
                pass  # Safe fallback

            # Use SAM3SemanticPredictor for concept segmentation
            from ultralytics.models.sam import SAM3SemanticPredictor

            overrides = dict(
                conf=self.config.confidence_threshold,
                task="segment",
                mode="predict",
                model=self.config.model_path,
                half=self.config.half_precision,
                save=False,  # We'll handle saving ourselves
                verbose=self.config.verbose,
                imgsz=644,  # Must be multiple of max stride 14 (14 * 46 = 644)
            )

            self.predictor = SAM3SemanticPredictor(overrides=overrides)
            
        else:
            # Use standard SAM for visual prompts (point, box, mask)
            from ultralytics import SAM
            
            self.model = SAM(self.config.model_path)
            self.predictor = None
        
        if self.config.verbose:
            print("Model initialized successfully!\n")

    def _set_memory_optimization(self) -> None:
        """Set PyTorch memory optimization settings based on appConfig model_mode."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            app_config = get_config()

            # Get model mode from appConfig
            model_mode = app_config.model_mode  # 'ultra' or 'lite'

            # Set GPU memory fraction based on model_mode
            if model_mode == "lite":
                torch_memory_fraction = 0.9
                max_detections = 50
            else:  # ultra mode (default)
                torch_memory_fraction = 1.0  # Full GPU for powerful cards
                max_detections = 100

            # Apply memory fraction only if limiting
            if torch_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(torch_memory_fraction)

            # Check if using AMD ROCm (not supported expandable_segments)
            is_rocm = False
            try:
                is_rocm = getattr(torch.version, 'hip', None) is not None
            except Exception:
                pass

            # Set PYTORCH_CUDA_ALLOC_CONF based on platform
            # expandable_segments is NOT supported on AMD ROCm
            if is_rocm:
                # AMD ROCm: use simpler settings
                os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256')
                logger.info("Detected AMD ROCm, using simplified memory config")
            else:
                # NVIDIA CUDA: use expandable_segments
                os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:256')

            # Enable memory-efficient attention backends (works on both NVIDIA and AMD)
            try:
                from torch.nn.attention import SDPBackend
                # Enable all efficient SDPA backends globally
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
                    torch.backends.cuda.enable_cudnn_sdp(True)
                logger.info("Enabled Flash/MemEfficient/Math/cuDNN SDPA backends")
            except Exception as e:
                logger.warning(f"Failed to enable SDPA backends: {e}")

            logger.info(f"Memory optimization enabled: Model Mode={model_mode}, GPU Memory={torch_memory_fraction * 100}%, Max Detections={max_detections}")

    def is_video(self, file_path: str) -> bool:
        """Check if the input file is a video.
        
        Args:
            file_path: Path to input file
            
        Returns:
            True if video, False otherwise
        """
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        return any(file_path.lower().endswith(ext) for ext in video_extensions)
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and validate input image or extract first frame from video.
        
        Args:
            image_path: Path to input image or video file
            
        Returns:
            Image as numpy array (RGB)
        """
        if self.is_video(image_path):
            if self.config.verbose:
                print(f"Video file detected, extracting first frame...")
            
            # Open video and read first frame
            cap = cv2.VideoCapture(image_path)
            ret, image = cap.read()
            cap.release()
            
            if not ret or image is None:
                raise ValueError(f"Failed to read video: {image_path}")
        else:
            # Losdfad image file
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def detect_by_text(self, image: np.ndarray, set_image: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect objects using text prompts.
        
        Args:
            image: Input image (RGB) or path to image/video
            set_image: Whether to call set_image (only needed once for video)
            
        Returns:
            Tuple of (masks, boxes, scores)
        """
        if self.config.verbose:
            print(f"Running text-based detection...")
            print(f"Text prompts: {self.config.text_prompts}")
        
        # Set image only if needed (to avoid reprocessing video)
        if set_image:
            # Use provided image if available, otherwise config input
            source = image if image is not None else self.config.input_image
            self.predictor.set_image(source)
        
        # Run detection with text prompts only
        results = self.predictor(text=self.config.text_prompts)
        
        # Extract results
        if results and hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
        else:
            masks = np.array([])
            boxes = np.array([])
            scores = np.array([])
        
        return masks, boxes, scores
    
    def refine_masks(self, masks: np.ndarray) -> np.ndarray:
        """Refine masks using morphological operations to reduce oversegmentation.
        
        Args:
            masks: Input masks
            
        Returns:
            Refined masks
        """
        if not self.config.refine_masks or len(masks) == 0:
            return masks
        
        refined_masks = []
        kernel_size = self.config.erosion_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        for mask in masks:
            # Convert to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Erosion to shrink mask
            eroded = cv2.erode(mask_uint8, kernel, iterations=1)
            
            # Convert back to float
            refined = (eroded > 127).astype(np.float32)
            refined_masks.append(refined)
        
        if self.config.verbose and len(refined_masks) > 0:
            print(f"Refined {len(refined_masks)} masks with erosion kernel size {kernel_size}")
        
        return np.array(refined_masks)
    
    def detect_by_exemplar(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect objects using image exemplars (bounding boxes).

        Args:
            image: Input image (RGB)

        Returns:
            Tuple of (masks, boxes, scores)
        """
        if self.config.verbose:
            print(f"Running exemplar-based detection...")
            print(f"Exemplar boxes: {self.config.exemplar_boxes}")

        # Set image - use the image parameter
        self.predictor.set_image(image)

        # Run detection
        results = self.predictor(bboxes=self.config.exemplar_boxes)

        # Extract results
        if results and hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
        else:
            masks = np.array([])
            boxes = np.array([])
            scores = np.array([])

        return masks, boxes, scores
    
    def detect_by_points(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect objects using point prompts.

        Args:
            image: Input image (RGB)

        Returns:
            Tuple of (masks, boxes, scores)
        """
        if self.config.verbose:
            print(f"Running point-based detection...")
            print(f"Points: {self.config.points}")
            print(f"Labels: {self.config.point_labels}")

        # Run detection - use the image parameter
        results = self.model.predict(
            source=image,
            points=self.config.points,
            labels=self.config.point_labels,
        )

        # Extract results
        if results and hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else np.array([])
            scores = results[0].boxes.conf.cpu().numpy() if results[0].boxes else np.ones(len(masks))
        else:
            masks = np.array([])
            boxes = np.array([])
            scores = np.array([])

        return masks, boxes, scores
    
    def detect_by_boxes(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect objects using bounding box prompts.

        Args:
            image: Input image (RGB)

        Returns:
            Tuple of (masks, boxes, scores)
        """
        if self.config.verbose:
            print(f"Running box-based detection...")
            print(f"Bounding boxes: {self.config.bounding_boxes}")

        # Run detection - use the image parameter
        results = self.model.predict(
            source=image,
            bboxes=self.config.bounding_boxes,
        )

        # Extract results
        if results and hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else np.array(self.config.bounding_boxes)
            scores = results[0].boxes.conf.cpu().numpy() if results[0].boxes else np.ones(len(masks))
        else:
            masks = np.array([])
            boxes = np.array(self.config.bounding_boxes)
            scores = np.array([])

        return masks, boxes, scores
    
    def detect_by_masks(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect objects using mask prompts.

        Args:
            image: Input image (RGB)

        Returns:
            Tuple of (masks, boxes, scores)
        """
        if self.config.verbose:
            print(f"Running mask-based detection...")
            print(f"Input masks: {self.config.input_masks}")

        # Load input masks
        input_mask = cv2.imread(self.config.input_masks, cv2.IMREAD_GRAYSCALE)
        if input_mask is None:
            raise ValueError(f"Failed to load mask: {self.config.input_masks}")

        # Run detection - use the image parameter
        results = self.model.predict(
            source=image,
            masks=input_mask,
        )

        # Extract results
        if results and hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else np.array([])
            scores = results[0].boxes.conf.cpu().numpy() if results[0].boxes else np.ones(len(masks))
        else:
            masks = np.array([])
            boxes = np.array([])
            scores = np.array([])

        return masks, boxes, scores
    
    def process_video_frames(self) -> Tuple[List[str], Dict[str, Any]]:
        """Process video by extracting frames first, then detecting on each frame image.
        
        Returns:
            Tuple of (list of output image paths, first_frame_results)
        """
        if self.config.verbose:
            print(f"Processing video frame-by-frame with SAM3SemanticPredictor...")
        
        # Open video
        cap = cv2.VideoCapture(self.config.input_image)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.config.verbose:
            print(f"Video: {total_frames} frames @ {fps} FPS ({width}x{height})")
            print(f"Step 1: Extracting all frames from video...")
        
        # Prepare output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        raw_frames_dir = output_dir / "raw_frames"
        raw_frames_dir.mkdir(exist_ok=True)
        
        input_path = Path(self.config.input_image)
        base_name = input_path.stem
        
        # Step 1: Extract all frames from video
        frames = []
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                
                # Save raw frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_frame_path = raw_frames_dir / f"{base_name}_raw_frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(raw_frame_path), frame)
                frames.append((frame_idx, frame_rgb, str(raw_frame_path)))
                
                if self.config.verbose and frame_idx % 30 == 0:
                    print(f"  Extracted {frame_idx}/{total_frames} frames...")
        finally:
            cap.release()
        
        if self.config.verbose:
            print(f"✓ Extracted {len(frames)} frames")
            print(f"Step 2: Running detection on each frame...")
        
        # Step 2: Run detection on each extracted frame
        first_frame_results = None
        output_paths = []
        
        for idx, (frame_num, frame_rgb, raw_path) in enumerate(frames, 1):
            if self.config.verbose and idx % 30 == 0:
                print(f"  Processing frame {idx}/{len(frames)}...")
            
            # Run detection on this frame as an individual image
            if self.config.mode == DetectionMode.TEXT:
                masks, boxes, scores = self.detect_by_text(frame_rgb)
                masks = self.refine_masks(masks)
                n_prompts = max(1, len(self.config.text_prompts))
                labels = self.config.text_prompts * (len(masks) // n_prompts + 1)
                labels = labels[:len(masks)]
            else:
                # For other modes, would need to implement similarly
                masks, boxes, scores, labels = np.array([]), np.array([]), np.array([]), []
            
            # Store first frame results
            if idx == 1:
                annotated = None
                if len(masks) > 0:
                    annotated = self.visualizer.overlay_masks(
                        image=frame_rgb, masks=masks, boxes=boxes, scores=scores, labels=labels,
                    )
                first_frame_results = {
                    'image': frame_rgb, 'masks': masks, 'boxes': boxes,
                    'scores': scores, 'labels': labels, 'annotated_image': annotated,
                }
            
            # Save annotated frame image
            if len(masks) > 0 and self.config.save_visualization:
                annotated = self.visualizer.overlay_masks(
                    image=frame_rgb, masks=masks, boxes=boxes, scores=scores, labels=labels,
                )
                frame_path = frames_dir / f"{base_name}_frame_{frame_num:05d}.jpg"
                self.visualizer.save_visualization(annotated, str(frame_path))
                output_paths.append(str(frame_path))
        
        if self.config.verbose:
            print(f"✓ Saved {len(output_paths)} processed frame images to: {frames_dir}")
        
        return output_paths, first_frame_results
    
    def run(self) -> Dict[str, Any]:
        """Run detection based on configured mode.
        
        Returns:
            Dictionary containing detection results
        """
        # Check if input is a video
        if self.is_video(self.config.input_image):
            if self.config.verbose:
                print("Video input detected. Processing frames individually...\n")
            
            # Process video frames and output individual images
            output_frame_paths, first_frame_data = self.process_video_frames()
            
            # Use first frame results for return value
            image = first_frame_data['image']
            masks = first_frame_data['masks']
            boxes = first_frame_data['boxes']
            scores = first_frame_data['scores']
            labels = first_frame_data['labels']
            annotated_image = first_frame_data['annotated_image']
            
        else:
            if self.config.verbose:
                print("Image input detected. Processing image...\n")
            
            output_frame_paths = []
            
            # Load image
            image = self.load_image(self.config.input_image)
            
            # Run detection based on mode
            if self.config.mode == DetectionMode.TEXT:
                masks, boxes, scores = self.detect_by_text(image)
                masks = self.refine_masks(masks)
                labels = self.config.text_prompts * (len(masks) // len(self.config.text_prompts) + 1)
                labels = labels[:len(masks)]
            
            elif self.config.mode == DetectionMode.EXEMPLAR:
                masks, boxes, scores = self.detect_by_exemplar(image)
                masks = self.refine_masks(masks)
                labels = [f"Object {i+1}" for i in range(len(masks))]
            
            elif self.config.mode == DetectionMode.POINT:
                masks, boxes, scores = self.detect_by_points(image)
                masks = self.refine_masks(masks)
                labels = [f"Object {i+1}" for i in range(len(masks))]
            
            elif self.config.mode == DetectionMode.BOX:
                masks, boxes, scores = self.detect_by_boxes(image)
                masks = self.refine_masks(masks)
                labels = [f"Object {i+1}" for i in range(len(masks))]
            
            elif self.config.mode == DetectionMode.MASK:
                masks, boxes, scores = self.detect_by_masks(image)
                masks = self.refine_masks(masks)
                labels = [f"Object {i+1}" for i in range(len(masks))]
            
            else:
                raise ValueError(f"Unknown detection mode: {self.config.mode}")
            
            # Create visualization
            annotated_image = None
            if self.config.save_visualization and len(masks) > 0:
                annotated_image = self.visualizer.overlay_masks(
                    image=image,
                    masks=masks,
                    boxes=boxes if len(boxes) > 0 else None,
                    scores=scores if len(scores) > 0 else None,
                    labels=labels if len(labels) > 0 else None,
                )
        
        # Print statistics
        if self.config.verbose:
            self.visualizer.print_statistics(
                n_detections=len(masks),
                mode=self.config.mode.value,
                scores=scores if len(scores) > 0 else None,
                labels=labels if len(labels) > 0 else None,
            )
        
        # Save results (image outputs only)
        if not self.is_video(self.config.input_image):
            self._save_results(image, masks, boxes, scores, annotated_image)
        
        # Clear memory before returning
        torch.cuda.empty_cache()

        return {
            'masks': masks,
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'image': image,
            'annotated_image': annotated_image,
            'output_video': None,  # Detector outputs images only, not compiled video
            'output_frame_paths': output_frame_paths,  # List of individual frame image paths
        }

    def detect_batch(self, images: list, batch_size: int = 4) -> list:
        """Detect objects in multiple images using batch processing.

        Args:
            images: List of input images (RGB format)
            batch_size: Batch size for GPU processing (default: 4 for SAM3 memory efficiency)

        Returns:
            List of detection results, each containing masks, boxes, scores
        """
        if not images:
            return []

        all_results = []
        device = self.device

        if self.config.verbose:
            print(f"[SAM3 Batch] Processing {len(images)} images with batch_size={batch_size}")

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_start_idx = i

            if self.config.verbose:
                print(f"[SAM3 Batch] Processing batch {batch_start_idx}-{batch_start_idx + len(batch_images) - 1}")

            # Process each image in batch
            # Note: SAM3 doesn't support true batch inference in the current implementation
            # But we can optimize by keeping model on GPU and processing sequentially
            batch_results = []

            for idx, image in enumerate(batch_images):
                try:
                    # Run detection based on mode
                    if self.config.mode == DetectionMode.TEXT:
                        masks, boxes, scores = self.detect_by_text(image)
                    elif self.config.mode == DetectionMode.EXEMPLAR:
                        masks, boxes, scores = self.detect_by_exemplar(image)
                    elif self.config.mode == DetectionMode.POINT:
                        masks, boxes, scores = self.detect_by_points(image)
                    elif self.config.mode == DetectionMode.BOX:
                        masks, boxes, scores = self.detect_by_boxes(image)
                    elif self.config.mode == DetectionMode.MASK:
                        masks, boxes, scores = self.detect_by_masks(image)
                    else:
                        raise ValueError(f"Unknown detection mode: {self.config.mode}")

                    # Refine masks
                    masks = self.refine_masks(masks)

                    # Generate labels
                    if self.config.mode == DetectionMode.TEXT:
                        n_prompts = max(1, len(self.config.text_prompts))
                        labels = self.config.text_prompts * (len(masks) // n_prompts + 1)
                        labels = labels[:len(masks)]
                    else:
                        labels = [f"Object {j+1}" for j in range(len(masks))]

                    batch_results.append({
                        'masks': masks,
                        'boxes': boxes,
                        'scores': scores,
                        'labels': labels,
                        'image': image
                    })

                except Exception as e:
                    print(f"[SAM3 Batch] Error processing image {idx}: {e}")
                    # Return empty result for failed image
                    batch_results.append({
                        'masks': np.array([]),
                        'boxes': np.array([]),
                        'scores': np.array([]),
                        'labels': [],
                        'image': image
                    })

            all_results.extend(batch_results)

        if self.config.verbose:
            print(f"[SAM3 Batch] Processed {len(all_results)} images")

        return all_results

    
    def _save_results(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        annotated_image: Optional[np.ndarray],
    ) -> None:
        """Save detection results to disk.
        
        Args:
            image: Original image
            masks: Detection masks
            boxes: Bounding boxes
            scores: Confidence scores
            annotated_image: Annotated image with visualizations
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get base filename
        input_path = Path(self.config.input_image)
        base_name = input_path.stem
        
        # Save visualization
        if self.config.save_visualization and annotated_image is not None:
            vis_path = output_dir / f"{base_name}_annotated.jpg"
            self.visualizer.save_visualization(annotated_image, str(vis_path))
            if self.config.verbose:
                print(f"Saved visualization: {vis_path}")
        
        # Save masks
        if self.config.save_masks and len(masks) > 0:
            masks_dir = output_dir / "masks"
            masks_dir.mkdir(exist_ok=True)
            
            for i, mask in enumerate(masks):
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_path = masks_dir / f"{base_name}_mask_{i:03d}.png"
                cv2.imwrite(str(mask_path), mask_uint8)
            
            if self.config.verbose:
                print(f"Saved {len(masks)} masks to: {masks_dir}")
        
        # Save boxes
        if self.config.save_boxes and len(boxes) > 0:
            boxes_path = output_dir / f"{base_name}_boxes.txt"
            with open(boxes_path, 'w') as f:
                for i, (box, score) in enumerate(zip(boxes, scores)):
                    f.write(f"{i} {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f} {score:.4f}\n")
            
            if self.config.verbose:
                print(f"Saved boxes: {boxes_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SAM3 Image Detection Script - For video processing, use sam3_video_tracker.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-based detection on image
  python sam3_detector.py --config configs/text_detection.yaml
  
  # Exemplar-based detection
  python sam3_detector.py --config configs/exemplar_detection.yaml
  
  # Point-based detection with custom confidence threshold
  python sam3_detector.py --config configs/point_detection.yaml --conf 0.5
  
  # Box-based detection with custom output directory
  python sam3_detector.py --config configs/box_detection.yaml --output-dir results/

Note: This script only processes images. For video tracking, use sam3_video_tracker.py instead.
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    # Override options
    parser.add_argument("--input", type=str, help="Override input image path")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--conf", type=float, help="Override confidence threshold")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Override device")
    parser.add_argument("--model-path", type=str, help="Override model path")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = DetectionConfig.from_yaml(args.config)
        
        # Apply overrides
        if args.input:
            config.input_image = args.input
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.conf is not None:
            config.confidence_threshold = args.conf
        if args.device:
            config.device = args.device
        if args.model_path:
            config.model_path = args.model_path
        if args.no_viz:
            config.save_visualization = False
        if args.quiet:
            config.verbose = False
        
        # Re-validate after overrides
        config.__post_init__()
        
        # Run detection
        detector = SAM3Detector(config)
        results = detector.run()
        
        if config.verbose:
            print("\n✓ Detection completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())