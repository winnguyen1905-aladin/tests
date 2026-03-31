"""
Visualization utilities for SAM3 detection results.
"""

import cv2
import numpy as np
import matplotlib
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image


class DetectionVisualizer:
    """Visualize SAM3 detection results."""
    
    def __init__(
        self,
        colormap: str = "rainbow",
        alpha: float = 0.5,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_scores: bool = True,
    ) -> None:
        """Initialize visualizer.
        
        Args:
            colormap: Matplotlib colormap name
            alpha: Transparency for mask overlay (0-1)
            show_boxes: Whether to draw bounding boxes
            show_labels: Whether to show labels
            show_scores: Whether to show confidence scores
        """
        self.colormap = colormap
        self.alpha = alpha
        self.show_boxes = show_boxes
        self.show_labels = show_labels
        self.show_scores = show_scores
    
    def overlay_masks(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        boxes: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Overlay detection masks on image.
        
        Args:
            image: Input image (RGB, uint8)
            masks: Binary masks [N, H, W]
            boxes: Bounding boxes [N, 4] in xyxy format
            scores: Confidence scores [N]
            labels: Text labels for each detection
            
        Returns:
            Annotated image with overlaid masks
        """
        if len(masks) == 0:
            return image
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Create a copy to avoid modifying original
        result = image.copy()
        
        # Get colors from colormap
        n_masks = masks.shape[0]
        cmap = matplotlib.colormaps.get_cmap(self.colormap).resampled(n_masks)
        colors = [
            tuple(int(c * 255) for c in cmap(i)[:3])[::-1]  # Convert to BGR
            for i in range(n_masks)
        ]
        
        # Overlay masks
        for i, (mask, color) in enumerate(zip(masks, colors)):
            # Create colored mask overlay
            mask_uint8 = (mask * 255).astype(np.uint8)
            colored_mask = np.zeros_like(result)
            colored_mask[mask > 0] = color
            
            # Blend with original image
            result = cv2.addWeighted(result, 1, colored_mask, self.alpha, 0)
            
            # Draw contours for better visibility
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, color, 2)
            
            # Draw bounding box if available
            if self.show_boxes and boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = boxes[i].astype(int)
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
                
                # Add label and score
                label_text = ""
                if self.show_labels and labels is not None and i < len(labels):
                    label_text = labels[i]
                if self.show_scores and scores is not None and i < len(scores):
                    score_text = f"{scores[i]:.2f}"
                    label_text = f"{label_text} {score_text}" if label_text else score_text
                
                if label_text:
                    # Calculate text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # Draw background rectangle
                    cv2.rectangle(
                        result,
                        (x1, y1 - text_height - baseline - 5),
                        (x1 + text_width + 5, y1),
                        color,
                        -1,
                    )
                    
                    # Draw text
                    cv2.putText(
                        result,
                        label_text,
                        (x1 + 2, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
        
        return result
    
    def create_side_by_side(
        self,
        original: np.ndarray,
        annotated: np.ndarray,
    ) -> np.ndarray:
        """Create side-by-side comparison.
        
        Args:
            original: Original image
            annotated: Annotated image with detections
            
        Returns:
            Side-by-side comparison image
        """
        # Ensure same height
        h1, w1 = original.shape[:2]
        h2, w2 = annotated.shape[:2]
        
        if h1 != h2:
            scale = h1 / h2
            annotated = cv2.resize(annotated, (int(w2 * scale), h1))
        
        # Concatenate horizontally
        return np.hstack([original, annotated])
    
    def create_grid(
        self,
        images: List[np.ndarray],
        labels: Optional[List[str]] = None,
        cols: int = 2,
    ) -> np.ndarray:
        """Create grid layout of multiple images.
        
        Args:
            images: List of images
            labels: Optional labels for each image
            cols: Number of columns in grid
            
        Returns:
            Grid layout image
        """
        if not images:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        n_images = len(images)
        rows = (n_images + cols - 1) // cols
        
        # Get max dimensions
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        
        # Resize all images to same size
        resized = []
        for img in images:
            if img.shape[:2] != (max_h, max_w):
                resized.append(cv2.resize(img, (max_w, max_h)))
            else:
                resized.append(img.copy())
        
        # Add labels if provided
        if labels:
            for i, (img, label) in enumerate(zip(resized, labels)):
                cv2.putText(
                    img,
                    label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
        
        # Create grid
        grid_rows = []
        for r in range(rows):
            row_images = resized[r * cols:(r + 1) * cols]
            # Pad with black images if needed
            while len(row_images) < cols:
                row_images.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
            grid_rows.append(np.hstack(row_images))
        
        return np.vstack(grid_rows)
    
    def save_visualization(
        self,
        image: np.ndarray,
        output_path: str,
        quality: int = 95,
    ) -> None:
        """Save visualization to file.
        
        Args:
            image: Image to save
            output_path: Output file path
            quality: JPEG quality (0-100)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    def print_statistics(
        self,
        n_detections: int,
        mode: str,
        scores: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
    ) -> None:
        """Print detection statistics to console.
        
        Args:
            n_detections: Number of detections
            mode: Detection mode
            scores: Confidence scores
            labels: Detection labels
        """
        print(f"\n{'='*60}")
        print(f"Detection Results ({mode.upper()} mode)")
        print(f"{'='*60}")
        print(f"Total detections: {n_detections}")
        
        if scores is not None and len(scores) > 0:
            print(f"\nConfidence scores:")
            print(f"  Mean: {scores.mean():.3f}")
            print(f"  Min:  {scores.min():.3f}")
            print(f"  Max:  {scores.max():.3f}")
        
        if labels is not None:
            print(f"\nDetected objects:")
            for i, label in enumerate(labels):
                score_text = f" (conf: {scores[i]:.3f})" if scores is not None else ""
                print(f"  {i+1}. {label}{score_text}")
        
        print(f"{'='*60}\n")
