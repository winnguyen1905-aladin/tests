#!/usr/bin/env python3
"""
Batch SAM3 Pre-processing Script - Chessboard/Transparent Background

This script processes images from ./ingest-data/ingest-image/ and outputs
images with transparent/chessboard background to ./ingest-data/ingest-image-ready/

The output images can be used for ingestion or visualization.
"""

import argparse
import cv2
import logging
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.service.preprocessorService import PreprocessorService
from src.config.logging_config import setup_logging

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def create_chessboard_background(width: int, height: int, square_size: int = 20) -> np.ndarray:
    """Create a chessboard background pattern.

    Args:
        width: Image width
        height: Image height
        square_size: Size of each chessboard square

    Returns:
        Chessboard image as numpy array
    """
    import numpy as np

    # Create checkerboard pattern
    rows = (height + square_size - 1) // square_size
    cols = (width + square_size - 1) // square_size

    # Create checkerboard (gray and white)
    checker = np.indices((rows, cols)).sum(axis=0) % 2
    checker = checker.astype(np.float32) * 200 + 55  # Range 55-255 (gray/white)

    # Resize to image size
    background = cv2.resize(checker, (width, height), interpolation=cv2.INTER_NEAREST)

    return background.astype(np.uint8)


def process_folder(
    input_folder: str,
    output_folder: str,
    batch_size: int = 4,
    background_type: str = "chessboard",
):
    """Process all images in a folder using SAM3 segmentation.

    Args:
        input_folder: Path to input images
        output_folder: Path to save processed images
        batch_size: Number of images to process at once
        background_type: "chessboard", "transparent", or "black"
    """
    # Resolve paths
    input_path = Path(input_folder).resolve()
    output_path = Path(output_folder).resolve()

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(ext)))

    image_files = sorted(set(image_files))

    if not image_files:
        logger.error(f"No images found in {input_path}")
        return

    logger.info(f"Found {len(image_files)} images to process")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Background type: {background_type}")

    # Initialize SAM3 preprocessor
    logger.info("Initializing SAM3 preprocessor...")
    preprocessor = PreprocessorService(background_color="black")

    processed_count = 0
    failed_count = 0

    # Process images
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            # Check if already processed
            output_file = output_path / img_file.name
            if output_file.exists():
                processed_count += 1
                continue

            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning(f"Could not read {img_file}, skipping")
                failed_count += 1
                continue

            # Process with SAM3 to get mask
            segmented, mask = preprocessor.segment_with_sam3(img)

            # Create output based on background type
            if background_type == "chessboard":
                # Create chessboard background
                h, w = img.shape[:2]
                background = create_chessboard_background(w, h)
                background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

                # Apply mask: use segmented where mask is white, else chessboard
                # mask is 255 (white) for object, 0 (black) for background
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
                output = (segmented * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)

            elif background_type == "transparent":
                # For PNG with transparency, we need RGBA
                h, w = img.shape[:2]
                # Create RGBA image
                output = cv2.cvtColor(segmented, cv2.COLOR_BGR2BGRA)
                # Set alpha channel: 255 where mask is white, 0 where mask is black
                output[:, :, 3] = mask

            else:  # black
                output = segmented

            # Save output image
            if background_type == "transparent":
                cv2.imwrite(str(output_file), output, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                cv2.imwrite(str(output_file), output)

            processed_count += 1

            # Clear GPU memory periodically
            if processed_count % batch_size == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            logger.error(f"Error processing {img_file}: {e}")
            failed_count += 1
            continue

    logger.info(f"Processing complete!")
    logger.info(f"  Processed: {processed_count}")
    logger.info(f"  Failed: {failed_count}")
    logger.info(f"  Output: {output_path}")

    # Cleanup GPU memory
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Batch SAM3 Pre-processing - Chessboard Background"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="ingest-data/ingest-image/durian_10",
        help="Input folder containing images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="ingest-data/ingest-image-ready/durian_10",
        help="Output folder for processed images",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=4,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="chessboard",
        choices=["chessboard", "transparent", "black"],
        help="Background type (default: chessboard)",
    )

    args = parser.parse_args()

    # Set memory optimization
    is_rocm = False
    try:
        is_rocm = getattr(torch.version, 'hip', None) is not None
    except Exception:
        pass

    if is_rocm:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    else:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'

    process_folder(
        input_folder=args.input,
        output_folder=args.output,
        batch_size=args.batch_size,
        background_type=args.background,
    )


if __name__ == "__main__":
    main()
