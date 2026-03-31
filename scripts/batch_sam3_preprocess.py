#!/usr/bin/env python3
"""
Batch SAM3 Pre-processing Script

This script processes images from ./ingest-data/ingest-image/ and outputs
segmented images + masks to ./ingest-data/ingest-image-ready/

The segmented images can then be sent to POST /ingest with a pre-computed mask
to skip the SAM3 processing step.

Output structure:
  ingest-image-ready/
    durian_10/
      frame_0000.jpg       # Segmented image
      frame_0000.mask.png  # Binary mask (255 = object, 0 = background)
"""

import argparse
import cv2
import logging
import os
import sys
import torch
from pathlib import Path
from glob import glob
from tqdm import tqdm
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.appConfig import get_config
from src.service.preprocessorService import PreprocessorService
from src.config.logging_config import setup_logging

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def process_folder(
    input_folder: str,
    output_folder: str,
    batch_size: int = 4,
    background_color: str = "black",
    save_masks: bool = True,
):
    """Process all images in a folder using SAM3 segmentation.

    Args:
        input_folder: Path to input images
        output_folder: Path to save segmented images and masks
        batch_size: Number of images to process at once
        background_color: "black" or "white" for background
        save_masks: Whether to save mask files alongside images
    """
    # Resolve paths
    input_path = Path(input_folder).resolve()
    output_path = Path(output_folder).resolve()

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all image files (only in the immediate folder, not recursively)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(ext)))

    # Sort by filename
    image_files = sorted(image_files)

    if not image_files:
        logger.error(f"No images found in {input_path}")
        return

    logger.info(f"Found {len(image_files)} images to process")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Background: {background_color}")

    # Initialize SAM3 preprocessor
    logger.info("Initializing SAM3 preprocessor...")
    preprocessor = PreprocessorService(background_color=background_color)

    processed_count = 0
    failed_count = 0

    # Process images
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            # Get output filename
            output_file = output_path / img_file.name

            # Skip if already processed (only if both image and mask exist)
            if save_masks:
                output_mask_file = output_path / f"{img_file.stem}.mask.png"
                if output_file.exists() and output_mask_file.exists():
                    processed_count += 1
                    continue
            else:
                if output_file.exists():
                    processed_count += 1
                    continue

            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning(f"Could not read {img_file}, skipping")
                failed_count += 1
                continue

            # Process with SAM3
            segmented, mask = preprocessor.segment_with_sam3(img)

            # Save segmented image
            cv2.imwrite(str(output_file), segmented)

            # Save mask
            if save_masks:
                cv2.imwrite(str(output_mask_file), mask)

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
        description="Batch SAM3 Pre-processing for Tree Identification"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="ingest-data/ingest-image/durian_10",
        help="Input folder containing images (default: ingest-data/ingest-image/durian_10)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="ingest-data/ingest-image-ready/durian_10",
        help="Output folder for segmented images (default: ingest-data/ingest-image-ready/durian_10)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="black",
        choices=["black", "white"],
        help="Background color after segmentation (default: black)",
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Don't save mask files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        setup_logging(level="DEBUG")
        logger.setLevel(logging.DEBUG)

    # Set memory optimization before importing torch
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
        background_color=args.background,
        save_masks=not args.no_masks,
    )


if __name__ == "__main__":
    main()
