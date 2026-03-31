#!/usr/bin/env python3
"""
Batch Ingest Script - Uploads all preprocessed images to the API

Usage:
    python scripts/batch_ingest.py --api-url http://localhost:8001 --tree-id durian_10
"""

import argparse
import requests
import json
import os
from pathlib import Path
from tqdm import tqdm
import time

def create_items_metadata(image_ids, tree_id, latitude, longitude, hor_angle, ver_angle, pitch, captured_at):
    """Create metadata items for each image."""
    items = []
    for img_id in image_ids:
        items.append({
            "imageId": img_id,
            "treeId": tree_id,
            "latitude": latitude,
            "longitude": longitude,
            "hor_angle": hor_angle,
            "ver_angle": ver_angle,
            "pitch": pitch,
            "captured_at": captured_at
        })
    return items

def upload_batch(api_url, image_dir, items, batch_size=8, skip_sam3=True):
    """Upload a batch of images to the API."""
    image_files = []
    mask_files = []

    for item in items:
        img_id = item["imageId"]
        img_path = Path(image_dir) / f"{img_id}.jpg"

        if img_path.exists():
            image_files.append(open(img_path, "rb"))
            # No mask files needed - will use full mask
            mask_files.append(None)
        else:
            print(f"Warning: Image not found: {img_path}")
            return None

    # Prepare form data
    files = {}
    for i, img_file in enumerate(image_files):
        files[f"images[]"] = (items[i]["imageId"] + ".jpg", img_file, "image/jpeg")

    # Add masks if they exist (optional)
    # Since we don't have masks, we won't include them

    # Add items metadata
    data = {
        "items": json.dumps(items),
        "batch_size": str(batch_size),
        "skip_sam3": str(skip_sam3).lower()
    }

    try:
        response = requests.post(
            api_url,
            files=files if files else None,
            data=data,
            timeout=300
        )
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        # Close all file handles
        for f in image_files:
            f.close()

def main():
    parser = argparse.ArgumentParser(description="Batch ingest images to API")
    parser.add_argument("--api-url", default="http://localhost:8001/ingest-batch-with-mask",
                        help="API URL")
    parser.add_argument("--image-dir", default="ingest-data/ingest-image-ready/durian_10",
                        help="Directory containing processed images")
    parser.add_argument("--tree-id", default="durian_10", help="Tree ID for all images")
    parser.add_argument("--latitude", type=float, default=13.75, help="Latitude")
    parser.add_argument("--longitude", type=float, default=100.5, help="Longitude")
    parser.add_argument("--hor-angle", type=float, default=0, help="Horizontal angle")
    parser.add_argument("--ver-angle", type=float, default=0, help="Vertical angle")
    parser.add_argument("--pitch", type=float, default=0, help="Pitch angle")
    parser.add_argument("--captured-at", default="2025-01-01T00:00:00Z", help="Capture timestamp")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--start-index", type=int, default=0, help="Start index")
    parser.add_argument("--end-index", type=int, default=150, help="End index")

    args = parser.parse_args()

    # Find all images
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Directory not found: {image_dir}")
        return

    # Get all jpg files and sort them
    image_files = sorted(image_dir.glob("*.jpg"))

    # Filter by index range
    image_files = image_files[args.start_index:args.end_index]

    print(f"Found {len(image_files)} images to upload")

    # Create items for all images
    image_ids = [f.stem for f in image_files]
    items = create_items_metadata(
        image_ids,
        args.tree_id,
        args.latitude,
        args.longitude,
        args.hor_angle,
        args.ver_angle,
        args.pitch,
        args.captured_at
    )

    # Process in batches
    all_results = []
    total_succeeded = 0
    total_failed = 0

    for i in tqdm(range(0, len(items), args.batch_size), desc="Uploading batches"):
        batch_items = items[i:i + args.batch_size]

        # Prepare files for this batch
        files = {}
        for j, item in enumerate(batch_items):
            img_path = image_dir / f"{item['imageId']}.jpg"
            if img_path.exists():
                files[f"images[]"] = (item['imageId'] + ".jpg", open(img_path, "rb"), "image/jpeg")

        data = {
            "items": json.dumps(batch_items),
            "batch_size": str(args.batch_size),
            "skip_sam3": "true"
        }

        try:
            response = requests.post(
                args.api_url,
                files=files,
                data=data,
                timeout=300
            )

            result = response.json()
            print(f"\nBatch {i//args.batch_size + 1}: {result}")

            if result.get("success"):
                all_results.append(result)
                total_succeeded += result.get("succeeded", 0)
                total_failed += result.get("failed", 0)

        except Exception as e:
            print(f"Error uploading batch {i//args.batch_size + 1}: {e}")
            total_failed += len(batch_items)
        finally:
            # Close all file handles
            for f in files.values():
                if hasattr(f, 'close'):
                    f.close()

        # Small delay between batches
        time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"Upload complete!")
    print(f"  Total succeeded: {total_succeeded}")
    print(f"  Total failed: {total_failed}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
