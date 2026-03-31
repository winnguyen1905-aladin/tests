#!/usr/bin/env python3
"""
Batch background removal using SAM3 segmentation pipeline.

Reads images from  ./ingest-data/ingest-image/<tree>/<frame>.jpg
Writes RGBA PNGs to ./ingest-data/ingest-image-ready/<tree>/<frame>.png

The segmentation logic mirrors test_segmentation.py:
  1. SAM3 semantic segmentation  → binary mask (foreground / background)
  2. Morphological cleanup
  3. Apply mask as alpha channel → transparent-background RGBA PNG

Usage:
    python batch_remove_background.py
    python batch_remove_background.py --input  ingest-data/ingest-image
    python batch_remove_background.py --output ingest-data/ingest-image-ready
    python batch_remove_background.py --workers 1          # serial (default)
    python batch_remove_background.py --ext jpg png jpeg   # accepted input extensions
    python batch_remove_background.py --skip-existing      # skip already-done files
    python batch_remove_background.py --tree durian_5      # process one tree only
"""

import sys
import time
import argparse
import logging
import traceback
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

# ── make sure project root is on path ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.config.appConfig import get_config
from src.service.preprocessorService import PreprocessorService

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEP = "─" * 72


# ─────────────────────────────────────────────────────────────────────────────
# Core: remove background from a single image
# ─────────────────────────────────────────────────────────────────────────────

def remove_background(image_bgr: np.ndarray, preprocessor: PreprocessorService) -> np.ndarray:
    """Return an RGBA image with the background set to transparent.

    Uses exactly the same SAM3 → mask pipeline as test_segmentation.py:
      - segment_with_sam3()  → binary mask (0 / 255)
      - alpha = mask          → foreground pixels are opaque, background transparent
    """
    _, mask = preprocessor.segment_with_sam3(image_bgr)

    # mask should be uint8 with values 0 or 255; normalise just in case
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask          # alpha = foreground mask
    return rgba


# ─────────────────────────────────────────────────────────────────────────────
# Collect input files
# ─────────────────────────────────────────────────────────────────────────────

def collect_images(input_dir: Path, extensions: List[str], tree_filter: Optional[str]) -> List[Path]:
    """Recursively find images under input_dir."""
    exts = {e.lower().lstrip(".") for e in extensions}
    images: List[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lstrip(".").lower() not in exts:
            continue
        if tree_filter and p.parent.name != tree_filter:
            continue
        images.append(p)
    return images


# ─────────────────────────────────────────────────────────────────────────────
# Process one image
# ─────────────────────────────────────────────────────────────────────────────

def process_one(
    src: Path,
    input_dir: Path,
    output_dir: Path,
    preprocessor: PreprocessorService,
    skip_existing: bool,
) -> dict:
    """Process a single image file and return a result dict."""
    rel = src.relative_to(input_dir)
    dst = output_dir / rel.parent / (rel.stem + ".png")

    result = {
        "src": src,
        "dst": dst,
        "status": "ok",
        "elapsed": 0.0,
        "error": None,
    }

    if skip_existing and dst.exists():
        result["status"] = "skipped"
        return result

    t0 = time.time()
    try:
        image = cv2.imread(str(src))
        if image is None:
            raise ValueError(f"cv2.imread returned None for {src}")

        rgba = remove_background(image, preprocessor)

        dst.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(dst), rgba)
        if not ok:
            raise IOError(f"cv2.imwrite failed writing {dst}")

        result["elapsed"] = time.time() - t0

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["elapsed"] = time.time() - t0
        logger.debug(traceback.format_exc())

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch background removal using SAM3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default="ingest-data/ingest-image",
        help="Root folder containing tree sub-folders with images",
    )
    parser.add_argument(
        "--output", default="ingest-data/ingest-image-ready",
        help="Root folder where RGBA PNGs will be written",
    )
    parser.add_argument(
        "--ext", nargs="+", default=["jpg", "jpeg", "png"],
        help="Accepted input file extensions",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip images whose output already exists",
    )
    parser.add_argument(
        "--tree", default=None,
        help="Process only images inside this tree folder (e.g. durian_5)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── paths ────────────────────────────────────────────────────────────────
    root = Path(__file__).parent
    input_dir  = (root / args.input).resolve()
    output_dir = (root / args.output).resolve()

    if not input_dir.exists():
        print(f"❌  Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── collect images ───────────────────────────────────────────────────────
    images = collect_images(input_dir, args.ext, args.tree)
    if not images:
        print(f"⚠️   No images found in {input_dir}")
        sys.exit(0)

    # ── config & preprocessor ────────────────────────────────────────────────
    cfg    = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(SEP)
    print(f"  SAM3 Batch Background Removal")
    print(SEP)
    print(f"  Input    : {input_dir}")
    print(f"  Output   : {output_dir}")
    print(f"  Images   : {len(images)}")
    print(f"  Device   : {device}  (CUDA: {torch.cuda.is_available()})")
    print(f"  Model    : {cfg.model_path}")
    print(f"  Prompts  : {cfg.sam3_text_prompts}")
    print(f"  Skip dup : {args.skip_existing}")
    if args.tree:
        print(f"  Tree     : {args.tree}")
    print(SEP)

    # Initialise PreprocessorService once (heavy model load)
    print("  Loading SAM3 model …", flush=True)
    t_load = time.time()
    preprocessor = PreprocessorService(sam3_model_path=cfg.model_path)
    preprocessor._init_sam3()           # eager init so we fail fast
    t_load = time.time() - t_load
    print(f"  ✅ Model loaded in {t_load:.1f}s\n")

    # ── process images ───────────────────────────────────────────────────────
    n_ok = n_skip = n_err = 0
    t_total_start = time.time()

    for i, src in enumerate(images, 1):
        rel = src.relative_to(input_dir)
        prefix = f"  [{i:4d}/{len(images)}]  {rel}"

        result = process_one(src, input_dir, output_dir, preprocessor, args.skip_existing)

        if result["status"] == "skipped":
            n_skip += 1
            print(f"{prefix}  ⏭  skipped")
        elif result["status"] == "error":
            n_err += 1
            print(f"{prefix}  ❌  {result['error']}")
        else:
            n_ok += 1
            dst_rel = result["dst"].relative_to(output_dir)
            print(f"{prefix}  ✅  → {dst_rel}  ({result['elapsed']:.2f}s)")

    # ── summary ──────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_total_start
    print()
    print(SEP)
    print(f"  Done  —  {n_ok} saved  |  {n_skip} skipped  |  {n_err} errors")
    print(f"  Total time: {total_elapsed:.1f}s  "
          f"(avg {total_elapsed / max(n_ok, 1):.2f}s per image)")
    print(SEP)

    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
