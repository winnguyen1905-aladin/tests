#!/usr/bin/env python3
"""
Batch POST /ingest-transparent for all images under a directory tree.

Pipeline
--------
1. Decode each image like the API (transparent PNG / checkerboard JPEG) — same as ``main._decode_transparent_png``.
2. Run the same preprocess → DINO path as ``IngestionService.ingest`` (mask, bbox crop, ``prepare_for_dino``, ``DinoProcessor.extract``).
3. Cluster images with **union-find**: merge pairs whose **cosine similarity** is >= ``--min-cos-sim`` (L2-normalized dot product).
4. Assign GPS/orientation **from DINO cosine geometry** (per tree cluster):
   - Build pairwise distances ``sqrt(2 - 2·cos)`` from L2-normalized embeddings (chord length = Euclidean distance on the unit sphere).
   - **Classical MDS** on that matrix → 2D plane coordinates, scaled so the largest pairwise horizontal separation is ≤ ``--max-intra-tree-m``.
   - **Latitude/longitude** = tree centre + that 2D offset (metres east/north).
   - **Heading** = ``atan2(east, north)`` in degrees (same layout as GPS).
   - **Pitch** from cosine to the cluster centroid: ``1 - cos(v, c)`` → larger when the view is farther from the “typical” direction in embedding space.
   - **Roll** from a third MDS axis when ``n ≥ 3`` (otherwise 0).
   - **Different trees:** cluster centres offset along east by ``--inter-tree-spacing-m`` (default 18 m, > 15 m).
5. POST multipart to ``/ingest-transparent`` with ``tree_id``, ``image_id``, ``time_series`` JSON.

Requires: repo root on ``PYTHONPATH`` (run from repo root: ``python scripts/batch_ingest_transparent_by_dino.py ...``),
same ``.env`` / DINO credentials as the app, and a running API (or use ``--dry-run``).

After DINO finishes, embeddings are written to ``--dino-cache`` (default: ``<root>/_batch_ingest_dino_cache.npz``).
The ingest plan JSON is written **before** any HTTP POST. If the API was down (e.g. ``Connection refused``), start
the server and resume POST only::

    python scripts/batch_ingest_transparent_by_dino.py \\
        --root ingest-data/ingest-image-ready \\
        --load-dino-cache \\
        --base-url http://127.0.0.1:8001

Example (full run)::

    python scripts/batch_ingest_transparent_by_dino.py \\
        --root ingest-data/ingest-image-ready \\
        --base-url http://127.0.0.1:8001 \\
        --base-lat 10.762622 --base-lon 106.660172 \\
        --min-cos-sim 0.88
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import requests

# Repo root = parent of scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import HTTPException

from src.config.appConfig import get_config
from src.processor.dinoProcessor import DinoConfig, DinoProcessor
from src.service.preprocessorService import PreprocessorService


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

DINO_CACHE_VERSION = 1


@dataclass
class ImageRecord:
    path: Path
    rel_key: str
    vector: np.ndarray


class UnionFind:
    def __init__(self, n: int) -> None:
        self._p = list(range(n))

    def find(self, i: int) -> int:
        while self._p[i] != i:
            self._p[i] = self._p[self._p[i]]
            i = self._p[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self._p[rj] = ri


def _decode_transparent_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Match server decoding; raises ``ValueError`` on failure."""
    from main import _decode_transparent_png

    data = path.read_bytes()
    try:
        return _decode_transparent_png(data)
    except HTTPException as e:
        detail = e.detail if isinstance(e.detail, str) else str(e.detail)
        raise ValueError(f"{path}: {detail}") from e


def _dino_embedding(
    bgr: np.ndarray,
    mask: np.ndarray,
    preprocessor: PreprocessorService,
    dino: DinoProcessor,
) -> np.ndarray:
    """Same mask → crop → DINO path as ``IngestionService.ingest``."""
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    masked_image = preprocessor.apply_mask(bgr, mask_gray)
    bbox = preprocessor.get_bounding_box(mask_gray)
    cropped_image, _, _ = preprocessor.crop_to_bounding_box(masked_image, bbox, mask=mask_gray)
    dino_input = preprocessor.prepare_for_dino(cropped_image)
    result = dino.extract(dino_input)
    if result is None:
        raise ValueError("DINO extract returned None")
    vec = np.asarray(result.vector, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(vec))
    if n > 1e-12:
        vec = vec / n
    return vec


def _cluster_by_cosine(vectors: np.ndarray, min_cos_sim: float) -> np.ndarray:
    """Return cluster index per row using union-find on pairs with cosine >= min_cos_sim."""
    n = vectors.shape[0]
    uf = UnionFind(n)
    # Upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.dot(vectors[i], vectors[j])) >= min_cos_sim:
                uf.union(i, j)
    roots = [uf.find(i) for i in range(n)]
    unique = sorted(set(roots))
    root_to_id = {r: k for k, r in enumerate(unique)}
    return np.array([root_to_id[r] for r in roots], dtype=np.int32)


def _meters_to_lat_lon_delta(
    east_m: float, north_m: float, ref_lat_deg: float
) -> Tuple[float, float]:
    """Small ENU offset (east, north metres) → (d_lat, d_lon) in degrees."""
    lat_rad = np.radians(ref_lat_deg)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = max(111_320.0 * np.cos(lat_rad), 1e-6)
    return north_m / m_per_deg_lat, east_m / m_per_deg_lon


def _classical_mds_from_distance_sq(D2: np.ndarray, n_dims: int) -> np.ndarray:
    """Double-centred classical MDS; returns ``(n, n_dims)`` configuration (may have small neg. eigenvalues clipped)."""
    n = D2.shape[0]
    if n == 0:
        return np.zeros((0, n_dims))
    J = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    B = -0.5 * (J @ D2 @ J)
    B = (B + B.T) / 2.0
    w, U = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w = w[order]
    U = U[:, order]
    k = min(n_dims, n)
    vals = np.maximum(w[:k], 0.0)
    return U[:, :k] * np.sqrt(vals)


def _layout_from_cosine_matrix(
    V: np.ndarray,
    max_pairwise_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive east/north (m), heading_deg, pitch_deg, roll_deg from unit-norm rows of ``V`` (one cluster).

    Uses chord distance ``||vi - vj|| = sqrt(2 - 2 vi·vj)`` (fully determined by cosine similarity).
    """
    n = V.shape[0]
    max_pairwise_m = float(max_pairwise_m)
    if n == 0:
        e = np.zeros(0, dtype=np.float64)
        return e, e.copy(), e.copy(), e.copy(), e.copy()

    # Cluster centroid on the sphere (mean direction, normalized) — cosine to it is v·c
    c = V.sum(axis=0)
    cn = float(np.linalg.norm(c))
    if cn > 1e-12:
        c = (c / cn).astype(np.float64)
    else:
        c = V[0].astype(np.float64)

    cos_to_centroid = (V @ c).astype(np.float64)

    if n == 1:
        east = np.array([0.0])
        north = np.array([0.0])
        heading = np.array([0.0])
        pitch = np.array([5.0 + 30.0 * (1.0 - float(cos_to_centroid[0]))])
        pitch = np.clip(pitch, 0.0, 40.0)
        roll = np.array([0.0])
        return east, north, heading, pitch, roll

    # Pairwise squared Euclidean distances in R^d (same as chord^2 on sphere for unit vectors)
    G = (V @ V.T).astype(np.float64)
    D2 = np.clip(2.0 - 2.0 * G, 0.0, None)

    # At most ``n - 1`` positive MDS dimensions; ask for up to 3 (ENU + roll hint).
    n_mds = max(1, min(3, n - 1))
    X = _classical_mds_from_distance_sq(D2, n_dims=n_mds)
    if X.shape[1] < 3:
        X = np.pad(X, ((0, 0), (0, 3 - X.shape[1])), mode="constant")
    east = X[:, 0].copy()
    north = X[:, 1].copy()
    z3 = X[:, 2].copy()

    # Isotropic scale on (east, north) so max pairwise horizontal distance ≤ max_pairwise_m
    if n >= 2:
        diff = np.sqrt((east[:, None] - east[None, :]) ** 2 + (north[:, None] - north[None, :]) ** 2)
        max_d = float(diff.max())
        if max_d > 1e-9:
            scale = (max_pairwise_m * 0.98) / max_d
        else:
            scale = 0.0
        east *= scale
        north *= scale

    heading = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0

    # Pitch: embedding dissimilarity to centroid (pure cosine)
    pitch = 5.0 + 30.0 * (1.0 - cos_to_centroid)
    pitch = np.clip(pitch, 0.0, 40.0)

    # Roll: third MDS direction, normalized per cluster (only when variation exists)
    if n >= 3 and float(np.std(z3)) > 1e-9:
        z_norm = (z3 - z3.mean()) / (z3.std() + 1e-12)
        roll = np.clip(12.0 * z_norm, -25.0, 25.0)
    else:
        roll = np.zeros(n)

    return east, north, heading, pitch, roll


def _collect_images(root: Path, include_dirs: Optional[Set[str]] = None) -> List[Path]:
    paths: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            if include_dirs:
                rel = p.relative_to(root)
                if not rel.parts or rel.parts[0] not in include_dirs:
                    continue
            paths.append(p)
    return paths


def _save_dino_cache(cache_path: Path, root: Path, records: List[ImageRecord], min_cos_sim: float) -> None:
    root = root.resolve()
    vectors = np.stack([r.vector for r in records], axis=0).astype(np.float32)
    rel_keys = np.array([r.rel_key for r in records], dtype=object)
    np.savez_compressed(
        cache_path,
        version=np.int32(DINO_CACHE_VERSION),
        root=str(root),
        min_cos_sim=np.float64(min_cos_sim),
        vectors=vectors,
        rel_keys=rel_keys,
    )
    print(f"Wrote DINO cache ({len(records)} vectors): {cache_path}")


def _load_dino_cache(cache_path: Path, root: Path) -> List[ImageRecord]:
    if not cache_path.is_file():
        raise FileNotFoundError(f"DINO cache not found: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    ver = int(np.asarray(data["version"]).item())
    if ver != DINO_CACHE_VERSION:
        raise ValueError(f"Unsupported DINO cache version {ver} (expected {DINO_CACHE_VERSION})")
    cached_root = Path(str(np.asarray(data["root"]).item())).resolve()
    if cached_root != root.resolve():
        print(
            f"WARNING: cache root {cached_root} != --root {root.resolve()}; using paths under --root",
            file=sys.stderr,
        )
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    rel_keys = data["rel_keys"]
    records: List[ImageRecord] = []
    for i in range(vectors.shape[0]):
        rk = str(rel_keys[i])
        p = (root / rk).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Image from cache missing: {p} (rel={rk})")
        v = vectors[i].reshape(-1)
        n = float(np.linalg.norm(v))
        if n > 1e-12:
            v = (v / n).astype(np.float32)
        records.append(ImageRecord(path=p, rel_key=rk, vector=v))
    print(f"Loaded DINO cache: {len(records)} vectors from {cache_path}")
    return records


def _image_id_from_rel(rel: Path) -> str:
    s = str(rel).replace(os.sep, "_")
    for ch in (".", " ", "-"):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "image"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT / "ingest-data" / "ingest-image-ready",
        help="Root directory to scan recursively for images",
    )
    parser.add_argument("--base-url", default=os.environ.get("SAM3_API_URL", "http://127.0.0.1:8001"), help="API base URL")
    parser.add_argument("--base-lat", type=float, default=10.762622, help="Reference latitude for synthetic GPS")
    parser.add_argument("--base-lon", type=float, default=106.660172, help="Reference longitude for synthetic GPS")
    parser.add_argument(
        "--include-dirs",
        nargs="*",
        default=None,
        help="Optional top-level folder names under --root to include (e.g. durian_5 durian_6)",
    )
    parser.add_argument(
        "--min-cos-sim",
        type=float,
        default=0.88,
        help="Merge into same tree if cosine similarity >= this (tune per dataset)",
    )
    parser.add_argument(
        "--max-intra-tree-m",
        type=float,
        default=9.0,
        help="Max horizontal distance between any two images of the same tree (default 9 m, < 10 m)",
    )
    parser.add_argument(
        "--inter-tree-spacing-m",
        type=float,
        default=18.0,
        help="Distance between successive tree centres along east axis (> 15 m)",
    )
    parser.add_argument("--base-timestamp-ms", type=int, default=None, help="Epoch ms for first image (default: now)")
    parser.add_argument("--tree-prefix", default="dino-tree", help="tree_id = {prefix}-{cluster:04d}")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N images (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Compute clusters and GPS only; no HTTP")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between POSTs")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1

    include_dirs = set(args.include_dirs) if args.include_dirs else None
    paths = _collect_images(root, include_dirs=include_dirs)
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        print(f"No images under {root}", file=sys.stderr)
        return 1

    print(f"Found {len(paths)} images under {root}")
    cfg = get_config()
    print(f"DINO model type: {cfg.dino_model_type}, dim target (storage): {cfg.postgres_vector_dim}")

    preprocessor = PreprocessorService()
    dino = DinoProcessor(DinoConfig())

    records: List[ImageRecord] = []
    try:
        for idx, path in enumerate(paths):
            rel = path.relative_to(root)
            print(f"[{idx + 1}/{len(paths)}] DINO: {rel}")
            try:
                bgr, mask = _decode_transparent_file(path)
                if not np.any(mask):
                    print(f"  skip: empty mask", file=sys.stderr)
                    continue
                vec = _dino_embedding(bgr, mask, preprocessor, dino)
            except Exception as e:
                print(f"  FAILED: {e}", file=sys.stderr)
                continue
            records.append(ImageRecord(path=path, rel_key=str(rel.as_posix()), vector=vec))
    finally:
        dino.close()

    if not records:
        print("No usable images after decoding/DINO.", file=sys.stderr)
        return 1

    mat = np.stack([r.vector for r in records], axis=0)
    labels = _cluster_by_cosine(mat, float(args.min_cos_sim))
    n_clusters = int(labels.max()) + 1
    print(f"Clusters (min_cos_sim={args.min_cos_sim}): {n_clusters} trees from {len(records)} images")

    base_ts = args.base_timestamp_ms if args.base_timestamp_ms is not None else int(time.time() * 1000)
    headers = {}
    if os.environ.get("API_KEY"):
        headers["X-API-Key"] = os.environ["API_KEY"]

    plan_path = root / "_ingest_transparent_plan.json"
    plan_rows = []

    # Per-cluster centre: spaced along +east from base
    cluster_centre_en: dict[int, Tuple[float, float]] = {}
    for c in range(n_clusters):
        east_c = c * float(args.inter_tree_spacing_m)
        cluster_centre_en[c] = (east_c, 0.0)

    by_cluster: dict[int, List[int]] = {c: [] for c in range(n_clusters)}
    for i, lab in enumerate(labels.tolist()):
        by_cluster[lab].append(i)

    for c in range(n_clusters):
        idxs = by_cluster[c]
        east0, north0 = cluster_centre_en[c]
        tree_id = f"{args.tree_prefix}-{c:04d}"

        Vc = np.stack([records[i].vector for i in idxs], axis=0)
        de_arr, dn_arr, heading_arr, pitch_arr, roll_arr = _layout_from_cosine_matrix(
            Vc, float(args.max_intra_tree_m)
        )

        for j, rec_i in enumerate(idxs):
            rec = records[rec_i]
            east_m = east0 + float(de_arr[j])
            north_m = north0 + float(dn_arr[j])
            d_lat, d_lon = _meters_to_lat_lon_delta(east_m, north_m, args.base_lat)
            lat = args.base_lat + d_lat
            lon = args.base_lon + d_lon

            heading = float(heading_arr[j])
            pitch = float(pitch_arr[j])
            roll = float(roll_arr[j])

            ts_ms = base_ts + rec_i * 1500
            image_id = _image_id_from_rel(Path(rec.rel_key))

            ts_payload = {
                "latitude": lat,
                "longitude": lon,
                "timestamp": ts_ms,
                "heading": heading,
                "pitch": pitch,
                "roll": roll,
            }

            row = {
                "path": str(rec.path),
                "rel": rec.rel_key,
                "cluster": c,
                "tree_id": tree_id,
                "image_id": image_id,
                "time_series": ts_payload,
            }
            plan_rows.append(row)

            if args.dry_run:
                print(f"DRY {tree_id} {image_id} lat={lat:.7f} lon={lon:.7f} h={heading:.1f}")
                continue

            url = f"{args.base_url.rstrip('/')}/ingest-transparent"
            data = {"tree_id": tree_id, "image_id": image_id, "time_series": json.dumps(ts_payload)}
            with open(rec.path, "rb") as f:
                files = {"image": (rec.path.name, f, "application/octet-stream")}
                resp = requests.post(url, data=data, files=files, headers=headers, timeout=600)

            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text[:500]}

            top_success = body.get("success")
            nested_success = body.get("data", {}).get("success") if isinstance(body.get("data"), dict) else None
            ok = resp.status_code == 200 and (top_success is True or nested_success is True)
            status = "OK" if ok else f"FAIL {resp.status_code}"
            print(f"{status} {tree_id} {image_id} — {body.get('message', body)}")
            if not ok:
                print(f"  detail: {body}", file=sys.stderr)

            if args.sleep > 0:
                time.sleep(args.sleep)

    plan_path.write_text(json.dumps(plan_rows, indent=2), encoding="utf-8")
    print(f"Wrote plan: {plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
