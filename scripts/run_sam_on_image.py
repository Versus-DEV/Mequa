"""Run SAM on an image from the command line.

Usage:
    python scripts/run_sam_on_image.py --demo
    python scripts/run_sam_on_image.py --image data/images/bbbc_sample_0001.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mequa.physics.stiffness_from_geometry import (
    geometry_to_stiffness,
    stiffness_array,
)
from mequa import ingest
from mequa.vision.cells import cells_from_masks
from mequa.vision.demo_masks import make_demo_image, make_demo_masks
from mequa.vision.image_io import (
    load_image_file,
    resize_max_side,
    save_image_rgb,
)
from mequa.vision.mask_overlay import overlay_masks
from mequa.vision.sam_loader import SamLoader
from mequa.vision.sam_physics_parameterizer import extract_geometry_from_masks


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAM-to-stiffness pipeline.")

    parser.add_argument("--image", help="Path to input image.")
    parser.add_argument("--demo", action="store_true", help="Use demo masks.")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--max-masks", type=int, default=12)
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        default="outputs/sam_run",
        help="Where to save results.",
    )

    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Mequa SAM CLI")
    print("=" * 60)

    start = time.perf_counter()

    if args.demo:
        source = "demo"
        masks = make_demo_masks()
        image = make_demo_image(masks)
        sam_message = "Demo masks used. SAM not called."

    elif args.image:
        source = args.image
        image_full = load_image_file(args.image)
        image = resize_max_side(image_full, args.max_side)

        loader = SamLoader(
            max_image_side=args.max_side,
            max_masks=args.max_masks,
        )

        available, message = loader.is_available()

        if not available:
            print(f"ERROR: SAM not available: {message}")
            return 1

        print("Loading SAM...")
        result = loader.generate_masks(image)

        masks = result.masks
        sam_message = result.message

    else:
        parser.error("Use --demo or --image path/to/image.png")

    cellset = cells_from_masks(masks, min_area=args.min_area)
    features_k = cellset.features
    stiffness = cellset.stiffness
    ingest.save_cells_csv(output_dir / "cells.csv", cellset)
    print(f"{len(masks)} masks -> {cellset.n} cells "
          f"({cellset.masks_dropped_background} background, {cellset.masks_dropped_overlap} overlapping dropped)")

    duration = time.perf_counter() - start

    if masks:
        mask_array = np.stack([mask.astype(bool) for mask in masks], axis=0)
    else:
        height, width = image.shape[:2]
        mask_array = np.zeros((0, height, width), dtype=bool)

    masks_path = output_dir / "masks.npy"
    overlay_path = output_dir / "mask_overlay.png"
    geometry_path = output_dir / "geometry_features.csv"
    stiffness_path = output_dir / "stiffness_map.npy"
    summary_path = output_dir / "run_summary.json"

    np.save(masks_path, mask_array)

    overlay = overlay_masks(image, masks)

    try:
        save_image_rgb(overlay_path, overlay)
    except Exception as exc:
        print(f"WARNING: could not save overlay: {exc}")

    write_csv(geometry_path, features_k)
    np.save(stiffness_path, stiffness)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "sam_message": sam_message,
        "mask_count": len(masks),
        "region_count": len(features_k),
        "stiffness_count": int(stiffness.size),
        "stiffness_min": None if stiffness.size == 0 else float(np.min(stiffness)),
        "stiffness_max": None if stiffness.size == 0 else float(np.max(stiffness)),
        "stiffness_mean": None if stiffness.size == 0 else float(np.mean(stiffness)),
        "duration_seconds": round(duration, 4),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Source: {source}")
    print(f"SAM message: {sam_message}")
    print(f"Masks: {len(masks)}")
    print(f"Regions: {len(features_k)}")
    print(f"Duration: {duration:.3f} s")
    print("=" * 60)
    print(f"Wrote: {masks_path}")
    print(f"Wrote: {overlay_path}")
    print(f"Wrote: {geometry_path}")
    print(f"Wrote: {stiffness_path}")
    print(f"Wrote: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())