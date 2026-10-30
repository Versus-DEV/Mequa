"""Run the first Mequa core loop.

Today this supports a demo mask mode so you can validate the pipeline before
SAM is fully connected:

    python scripts/run_sam_to_solver.py --demo

Later you can pass a saved SAM mask array:

    python scripts/run_sam_to_solver.py --masks outputs/sam/masks.npy
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mequa.physics.stiffness_from_geometry import geometry_to_stiffness, stiffness_array
from mequa.vision.sam_physics_parameterizer import extract_geometry_from_masks


OUTPUT_DIR = ROOT / "outputs"
SAM_OUTPUT_DIR = OUTPUT_DIR / "sam"
SOLVER_OUTPUT_DIR = OUTPUT_DIR / "solver"


def make_demo_masks(width: int = 256, height: int = 256):
    """Create simple demo masks.

    These are not real biology. They only validate the pipeline:
    mask -> geometry -> stiffness -> output files.
    """
    y, x = np.indices((height, width)).astype(np.float32)

    round_cell = ((x - 70.0) ** 2 + (y - 70.0) ** 2) <= (32.0 ** 2)

    elongated = (
        (((x - 180.0) / 18.0) ** 2)
        + (((y - 170.0) / 48.0) ** 2)
    ) <= 1.0

    bite = ((x - 166.0) ** 2 + (y - 170.0) ** 2) <= (12.0 ** 2)

    irregular = elongated & ~bite

    return [
        round_cell.astype(bool),
        elongated.astype(bool),
        irregular.astype(bool),
    ]


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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_masks(mask_path: str):
    arr = np.load(mask_path, allow_pickle=False)

    if arr.ndim == 2:
        return [arr.astype(bool)]

    if arr.ndim == 3:
        return [arr[i].astype(bool) for i in range(arr.shape[0])]

    raise ValueError(f"Unsupported mask array shape: {arr.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Mequa SAM-to-solver parameterization pipeline."
    )

    parser.add_argument(
        "--masks",
        help="Path to a .npy file containing SAM masks.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use generated demo masks so the pipeline can run before SAM is connected.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="Ignore connected components smaller than this many pixels.",
    )

    args = parser.parse_args()

    if args.masks:
        masks = load_masks(args.masks)
        source = args.masks
    elif args.demo:
        masks = make_demo_masks()
        source = "demo"
    else:
        parser.error("Use --demo for now, or provide --masks outputs/sam/masks.npy")

    print("Mequa SAM-to-solver parameterization")
    print("=" * 50)
    print(f"Source: {source}")
    print(f"Masks: {len(masks)}")

    features = extract_geometry_from_masks(masks, min_area=args.min_area)

    print(f"Detected regions: {len(features)}")

    features_with_stiffness = geometry_to_stiffness(features)
    stiffness = stiffness_array(features_with_stiffness)

    geometry_csv = SAM_OUTPUT_DIR / "geometry_features.csv"
    stiffness_npy = SOLVER_OUTPUT_DIR / "stiffness_map.npy"
    summary_json = SOLVER_OUTPUT_DIR / "run_summary.json"

    write_csv(geometry_csv, features_with_stiffness)

    SOLVER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(stiffness_npy, stiffness)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "mask_count": len(masks),
        "region_count": len(features_with_stiffness),
        "min_area_px": args.min_area,
        "stiffness_count": int(stiffness.size),
        "stiffness_min": None if stiffness.size == 0 else float(np.min(stiffness)),
        "stiffness_max": None if stiffness.size == 0 else float(np.max(stiffness)),
        "stiffness_mean": None if stiffness.size == 0 else float(np.mean(stiffness)),
        "outputs": {
            "geometry_csv": str(geometry_csv.relative_to(ROOT)),
            "stiffness_npy": str(stiffness_npy.relative_to(ROOT)),
        },
    }

    write_json(summary_json, summary)

    print("=" * 50)
    print(f"Wrote: {geometry_csv}")
    print(f"Wrote: {stiffness_npy}")
    print(f"Wrote: {summary_json}")

    if stiffness.size:
        print(f"Stiffness min: {summary['stiffness_min']:.4f}")
        print(f"Stiffness max: {summary['stiffness_max']:.4f}")
        print(f"Stiffness mean: {summary['stiffness_mean']:.4f}")
    else:
        print("No regions passed the min-area filter.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())