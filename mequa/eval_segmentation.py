"""Score cell detection against expert ground truth: classical thresholding vs SAM.

    python -m mequa.eval_segmentation --image data/images/IMG.tif --mask data/gt/IMG.png \
        --methods classical classical_split sam --grids 16 24 32 --out results/seg_eval.json

Ground truth is decoded into one instance per nucleus (each distinct colour or label value, split
into connected pieces). A detection is correct when its centre lies within --radius pixels of an
unmatched ground-truth centre (optimal one-to-one matching). Reported: precision, recall, F1,
count error, median centre error, and seconds. If the mask is binary, touching nuclei count as one
ground-truth object; the output says how many instances were found so you can sanity-check it.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def decode_gt_instances(arr):
    """Label image from a ground-truth mask: colour or integer values give instances."""
    from scipy import ndimage as ndi

    arr = np.asarray(arr)
    if arr.ndim == 3:
        a = arr[:, :, :3].astype(np.uint32)
        key = (a[:, :, 0] << 16) | (a[:, :, 1] << 8) | a[:, :, 2]
    else:
        key = arr.astype(np.int64)
    out = np.zeros(key.shape, dtype=np.int32)
    nxt = 0
    for v in np.unique(key):
        if v == 0:
            continue
        comp, n = ndi.label(key == v)
        for c in range(1, n + 1):
            nxt += 1
            out[comp == c] = nxt
    return out


def gt_centroids(labels, min_area=10):
    from scipy import ndimage as ndi

    ids = [i for i in range(1, int(labels.max()) + 1) if (labels == i).sum() >= min_area]
    if not ids:
        return np.zeros((0, 2))
    c = np.array(ndi.center_of_mass(labels > 0, labels, ids))
    return np.column_stack([c[:, 1], c[:, 0]])  # x, y in image coordinates (y down)


def match_points(pred, gt, radius=6.0):
    """Optimal one-to-one matching within `radius`. Inputs are (n, 2) arrays in the same frame."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    pred, gt = np.asarray(pred, dtype=float).reshape(-1, 2), np.asarray(gt, dtype=float).reshape(-1, 2)
    if len(pred) == 0 or len(gt) == 0:
        tp, errs = 0, []
    else:
        d = cdist(pred, gt)
        cost = np.where(d <= radius, d, 1e6)
        r, c = linear_sum_assignment(cost)
        ok = cost[r, c] < 1e6
        tp, errs = int(ok.sum()), d[r[ok], c[ok]].tolist()
    fp, fn = len(pred) - tp, len(gt) - tp
    precision = tp / len(pred) if len(pred) else 0.0
    recall = tp / len(gt) if len(gt) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "detected": int(len(pred)), "ground_truth": int(len(gt)), "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "count_error_pct": round(100.0 * (len(pred) - len(gt)) / max(len(gt), 1), 1),
        "median_center_error_px": round(float(np.median(errs)), 3) if errs else None,
    }


def to_image_frame(points, height):
    """Flip detections from the y-up convention used by the solver back to image rows."""
    p = np.asarray(points, dtype=float).reshape(-1, 2).copy()
    if len(p):
        p[:, 1] = height - p[:, 1]
    return p


def run_classical(image, split, min_distance):
    from mequa import ingest

    t = time.perf_counter()
    pts, _ = ingest.segment_classical(image, min_area=20, split=split, min_distance=min_distance)
    return pts, time.perf_counter() - t  # classical returns image-frame (x, col; y, row)


def run_sam(image, grid, min_area):
    from mequa.vision.cells import cells_from_masks
    from mequa.vision.sam_loader import SamLoader

    loader = SamLoader()
    loader.configure(points_per_side=grid, max_masks=1000, max_image_side=max(image.shape[:2]))
    ok, msg = loader.is_available()
    if not ok:
        raise RuntimeError(f"SAM not available: {msg}")
    t = time.perf_counter()
    res = loader.generate_masks(image)
    cells = cells_from_masks(res.masks, min_area=min_area)
    secs = time.perf_counter() - t
    extra = {"masks": len(res.masks), "background_dropped": cells.masks_dropped_background, "overlap_dropped": cells.masks_dropped_overlap}
    return to_image_frame(cells.points, image.shape[0]), secs, extra


def main():
    import cv2

    from mequa.vision.image_io import load_image_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--methods", nargs="+", default=["classical", "classical_split"], choices=["classical", "classical_split", "sam"])
    ap.add_argument("--grids", type=int, nargs="+", default=[16, 24])
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--min-distance", type=int, default=6)
    ap.add_argument("--min-area", type=int, default=40)
    ap.add_argument("--out", default="results/seg_eval.json")
    args = ap.parse_args()

    image = load_image_file(args.image)
    gt_raw = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
    if gt_raw is None:
        raise SystemExit(f"cannot read mask {args.mask}")
    labels = decode_gt_instances(gt_raw)
    gt = gt_centroids(labels)
    print(f"image {image.shape[:2]}, ground-truth instances: {len(gt)}")
    results = [{"method": "ground_truth", "instances": int(len(gt)), "image": Path(args.image).name}]
    for m in args.methods:
        if m == "sam":
            for g in args.grids:
                pts, secs, extra = run_sam(image, g, args.min_area)
                r = {"method": f"sam_grid{g}", "seconds": round(secs, 1), **extra, **match_points(pts, gt, args.radius)}
                results.append(r)
                print(r)
        else:
            pts, secs = run_classical(image, m == "classical_split", args.min_distance)
            r = {"method": m, "seconds": round(secs, 2), **match_points(pts, gt, args.radius)}
            results.append(r)
            print(r)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"radius_px": args.radius, "results": results}, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
