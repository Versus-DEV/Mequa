"""Real 3D nuclei -> strain analysis (BBBC033: mouse trophoblast stem cell nuclei, CC0).

Pipeline: load the 3D fluorescence stack, detect nuclei in 3D (smooth, Otsu, distance transform,
local maxima), score the detections against the expert 3D label volume, then run the same 3D spring
network analysis used everywhere else: connectivity check, residual force, SPHI, and a load sweep
for the failure threshold.

Coordinates are voxel units (x, y, z) with z multiplied by --z-scale. The dataset page gives the
z spacing (0.5 um) but the x-y pixel size must be read from the file metadata; set --z-scale to
z_spacing / xy_pixel_size if you want isotropic units. The default of 1.0 treats voxels as cubes.

Scoring is reported two ways so a low score can be diagnosed instead of guessed at:
  * centroid match: a detection is correct if it lies within --radius voxels of an expert centroid
    (--radius auto uses half the median expert nucleus radius, because a fixed 4 voxels is far
    smaller than a real nucleus and scores a correct detection as a miss);
  * inside-instance match: a detection is correct if it falls inside an expert nucleus, each
    nucleus counted once, extra detections in the same nucleus counted as splits.
The result also carries the volume and label shapes, the number of expert nuclei and the coordinate
ranges of both point sets, so an axis-order or cropping mismatch is visible in the output.

  python -m mequa.real3d --volume BBBC033_v1_dataset.zip --gt BBBC033DatasetGroundTruth.tif --out results/real3d.json
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ----------------------------------------------------------------------------- loading

def _read_tiff_from(path: str) -> np.ndarray:
    import tifffile

    p = Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            tifs = [i for i in z.infolist()
                    if i.filename.lower().endswith((".tif", ".tiff")) and not i.filename.startswith("__MACOSX")]
            if not tifs:
                raise ValueError(f"no .tif inside {p.name}")
            big = max(tifs, key=lambda i: i.file_size)
            out = p.with_suffix("").parent / Path(big.filename).name
            if not out.exists():
                out.write_bytes(z.read(big))
            p = out
    return np.squeeze(tifffile.imread(str(p)))


def _flatten_channels(arr: np.ndarray, labels: bool) -> np.ndarray:
    """Turn a 4D (…, C) or (C, …) stack into 3D.

    Intensity images: the brightest channel per voxel. Label images stored as RGB: every distinct
    colour becomes one instance id, because taking a maximum over colour channels would merge
    different nuclei into the same number.
    """
    if arr.ndim != 4:
        return arr
    if arr.shape[-1] in (3, 4):  # RGB / RGBA stored channels-last, the TIFF default
        chan = arr[..., :3]
    elif arr.shape[0] in (3, 4):  # channels-first
        chan = np.moveaxis(arr[:3], 0, -1)
    else:
        return arr
    if not labels:
        return chan.max(axis=-1)
    flat = chan.reshape(-1, chan.shape[-1]).astype(np.int64)
    key = np.zeros(len(flat), dtype=np.int64)
    for j in range(flat.shape[1]):
        key = key * 65536 + flat[:, j]
    ids = np.zeros(len(flat), dtype=np.int64)
    nonzero = key != 0  # colour (0, 0, 0) is background
    _, inv = np.unique(key[nonzero], return_inverse=True)
    ids[nonzero] = inv + 1
    return ids.reshape(chan.shape[:-1]).astype(np.int32)


def load_volume(path: str, labels: bool = False) -> np.ndarray:
    """Load a 3D stack (z, y, x) from a .tif or from the biggest .tif inside a .zip."""
    vol = _flatten_channels(_read_tiff_from(path), labels)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D stack (z, y, x), got shape {vol.shape}")
    return vol


def instances_from_labels(labels: np.ndarray) -> Tuple[np.ndarray, str]:
    """Return an instance-id volume. A binary mask (only 0 and one other value) is split into
    connected components, since it carries no instance ids of its own."""
    vals = np.unique(labels)
    vals = vals[vals > 0]
    if len(vals) == 1:
        lab, _ = ndi.label(labels > 0)
        return lab.astype(np.int32), "binary mask split into connected components"
    return labels, "instance ids as stored"


def match_label_axes(labels: np.ndarray, vol_shape: Tuple[int, ...]) -> Tuple[np.ndarray, Optional[str]]:
    """Make the label volume the same shape as the image. If it is the same shape with the axes
    in another order, permute it and say so; any other mismatch is an error, not a guess."""
    if labels.shape == tuple(vol_shape):
        return labels, None
    if sorted(labels.shape) == sorted(vol_shape):
        for perm in ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            if tuple(labels.shape[a] for a in perm) == tuple(vol_shape):
                return np.transpose(labels, perm), f"label axes permuted {perm} to match the image"
    raise ValueError(
        f"the expert label volume has shape {labels.shape} but the image has shape {tuple(vol_shape)}; "
        "they must describe the same voxels (z, y, x). Check that --gt is the ground-truth file for this exact volume."
    )


# ----------------------------------------------------------------------------- detection

def centroids_from_labels(labels: np.ndarray, z_scale: float = 1.0) -> np.ndarray:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return np.empty((0, 3))
    com = np.array(ndi.center_of_mass(np.ones_like(labels, dtype=np.uint8), labels, ids))  # (z, y, x)
    return np.column_stack([com[:, 2], com[:, 1], com[:, 0] * z_scale])


def nucleus_radii(labels: np.ndarray) -> np.ndarray:
    """Equivalent-sphere radius (voxels) of every expert nucleus."""
    counts = np.bincount(labels.ravel().astype(np.int64))[1:]
    counts = counts[counts > 0]
    return (3.0 * counts / (4.0 * np.pi)) ** (1.0 / 3.0)


def detect_nuclei_3d(vol: np.ndarray, sigma: float = 2.0, min_distance: Union[int, None, str] = 4,
                     z_scale: float = 1.0, h_frac: float = 0.25, progress=None,
                     threshold_scale: float = 1.0, refine: bool = False) -> np.ndarray:
    """Smooth, Otsu, distance transform, then one marker per nucleus.

    threshold_scale < 1 lowers the Otsu cut-off (catches dim nuclei, at the cost of fatter masks).
    refine=True moves each detection to the intensity-weighted centre of its nucleus.

    min_distance="auto" (or None) picks markers with h-maxima: two peaks of the distance map belong to
    different nuclei only if the valley between them is deeper than h = h_frac * typical radius. A
    nucleus with a lumpy interior gives one broad dome, so it gives one marker; two touching nuclei
    are separated by a neck, so they give two. The old fixed-spacing rule split every large nucleus
    (the v6 BBBC033 run: 282 detections for 15 nuclei). An integer min_distance keeps the old
    peak_local_max behaviour for people who want to tune it by hand.
    """
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    from skimage.morphology import h_maxima

    def tick(msg):
        if progress:
            progress(msg)

    tick("smoothing")
    v = ndi.gaussian_filter(vol.astype(np.float32), sigma)
    tick("thresholding")
    thr = float(threshold_otsu(v)) * float(threshold_scale)
    mask = v > thr
    cl, ncl = ndi.label(mask)
    if ncl:
        sizes = ndi.sum(mask, cl, np.arange(1, ncl + 1))
        mask = np.isin(cl, np.flatnonzero(sizes >= 30) + 1)
    # Pad with background first. Without it the distance transform treats the stack edge as open
    # space, so a nucleus cut by the first or last slice peaks AT that slice, not at its centre.
    mask = np.pad(mask, 1, constant_values=False)
    tick("distance transform")
    dist = ndi.distance_transform_edt(mask, sampling=(max(z_scale, 1e-6), 1.0, 1.0)).astype(np.float32)
    auto = min_distance is None or min_distance == "auto"
    if auto:
        if not mask.any():
            return np.empty((0, 3))
        r_typ = float(np.percentile(dist[mask], 95))
        sm = ndi.gaussian_filter(dist, max(1.0, 0.1 * r_typ))
        h = max(0.5, h_frac * r_typ)
        tick("finding one marker per nucleus")
        hm = h_maxima(sm, h)
        lab, n = ndi.label(hm, structure=np.ones((3, 3, 3)))
        if n == 0:
            return np.empty((0, 3))
        idx = np.arange(1, n + 1)
        peaks = np.array(ndi.center_of_mass(hm, lab, idx))
        heights = np.asarray(ndi.maximum(sm, lab, idx))
        ref_h = float(np.median(heights[heights >= 0.35 * r_typ])) if (heights >= 0.35 * r_typ).any() else r_typ
        keep = heights >= 0.35 * min(r_typ, ref_h) * 1.0   # drop debris; judged against real peak heights, not the 95th percentile
        peaks = np.round(peaks[keep]).astype(int)
    else:
        peaks = peak_local_max(ndi.gaussian_filter(dist, 1.0), min_distance=int(min_distance), labels=mask.astype(np.int32), exclude_border=False)
    if len(peaks) == 0:
        return np.empty((0, 3))
    peaks = peaks - 1  # undo the padding
    peaks = np.clip(peaks, 0, np.array(vol.shape) - 1)
    if refine:
        tick("refining centres")
        peaks = _refine_centres(v, peaks, thr, z_scale)
    return np.column_stack([peaks[:, 2], peaks[:, 1], peaks[:, 0] * z_scale]).astype(np.float64)


def _refine_centres(v: np.ndarray, peaks: np.ndarray, thr: float, z_scale: float, half: int = 6) -> np.ndarray:
    """Intensity-weighted centre inside a window around each detection (only voxels above threshold)."""
    out = []
    for z, y, x in np.asarray(peaks, dtype=float):
        z0, y0, x0 = int(round(z)), int(round(y)), int(round(x))
        hz = max(1, int(round(half / max(z_scale, 1e-6))))
        sl = (slice(max(z0 - hz, 0), z0 + hz + 1), slice(max(y0 - half, 0), y0 + half + 1), slice(max(x0 - half, 0), x0 + half + 1))
        w = np.clip(v[sl] - thr, 0, None)
        if w.sum() <= 0:
            out.append((z, y, x))
            continue
        zz, yy, xx = np.meshgrid(*[np.arange(a.start, a.stop) for a in sl], indexing="ij")
        # keep only the blob connected to the detection's own window: weight falls off with distance
        d2 = ((zz - z0) * z_scale) ** 2 + (yy - y0) ** 2 + (xx - x0) ** 2
        w = w * (d2 <= (half + 1) ** 2)
        t = w.sum()
        out.append(((w * zz).sum() / t, (w * yy).sum() / t, (w * xx).sum() / t))
    return np.array(out)


def match_offsets(pred: np.ndarray, gt: np.ndarray, radius: float) -> Dict[str, Any]:
    """Median (dx, dy, dz) of detection minus expert centre for matched pairs, to expose a systematic shift."""
    if len(pred) == 0 or len(gt) == 0:
        return {"pairs": 0}
    d = cdist(pred, gt)
    r, c = linear_sum_assignment(np.where(d <= radius * 2, d, 1e6))
    ok = d[r, c] <= radius * 2
    if not ok.any():
        return {"pairs": 0}
    diff = pred[r[ok]] - gt[c[ok]]
    return {"pairs": int(ok.sum()), "median_dx": round(float(np.median(diff[:, 0])), 2),
            "median_dy": round(float(np.median(diff[:, 1])), 2), "median_dz": round(float(np.median(diff[:, 2])), 2)}


def z_profile(vol: np.ndarray, labels: np.ndarray, pts: np.ndarray, z_scale: float) -> list:
    """Per z-slice: mean intensity, expert-labelled voxels, detections. Shows if expert and image disagree on z."""
    rows = []
    zi = np.clip(np.round(pts[:, 2] / z_scale).astype(int), 0, vol.shape[0] - 1) if len(pts) else np.array([], int)
    for z in range(vol.shape[0]):
        rows.append({"z": z, "mean_intensity": round(float(vol[z].mean()), 2), "expert_voxels": int((labels[z] > 0).sum()),
                     "detections": int((zi == z).sum())})
    return rows


def missed_report(vol: np.ndarray, labels: np.ndarray, pts: np.ndarray, z_scale: float, radius: float) -> list:
    """For each expert nucleus with no detection nearby: size and brightness relative to the image."""
    gt = centroids_from_labels(labels, z_scale)
    ids = [i for i in np.unique(labels) if i > 0]
    if len(gt) != len(ids):
        return []
    d = cdist(gt, pts) if len(pts) else np.full((len(gt), 1), 1e9)
    bg = float(np.median(vol))
    out = []
    for k, i in enumerate(ids):
        if d[k].min() > radius * 2:
            m = labels == i
            out.append({"label": int(i), "centre_xyz": [round(float(v), 1) for v in gt[k]], "voxels": int(m.sum()),
                        "mean_intensity": round(float(vol[m].mean()), 2), "background_median": round(bg, 2),
                        "nearest_detection_vox": round(float(d[k].min()), 1) if len(pts) else None})
    return out


# ----------------------------------------------------------------------------- scoring

def _prf(tp: int, n_pred: int, n_gt: int) -> Dict[str, Any]:
    fp, fn = n_pred - tp, n_gt - tp
    p = tp / n_pred if n_pred else 0.0
    r_ = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
    return {"detected": int(n_pred), "ground_truth": int(n_gt), "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(p, 4), "recall": round(r_, 4), "f1": round(f1, 4),
            "count_error_pct": round(100.0 * (n_pred - n_gt) / max(n_gt, 1), 1)}


def match(pred: np.ndarray, gt: np.ndarray, radius: float) -> Dict[str, Any]:
    if len(pred) == 0 or len(gt) == 0:
        tp, errs = 0, []
    else:
        d = cdist(pred, gt)
        cost = np.where(d <= radius, d, 1e6)
        r, c = linear_sum_assignment(cost)
        ok = cost[r, c] < 1e6
        tp, errs = int(ok.sum()), d[r[ok], c[ok]].tolist()
    out = _prf(tp, len(pred), len(gt))
    out["median_center_error_vox"] = round(float(np.median(errs)), 3) if errs else None
    return out


def match_inside(pred: np.ndarray, labels: np.ndarray, z_scale: float = 1.0) -> Dict[str, Any]:
    """A detection is correct when it lies inside an expert nucleus; each nucleus is credited once.
    Further detections inside an already-credited nucleus are reported as splits."""
    ids = np.unique(labels)
    n_gt = int((ids > 0).sum())
    hit, splits, outside = set(), 0, 0
    for x, y, z in pred:
        zi, yi, xi = int(round(z / z_scale)), int(round(y)), int(round(x))
        if not (0 <= zi < labels.shape[0] and 0 <= yi < labels.shape[1] and 0 <= xi < labels.shape[2]):
            outside += 1
            continue
        lab = int(labels[zi, yi, xi])
        if lab <= 0:
            outside += 1
        elif lab in hit:
            splits += 1
        else:
            hit.add(lab)
    out = _prf(len(hit), len(pred), n_gt)
    out["splits"] = splits
    out["detections_outside_nuclei"] = outside
    return out


def _span(a: np.ndarray) -> Dict[str, Any]:
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "min_xyz": [round(float(v), 1) for v in a.min(axis=0)],
            "max_xyz": [round(float(v), 1) for v in a.max(axis=0)]}


# ----------------------------------------------------------------------------- analysis

def analyze_points(points: np.ndarray, knn: int = 10, loads: int = 12, strain_limit: float = 0.05, k: int = 3) -> Dict[str, Any]:
    from mequa import parallel
    from mequa.session import Session

    if len(points) <= knn + 1:
        return {"cells": int(len(points)), "dim": int(points.shape[1]), "knn": knn,
                "note": f"only {len(points)} cells detected, need more than knn+1={knn + 1} for the strain analysis"}
    sess = Session(points, knn=knn, k=k)
    r = sess.run(0.05)
    out = {key: r.get(key) for key in ("n_components", "load_path", "residual", "sphi", "mean_strain", "max_strain") if key in r}
    out.update({"cells": int(len(points)), "dim": int(points.shape[1]), "knn": knn})
    if r.get("load_path"):
        sweep = parallel.sweep_loads(points, np.linspace(0.01, 0.12, loads).tolist(), k=k, knn=knn, comm=None)
        out["strain_limit"] = strain_limit
        out["threshold_load_p95"] = parallel.find_threshold(sweep["rows"], strain_limit)
        out["sweep_seconds"] = sweep["wall_seconds"]
    else:
        out["note"] = "pulled edge not connected to held edge; increase knn before trusting strain values"
    return out


def run(volume: str, gt: Optional[str], radius: Optional[float], z_scale: float, knn: int, sigma: float,
        min_distance: Union[int, None, str], save_points: Optional[str] = None,
        threshold_scale: float = 1.0, refine: bool = False) -> Dict[str, Any]:
    """radius=None means auto (half the median expert nucleus radius). min_distance=None/"auto"
    sizes the peak spacing from the detected nuclei."""
    t0 = time.perf_counter()
    vol = load_volume(volume)
    pts = detect_nuclei_3d(vol, sigma=sigma, min_distance=min_distance, z_scale=z_scale,
                           threshold_scale=threshold_scale, refine=refine)
    if save_points:
        Path(save_points).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(save_points, pts, delimiter=",", header="x,y,z", comments="")
    result: Dict[str, Any] = {"volume_shape_zyx": list(vol.shape), "detection_seconds": round(time.perf_counter() - t0, 1),
                              "z_scale": z_scale, "units": "voxels (x, y, z*z_scale)",
                              "threshold_scale": threshold_scale, "refine": refine,
                              "min_distance": "auto" if min_distance in (None, "auto") else int(min_distance)}
    if gt:
        raw = load_volume(gt, labels=True)
        raw, axes_note = match_label_axes(raw, vol.shape)
        labels, how = instances_from_labels(raw)
        gt_pts = centroids_from_labels(labels, z_scale)
        radii = nucleus_radii(labels)
        used = float(radius) if radius is not None else (max(2.0, 0.5 * float(np.median(radii))) if len(radii) else 4.0)
        result["expert_labels"] = {"shape_zyx": list(labels.shape), "nuclei": int(len(gt_pts)), "interpreted_as": how,
                                   "median_radius_vox": round(float(np.median(radii)), 2) if len(radii) else None,
                                   "axes_note": axes_note, "detected_span": _span(pts), "expert_span": _span(gt_pts)}
        result["detection_vs_expert_labels"] = {"radius_vox": round(used, 2), "radius_mode": "fixed" if radius is not None else "auto",
                                                **match(pts, gt_pts, used)}
        result["detection_inside_expert_nuclei"] = match_inside(pts, labels, z_scale)
        result["diagnostics"] = {"matched_offsets_vox": match_offsets(pts, gt_pts, used),
                                 "missed_nuclei": missed_report(vol, labels, pts, z_scale, used),
                                 "z_profile": z_profile(vol, labels, pts, z_scale)}
    result["analysis_on_detected_cells"] = analyze_points(pts, knn=knn)
    return result


def _int_or_auto(s: str):
    return "auto" if s == "auto" else int(s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True, help="3D .tif or the dataset .zip")
    ap.add_argument("--gt", default=None, help="expert 3D label volume (.tif)")
    ap.add_argument("--radius", default="auto", help="match radius in voxels, or auto (half the median expert nucleus radius)")
    ap.add_argument("--z-scale", type=float, default=1.0)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--min-distance", type=_int_or_auto, default="auto", help="peak spacing in voxels, or auto")
    ap.add_argument("--threshold-scale", type=float, default=1.0, help="multiply the Otsu cut-off (e.g. 0.7 to catch dim nuclei)")
    ap.add_argument("--refine", action="store_true", help="move each detection to the intensity-weighted centre")
    ap.add_argument("--save-points", default=None, help="write detected cell coordinates as CSV (x,y,z), e.g. for mequa.eval_agent --points")
    ap.add_argument("--out", default="results/real3d.json")
    args = ap.parse_args(argv)
    radius = None if args.radius == "auto" else float(args.radius)
    res = run(args.volume, args.gt, radius, args.z_scale, args.knn, args.sigma, args.min_distance, args.save_points,
              args.threshold_scale, args.refine)
    print(json.dumps(res, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
