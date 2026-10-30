import argparse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_PATH = Path("models") / "sam_vit_b_01ec64.pth"


def load_image(path, max_side=768):
    img = Image.open(path).convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    return np.array(img)


def load_cells_csv(path):
    """Read cell centers from a CSV with header x,y or x,y,z and an optional area column.

    Returns (points, areas); areas is None when the file has no area column.
    """
    import pandas as pd

    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    if "x" not in cols or "y" not in cols:
        raise ValueError("CSV needs columns named x and y (and optionally z, area)")
    axes = [cols["x"], cols["y"]] + ([cols["z"]] if "z" in cols else [])
    pts = np.ascontiguousarray(df[axes].to_numpy(dtype=np.float64))
    areas = df[cols["area"]].to_numpy(dtype=np.float64) if "area" in cols else None
    return pts, areas


def load_cells_full(path):
    """Like load_cells_csv but also returns a per-cell stiffness column when present:
    (points, areas_or_None, stiffness_or_None)."""
    import pandas as pd

    pts, areas = load_cells_csv(path)
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    stiff = df[cols["stiffness"]].to_numpy(dtype=np.float64) if "stiffness" in cols else None
    return pts, areas, stiff


def save_cells_csv(path, cellset):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([cellset.points, cellset.areas, cellset.stiffness])
    np.savetxt(path, data, delimiter=",", header="x,y,area,stiffness", comments="", fmt="%.6f")


def load_points_csv(path):
    return load_cells_csv(path)[0]


def save_points_csv(points, path, areas=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points)
    header = ",".join(["x", "y", "z"][: points.shape[1]])
    data = points
    if areas is not None:
        data = np.column_stack([points, np.asarray(areas)])
        header += ",area"
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.4f")


def otsu_threshold(gray):
    hist, edges = np.histogram(gray, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist).astype(float)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    valid = (w0 > 0) & (w1 > 0)
    between = np.zeros(len(centers))
    mean0 = m0[valid] / w0[valid]
    mean1 = (m0[-1] - m0[valid]) / w1[valid]
    between[valid] = w0[valid] * w1[valid] * (mean0 - mean1) ** 2
    return centers[int(between.argmax())]


def split_touching(fg, min_distance=6):
    """Split touching blobs with a distance-transform watershed (scikit-image).

    Seeds are the peaks of the smoothed distance map, at least min_distance pixels apart and found
    separately inside each connected blob, so a small cell next to a large one keeps its own seed.
    Returns an integer label image. Elongated cells can still be over-split, so check the count
    against ground truth when you have it (mequa.eval_segmentation).
    """
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    dist = ndi.distance_transform_edt(fg)
    comp, _ = ndi.label(fg)
    peaks = peak_local_max(ndi.gaussian_filter(dist, 0.7), min_distance=int(min_distance), labels=comp, exclude_border=False)
    if len(peaks) == 0:
        return comp
    markers = np.zeros(fg.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-dist, markers, mask=fg)


def segment_classical(image, min_area=20, split=False, min_distance=6):
    from scipy import ndimage as ndi

    gray = ndi.gaussian_filter(image.mean(axis=2), 1.5)
    fg = gray > otsu_threshold(gray)
    if fg.mean() > 0.5:
        fg = ~fg
    fg = ndi.binary_opening(fg, iterations=1)
    if split:
        labeled = split_touching(fg, min_distance=min_distance)
        count = int(labeled.max())
        fg = labeled > 0
    else:
        labeled, count = ndi.label(fg)
    if count == 0:
        return np.zeros((0, 2)), np.zeros(0)
    index = np.arange(1, count + 1)
    areas = ndi.sum(fg, labeled, index)
    cents = np.array(ndi.center_of_mass(fg, labeled, index))
    keep = areas >= min_area
    cents = cents[keep]
    return np.column_stack([cents[:, 1], cents[:, 0]]), areas[keep]


def ensure_checkpoint():
    if SAM_PATH.exists():
        return
    SAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading SAM ViT-B checkpoint (about 375 MB)...")
    urllib.request.urlretrieve(SAM_URL, str(SAM_PATH))


def model_type_of(path):
    name = Path(path).name.lower()
    for kind in ("vit_h", "vit_l", "vit_b"):
        if kind in name:
            return kind
    return None


def find_checkpoint(preferred=None):
    if preferred:
        kind = model_type_of(preferred)
        if kind is None:
            raise ValueError("checkpoint file name must contain vit_b, vit_l or vit_h")
        return Path(preferred), kind
    folder = SAM_PATH.parent
    found = []
    if folder.exists():
        for f in folder.glob("*.pth"):
            kind = model_type_of(f)
            if kind:
                found.append((("vit_b", "vit_l", "vit_h").index(kind), f, kind))
    if found:
        found.sort(key=lambda item: item[0])
        return found[0][1], found[0][2]
    ensure_checkpoint()
    return SAM_PATH, "vit_b"


def segment_sam(image, points_per_side=16, max_area_frac=0.05, min_area=20, checkpoint=None):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    path, kind = find_checkpoint(checkpoint)
    print(f"SAM checkpoint: {path} ({kind})")
    sam = sam_model_registry[kind](checkpoint=str(path))
    sam.to("cpu")
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.9,
        min_mask_region_area=min_area,
    )
    masks = generator.generate(image)
    h, w = image.shape[:2]
    cents = []
    areas = []
    for m in masks:
        if m["area"] > max_area_frac * h * w or m["area"] < min_area:
            continue
        ys, xs = np.nonzero(m["segmentation"])
        cents.append((xs.mean(), ys.mean()))
        areas.append(m["area"])
    return np.array(cents, dtype=float).reshape(-1, 2), np.array(areas, dtype=float)


def segment(image, method="classical", **kwargs):
    if method == "sam":
        cents, areas = segment_sam(image, **kwargs)
    else:
        cents, areas = segment_classical(image)
    h = image.shape[0]
    pts = np.column_stack([cents[:, 0], h - cents[:, 1]]) if len(cents) else np.zeros((0, 2))
    return pts, areas


def link_slices(slice_points, slice_areas, z_spacing=1.0, max_dist=6.0):
    """Turn per-slice cell detections from a z-stack into 3D cell centers.

    A cell cut by several adjacent slices shows up as one detection per slice. Detections in
    consecutive slices closer than max_dist (in pixels) are merged into one track. A track's
    x, y is its area-weighted mean, z is the area-weighted slice position times z_spacing, and
    its area is the largest cross-section seen. Greedy nearest-neighbor matching: simple and
    deterministic, but it can merge touching cells, so max_dist should be below a cell diameter.
    """
    from scipy.spatial import cKDTree

    tracks = []  # each: dict(sum_w, sum_xw, sum_yw, sum_zw, max_area, last_xy, last_slice)
    open_ids = []
    for zi, (pts, areas) in enumerate(zip(slice_points, slice_areas)):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        areas = np.asarray(areas, dtype=float)
        used = set()
        next_open = []
        if len(pts) and open_ids:
            prev_xy = np.array([tracks[t]["last_xy"] for t in open_ids])
            tree = cKDTree(prev_xy)
            dist, idx = tree.query(pts, k=1)
            order = np.argsort(dist)
            for i in order:
                if dist[i] > max_dist:
                    break
                t = open_ids[idx[i]]
                if t in used:
                    continue
                used.add(t)
                _extend(tracks[t], pts[i], areas[i], zi)
                next_open.append(t)
                dist[i] = np.inf
            matched = {i for i in range(len(pts)) if dist[i] == np.inf}
        else:
            matched = set()
        for i in range(len(pts)):
            if i in matched:
                continue
            tracks.append(_new_track(pts[i], areas[i], zi))
            next_open.append(len(tracks) - 1)
        open_ids = next_open
    if not tracks:
        return np.zeros((0, 3)), np.zeros(0)
    cents = np.array([[t["sx"] / t["sw"], t["sy"] / t["sw"], z_spacing * t["sz"] / t["sw"]] for t in tracks])
    return cents, np.array([t["max_area"] for t in tracks])


def _new_track(xy, area, zi):
    w = max(float(area), 1e-9)
    return {"sw": w, "sx": xy[0] * w, "sy": xy[1] * w, "sz": zi * w, "max_area": float(area), "last_xy": (xy[0], xy[1])}


def _extend(track, xy, area, zi):
    w = max(float(area), 1e-9)
    track["sw"] += w
    track["sx"] += xy[0] * w
    track["sy"] += xy[1] * w
    track["sz"] += zi * w
    track["max_area"] = max(track["max_area"], float(area))
    track["last_xy"] = (xy[0], xy[1])


def segment_stack(images, method="classical", z_spacing=1.0, max_dist=6.0, split=False, min_distance=6, **kwargs):
    """Segment every slice (SAM or classical) and link detections into 3D cell centers.

    Returns (points (n, 3), areas (n,)). Each slice is segmented independently; y is flipped
    to match the 2D convention used elsewhere (origin at the bottom left).
    """
    slice_pts, slice_areas = [], []
    for img in images:
        if method == "sam":
            cents, areas = segment_sam(img, **kwargs)
        else:
            cents, areas = segment_classical(img, split=split, min_distance=min_distance)
        slice_pts.append(cents)
        slice_areas.append(areas)
    pts3, areas3 = link_slices(slice_pts, slice_areas, z_spacing=z_spacing, max_dist=max_dist)
    if len(pts3):
        pts3[:, 1] = images[0].shape[0] - pts3[:, 1]
    return pts3, areas3


def make_demo_stack(n_cells=60, n_slices=12, size=256, radius=9.0, z_step=3.0, seed=5):
    """Synthetic z-stack of bright-on-light spheres, for tests and offline demos. Returns
    (images, truth) where truth is an (n, 3) array of sphere centers in (x, y, slice-units)."""
    rng = np.random.default_rng(seed)
    truth = []
    while len(truth) < n_cells:
        c = np.array([rng.uniform(20, size - 20), rng.uniform(20, size - 20), rng.uniform(1, n_slices - 2) * z_step])
        if all(np.linalg.norm(c - t) > 2.2 * radius for t in truth):
            truth.append(c)
    truth = np.array(truth)
    yy, xx = np.mgrid[0:size, 0:size]
    images = []
    for s in range(n_slices):
        z = s * z_step
        canvas = np.full((size, size, 3), 235.0)
        for cx, cy, cz in truth:
            r2 = radius**2 - (z - cz) ** 2
            if r2 > 4:
                canvas[(xx - cx) ** 2 + (yy - cy) ** 2 <= r2] = (60, 40, 120)
        images.append(np.clip(canvas + rng.normal(0, 3, canvas.shape), 0, 255).astype(np.uint8))
    truth[:, 2] /= z_step
    return images, truth


def make_demo_image(path, n=400, seed=3, size=640):
    rng = np.random.default_rng(seed)
    centers = rng.random((3, 2)) * (size * 0.6) + size * 0.2
    img = Image.new("RGB", (size, size), (238, 226, 236))
    draw = ImageDraw.Draw(img)
    placed = []
    attempts = 0
    while len(placed) < n and attempts < n * 40:
        attempts += 1
        c = centers[rng.integers(0, 3)]
        x, y = c + rng.normal(0, size * 0.09, 2)
        if not (10 < x < size - 10 and 10 < y < size - 10):
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < 14**2 for px, py in placed):
            continue
        r = rng.uniform(4.5, 6.5)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(70, 40, 120))
        placed.append((x, y))
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    arr = np.array(img).astype(float) + rng.normal(0, 4, (size, size, 3))
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)
    return len(placed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--method", choices=["classical", "sam"], default="classical")
    parser.add_argument("--out", default="data/cells.csv")
    parser.add_argument("--synthetic", default=None)
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--max-area-frac", type=float, default=0.05)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    if args.synthetic:
        count = make_demo_image(args.synthetic)
        print(f"wrote {args.synthetic} with {count} synthetic nuclei")
        return
    if not args.image:
        parser.error("image path required")
    image = load_image(args.image)
    if args.method == "sam":
        pts, areas = segment(image, "sam", points_per_side=args.points_per_side, max_area_frac=args.max_area_frac, checkpoint=args.checkpoint)
    else:
        pts, areas = segment(image, "classical")
    save_points_csv(pts, args.out)
    print(f"{args.method}: {len(pts)} cells -> {args.out}")


if __name__ == "__main__":
    main()
