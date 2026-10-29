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


def load_points_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return np.ascontiguousarray(data[:, :2], dtype=np.float64)


def save_points_csv(points, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, points, delimiter=",", header="x,y", comments="", fmt="%.4f")


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


def segment_classical(image, min_area=20):
    from scipy import ndimage as ndi

    gray = ndi.gaussian_filter(image.mean(axis=2), 1.5)
    fg = gray > otsu_threshold(gray)
    if fg.mean() > 0.5:
        fg = ~fg
    fg = ndi.binary_opening(fg, iterations=1)
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
