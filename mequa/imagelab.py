"""Image Lab core: take a real microscopy image or z-stack from the user, find the nuclei, hand back cells.

Nothing here is preloaded. Whatever the user uploads (PNG/JPG/TIFF, a multi-page TIFF, a zip of slices or
a zip holding one big TIFF) is read, turned into one grey-scale array, and run through the same detector
the BBBC033 benchmark uses (see real3d.detect_nuclei_3d). 2D images use the 2D version of the same idea.
"""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy import ndimage as ndi

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")
TIFF_EXT = (".tif", ".tiff")
MAX_VOXELS = 400_000_000   # refuse something that would not fit in memory; ask for downsampling instead


class ImageLabError(ValueError):
    """A problem the user can fix; the message says how."""


@dataclass
class Detection:
    points: np.ndarray                       # (n, 2) x,y or (n, 3) x,y,z in the ORIGINAL pixel/voxel frame
    areas: Optional[np.ndarray]              # per-cell area in original pixels (2D only)
    kind: str                                # "2D image" or "3D stack"
    shape: tuple                             # shape of the array that was analysed (after downsampling)
    downsample: int = 1
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(len(self.points))

    @property
    def dim(self) -> int:
        return int(self.points.shape[1]) if self.n else 2


# ------------------------------------------------------------------------------------------ loading

def _gray(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) if arr.ndim == 2 else arr.astype(np.float32).mean(axis=-1)


def _from_tiff(data: bytes) -> np.ndarray:
    import tifffile

    arr = np.squeeze(tifffile.imread(io.BytesIO(data)))
    return _normalise_dims(arr)


def _normalise_dims(arr: np.ndarray) -> np.ndarray:
    """Return a float32 array that is (y, x) or (z, y, x). Colour images are averaged to grey."""
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4) and arr.shape[0] > 4:        # (y, x, rgb)
            return _gray(arr[..., :3])
        if arr.shape[0] in (3, 4) and arr.shape[-1] > 4:        # (rgb, y, x)
            return arr[:3].astype(np.float32).mean(axis=0)
        return arr.astype(np.float32)
    if arr.ndim == 4:
        # (z, y, x, c) or (c, z, y, x): average the channel axis, which is the shortest of the outer ones
        ax = 3 if arr.shape[-1] <= 4 else 0
        return arr.astype(np.float32).mean(axis=ax)
    raise ImageLabError(f"cannot use an image with {arr.ndim} dimensions (shape {arr.shape}); expected 2D or a z-stack")


def _from_image(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        im.load()
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            return np.asarray(im).astype(np.float32)
        return _gray(np.asarray(im.convert("RGB" if im.mode not in ("L", "F") else im.mode)))


def load_array(data: bytes, name: str) -> np.ndarray:
    """Read an uploaded file into a float32 array, (y, x) for an image or (z, y, x) for a stack."""
    ext = PurePosixPath(name.lower()).suffix
    if ext in TIFF_EXT:
        return _from_tiff(data)
    if ext in IMAGE_EXT:
        return _from_image(data)
    if ext == ".zip":
        try:
            z = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise ImageLabError("that .zip file could not be opened") from exc
        files = [i for i in z.infolist() if not i.is_dir() and "__MACOSX" not in i.filename and not PurePosixPath(i.filename).name.startswith(".")]
        tifs = [i for i in files if i.filename.lower().endswith(TIFF_EXT)]
        imgs = [i for i in files if i.filename.lower().endswith(IMAGE_EXT)]
        if len(tifs) == 1:
            return _from_tiff(z.read(tifs[0]))
        pool = sorted(tifs or imgs, key=lambda i: i.filename)
        if not pool:
            raise ImageLabError("the zip holds no .tif/.png/.jpg files")
        slices = [_from_tiff(z.read(i)) if i.filename.lower().endswith(TIFF_EXT) else _from_image(z.read(i)) for i in pool]
        if any(s.ndim != 2 for s in slices):
            raise ImageLabError("a zip of several files must hold one 2D slice per file; for one multi-page stack, upload that .tif by itself")
        if len({s.shape for s in slices}) != 1:
            raise ImageLabError("the slices in the zip do not all have the same size")
        return np.stack(slices)
    raise ImageLabError(f"unsupported file type '{ext}'. Use .tif/.tiff, .png, .jpg or a .zip of slices.")


def combine_slices(files: List[tuple]) -> np.ndarray:
    """Several uploaded files -> one array. One file is read as is; several single-slice files become a stack, ordered by name."""
    if not files:
        raise ImageLabError("upload an image first")
    if len(files) == 1:
        return load_array(files[0][1], files[0][0])
    arrs = [load_array(d, n) for n, d in sorted(files, key=lambda t: t[0])]
    if any(a.ndim != 2 for a in arrs):
        raise ImageLabError("when uploading several files, each must be one 2D slice")
    if len({a.shape for a in arrs}) != 1:
        raise ImageLabError("the slices do not all have the same size")
    return np.stack(arrs)


def describe(arr: np.ndarray) -> Dict[str, object]:
    return {"kind": "3D stack" if arr.ndim == 3 else "2D image", "shape": tuple(int(s) for s in arr.shape),
            "min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean()), "voxels": int(arr.size)}


def downsample(arr: np.ndarray, f: int) -> np.ndarray:
    """Block-average the x and y axes by f (z is left alone)."""
    if f <= 1:
        return arr
    h, w = arr.shape[-2] // f * f, arr.shape[-1] // f * f
    a = arr[..., :h, :w]
    shp = a.shape[:-2] + (h // f, f, w // f, f)
    return a.reshape(shp).mean(axis=(-3, -1)).astype(np.float32)


def suggest_downsample(arr: np.ndarray, budget: int = 40_000_000) -> int:
    f = 1
    while arr.size / (f * f) > budget and f < 8:
        f *= 2
    return f


# ------------------------------------------------------------------------------------------ detection

def _markers_2d(mask: np.ndarray, h_frac: float):
    from skimage.morphology import h_maxima

    mask = np.pad(mask, 1, constant_values=False)
    dist = ndi.distance_transform_edt(mask).astype(np.float32)
    r_typ = float(np.percentile(dist[mask], 95)) if mask.any() else 0.0
    sm = ndi.gaussian_filter(dist, max(1.0, 0.1 * r_typ))
    hm = h_maxima(sm, max(0.5, h_frac * r_typ))
    lab, n = ndi.label(hm, structure=np.ones((3, 3)))
    if n == 0:
        return mask[1:-1, 1:-1], np.zeros(mask[1:-1, 1:-1].shape, np.int32), sm[1:-1, 1:-1], r_typ
    idx = np.arange(1, n + 1)
    keep = ndi.maximum(sm, lab, idx) >= 0.35 * r_typ
    new = np.zeros(n + 1, np.int32)
    new[1:][keep] = np.arange(1, int(keep.sum()) + 1)
    return mask[1:-1, 1:-1], new[lab][1:-1, 1:-1], sm[1:-1, 1:-1], r_typ


def detect_2d(img: np.ndarray, sigma: float = 1.5, bright: bool = True, min_area: int = 20, h_frac: float = 0.25):
    """Smooth, Otsu, one marker per nucleus (h-maxima of the distance map), watershed for the area of each."""
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed

    v = ndi.gaussian_filter(img.astype(np.float32), sigma)
    if v.max() <= v.min():
        raise ImageLabError("the image is a single flat colour; there is nothing to detect")
    if not bright:
        v = v.max() - v
    mask = v > threshold_otsu(v)
    if mask.mean() > 0.6:
        raise ImageLabError("more than 60% of the image is above threshold. If the nuclei are dark on a bright background, untick 'bright nuclei'.")
    mask, markers, sm, _ = _markers_2d(mask, h_frac)
    if markers.max() == 0:
        return np.zeros((0, 2)), np.zeros(0)
    lab = watershed(-sm, markers, mask=mask)
    idx = np.arange(1, int(lab.max()) + 1)
    areas = ndi.sum(mask, lab, idx)
    cents = np.array(ndi.center_of_mass(mask, lab, idx))
    keep = areas >= min_area
    return np.column_stack([cents[keep, 1], cents[keep, 0]]), areas[keep]


def detect(arr: np.ndarray, *, sigma: float = 2.0, bright: bool = True, z_scale: float = 1.0, downsample_by: int = 1,
           min_area: int = 20, progress: Optional[Callable[[str], None]] = None) -> Detection:
    """Find nuclei in an uploaded image (2D) or stack (3D). Coordinates are returned in the original pixel frame."""
    t0 = time.time()
    if arr.size > MAX_VOXELS:
        raise ImageLabError(f"{arr.size / 1e6:.0f} million voxels is too large to hold in memory; choose a larger downsample factor.")
    f = max(1, int(downsample_by))
    a = downsample(arr, f)
    notes = []
    if f > 1:
        notes.append(f"x and y were reduced {f}x before detection; coordinates are scaled back to the original image")
    if a.ndim == 2:
        if progress:
            progress("detecting nuclei")
        pts, areas = detect_2d(a, sigma=max(0.5, sigma * 0.75), bright=bright, min_area=max(3, int(min_area / (f * f))))
        pts = pts * f
        return Detection(pts, None if not len(areas) else areas * f * f, "2D image", tuple(a.shape), f, time.time() - t0, notes)
    from mequa import real3d

    v = a if bright else (a.max() - a)
    if v.shape[0] < 3:
        raise ImageLabError("a z-stack needs at least 3 slices; for one or two slices upload them as 2D images")
    pts = real3d.detect_nuclei_3d(v, sigma=sigma, min_distance="auto", z_scale=z_scale, progress=progress)
    if len(pts):
        pts[:, :2] *= f
    return Detection(pts, None, "3D stack", tuple(a.shape), f, time.time() - t0, notes)


def stiffness_from_detection(det: Detection) -> Optional[np.ndarray]:
    """Per-cell stiffness from cell area (larger cell = softer), the same rule the SAM page uses. None for 3D."""
    if det.areas is None or not len(det.areas):
        return None
    from mequa import numpy_backend as nb

    return nb.stiffness_from_areas(det.areas)


# ------------------------------------------------------------------------------------------ display helpers

def preview_slice(arr: np.ndarray, index: Optional[int] = None, max_side: int = 900) -> np.ndarray:
    """An 8-bit grey picture of one slice (or the image), contrast-stretched, no larger than max_side."""
    img = arr if arr.ndim == 2 else arr[int(np.clip(index if index is not None else arr.shape[0] // 2, 0, arr.shape[0] - 1))]
    lo, hi = np.percentile(img, [1, 99.8])
    if hi <= lo:
        hi = lo + 1
    g = (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    f = int(np.ceil(max(g.shape) / max_side))
    return g[::f, ::f] if f > 1 else g


def make_synthetic_stack(n: int = 12, shape=(24, 300, 300), seed: int = 0) -> np.ndarray:
    """A made-up fluorescence stack for checking the pipeline. It is labelled synthetic wherever it is shown."""
    rng = np.random.default_rng(seed)
    z, y, x = shape
    zz, yy, xx = np.mgrid[0:z, 0:y, 0:x]
    vol = rng.normal(100, 15, shape).astype(np.float32)
    placed = []
    tries = 0
    while len(placed) < n and tries < 500:
        tries += 1
        c = (rng.uniform(8, z - 8), rng.uniform(30, y - 30), rng.uniform(30, x - 30))
        if all(np.hypot(c[1] - p[1], c[2] - p[2]) > 55 for p in placed):
            placed.append(c)
    for c in placed:
        m = (((zz - c[0]) / 5.0) ** 2 + ((yy - c[1]) / 22.0) ** 2 + ((xx - c[2]) / 20.0) ** 2) <= 1
        vol += 600 * m * (0.8 + 0.4 * ndi.gaussian_filter(rng.normal(0, 1, shape).astype(np.float32), 3))
    return ndi.gaussian_filter(vol, 0.7)
