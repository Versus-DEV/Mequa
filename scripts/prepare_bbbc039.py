"""Unpack a few BBBC039 images and their expert masks into data/images and data/gt.

Download from https://bbbc.broadinstitute.org/BBBC039 :  images.zip  and  masks.zip
(CC0 per the dataset page; check that line yourself and fill data/images/LICENSE_CHECK.md).

    python scripts/prepare_bbbc039.py --images images.zip --masks masks.zip --count 3
"""

import argparse
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--count", type=int, default=3)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    img_dir, gt_dir = root / "data" / "images", root / "data" / "gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    zi, zm = zipfile.ZipFile(args.images), zipfile.ZipFile(args.masks)
    masks = {Path(n).stem: n for n in zm.namelist() if n.lower().endswith(".png")}
    images = sorted(n for n in zi.namelist() if n.lower().endswith((".tif", ".tiff")))
    done = 0
    for n in images:
        stem = Path(n).stem
        if stem not in masks:
            continue
        (img_dir / Path(n).name).write_bytes(zi.read(n))
        (gt_dir / Path(masks[stem]).name).write_bytes(zm.read(masks[stem]))
        print(f"extracted {Path(n).name}  +  {Path(masks[stem]).name}")
        done += 1
        if done >= args.count:
            break
    if done == 0:
        print("No image had a matching mask name. Listing a few names from each zip:")
        print(images[:3], list(masks)[:3])


if __name__ == "__main__":
    main()
