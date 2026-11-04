"""Download Segment Anything Model checkpoints.

Default is SAM ViT-B because it is the most practical for CPU / 16 GB RAM.
SAM 3.1 runs hosted through the Meta Model API (no local weights needed);
`--model sam3.1` only prints where the gated local checkpoint lives.

Usage:
    python scripts/download_sam_checkpoint.py
    python scripts/download_sam_checkpoint.py --model vit_b
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "sam"


CHECKPOINTS = {
    "vit_b": (
        "sam_vit_b_01ec04.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec04.pth",
    ),
    "vit_l": (
        "sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    ),
    "vit_h": (
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ),
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_suffix(destination.suffix + ".part")

    def hook(block_number: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            downloaded = block_number * block_size
            percent = min(100.0, downloaded * 100.0 / total_size)
            print(f"\rDownloading: {percent:6.2f}%", end="")

    print(f"URL: {url}")
    print(f"Destination: {destination}")

    with urllib.request.urlopen(url) as response:
        with temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    print("\nDownload complete.")
    temporary.rename(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SAM checkpoint.")
    parser.add_argument(
        "--model",
        choices=list(CHECKPOINTS.keys()) + ["sam3.1"],
        default="vit_b",
        help="SAM model variant. Use vit_b for CPU laptops; sam3.1 prints hosted-setup info.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if the file already exists.",
    )

    args = parser.parse_args()

    if args.model == "sam3.1":
        print("SAM 3.1 is used hosted (model 'sam-3.1' on the Meta Model API).")
        print("No local weights are needed: set META_API_KEY and install meta-sam-parser.")
        print("For a local copy instead, accept the SAM License and fetch the gated")
        print("checkpoint 'facebook/sam3.1' from https://huggingface.co/facebook/sam3.1")
        print(f"(file sam3.1_multiplex.pt, about 3.4 GB) into: {MODEL_DIR}")
        print("Note: local SAM 3.1 needs a CUDA GPU; this loader does not run it on CPU.")
        return 0

    filename, url = CHECKPOINTS[args.model]
    destination = MODEL_DIR / filename

    if destination.exists() and not args.force:
        print(f"Checkpoint already exists: {destination}")
        print("Use --force to download again.")
        return 0

    try:
        download(url, destination)
    except Exception as exc:
        print(f"\nERROR: download failed: {exc}", file=sys.stderr)
        print("\nManual fallback:")
        print(f"1. Download: {url}")
        print(f"2. Save as:  {destination}")
        return 1

    print(f"Saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())