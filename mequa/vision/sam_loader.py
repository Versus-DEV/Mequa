"""SAM loader for Mequa.

This module loads Segment Anything and produces masks.
It is intentionally CPU-first and hackathon-safe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "models" / "sam" / "sam_vit_b_01ec04.pth"


@dataclass
class SamResult:
    masks: List[np.ndarray]
    scores: List[float]
    source: str
    message: str


class SamLoader:
    """Loads SAM lazily and generates masks from an RGB image."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: Optional[str] = None,
        device: str = "cpu",
        max_image_side: int = 512,
        max_masks: int = 300,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.50,
        stability_score_thresh: float = 0.70,
        min_mask_region_area: int = 100,
        torch_threads: int = 4,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._model_type = model_type
        self.device = device

        self.max_image_side = max_image_side
        self.max_masks = max_masks
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self.torch_threads = torch_threads

        self._model = None
        self._generator = None

    def configure(self, points_per_side=None, max_masks=None, max_image_side=None) -> None:
        """Change settings between runs. The mask generator is built once, so changing the
        prompt-grid density must drop it or the new value would silently be ignored."""
        if points_per_side is not None and int(points_per_side) != self.points_per_side:
            self.points_per_side = int(points_per_side)
            self._generator = None
        if max_masks is not None:
            self.max_masks = int(max_masks)
        if max_image_side is not None:
            self.max_image_side = int(max_image_side)

    @property
    def checkpoint_path(self) -> Path:
        if self._checkpoint_path:
            return Path(self._checkpoint_path)

        environment_path = os.getenv("SAM_CHECKPOINT")
        if environment_path:
            return Path(environment_path)

        return DEFAULT_CHECKPOINT

    @property
    def model_type(self) -> str:
        if self._model_type:
            return self._model_type

        environment_type = os.getenv("SAM_MODEL_TYPE")
        if environment_type:
            return environment_type

        name = self.checkpoint_path.name.lower()

        if "vit_h" in name:
            return "vit_h"

        if "vit_l" in name:
            return "vit_l"

        return "vit_b"

    def is_available(self) -> Tuple[bool, str]:
        """Cheap availability check without loading the full model."""
        checkpoint = self.checkpoint_path

        if not checkpoint.exists():
            return False, (
                f"SAM checkpoint missing: {checkpoint}. "
                "Run: python scripts/download_sam_checkpoint.py --model vit_b"
            )

        try:
            import cv2  # noqa: F401
        except Exception:
            return False, "opencv-python is missing. Install with: pip install opencv-python"

        try:
            import torch  # noqa: F401
        except Exception:
            return False, "torch is missing. Install CPU torch first."

        try:
            import segment_anything  # noqa: F401
        except Exception:
            return False, "segment-anything is missing. Install SAM dependencies."

        return True, (
            f"SAM available. checkpoint={checkpoint}, model_type={self.model_type}"
        )

    def load(self) -> None:
        """Load SAM and create the automatic mask generator."""
        if self._generator is not None:
            return

        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        if self.torch_threads and self.torch_threads > 0:
            try:
                torch.set_num_threads(int(self.torch_threads))
            except Exception:
                pass

        checkpoint = self.checkpoint_path

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint}. "
                "Run: python scripts/download_sam_checkpoint.py --model vit_b"
            )

        model = sam_model_registry[self.model_type](checkpoint=str(checkpoint))
        model.to(self.device)

        self._model = model
        self._generator = SamAutomaticMaskGenerator(
            model=model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )

    def generate_masks(self, image_rgb: np.ndarray) -> SamResult:
        """Generate SAM masks from an RGB image."""
        from .image_io import resize_max_side

        image = np.asarray(image_rgb)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected RGB image with shape (H, W, 3).")

        if self.max_image_side and self.max_image_side > 0:
            image = resize_max_side(image, self.max_image_side)

        image = np.ascontiguousarray(image, dtype=np.uint8)

        self.load()

        raw_masks = self._generator.generate(image)

        if not raw_masks:
            return SamResult(
                masks=[],
                scores=[],
                source="sam",
                message="SAM returned no masks.",
            )

        raw_masks.sort(
            key=lambda item: (
                float(item.get("stability_score", 0.0)),
                float(item.get("area", 0.0)),
            ),
            reverse=True,
        )

        if self.max_masks and self.max_masks > 0:
            raw_masks = raw_masks[: self.max_masks]

        masks: List[np.ndarray] = []
        scores: List[float] = []

        for item in raw_masks:
            segmentation = item.get("segmentation")

            if segmentation is None:
                continue

            mask = np.asarray(segmentation, dtype=bool)

            if mask.size == 0:
                continue

            masks.append(mask)
            scores.append(
                float(item.get("stability_score", item.get("predicted_iou", 0.0)))
            )

        return SamResult(
            masks=masks,
            scores=scores,
            source="sam",
            message=f"Generated {len(masks)} masks.",
        )