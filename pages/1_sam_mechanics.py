"""SAM Mechanics page.

Core loop:

image / demo masks
  ↓
SAM masks
  ↓
geometry features
  ↓
stiffness values
  ↓
saved outputs
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import streamlit as st

from mequa.utils.env import load_env_file
from mequa.vision.demo_masks import make_demo_image, make_demo_masks
from mequa.vision.image_io import (
    decode_image_bytes,
    load_image_file,
    resize_max_side,
    save_image_rgb,
)
from mequa.vision.mask_overlay import overlay_masks
from mequa.vision.cells import cells_from_masks
from mequa.vision.sam_loader import SamLoader
from mequa import ingest
from mequa.vision.sam_physics_parameterizer import extract_geometry_from_masks
from mequa.physics.stiffness_from_geometry import (
    geometry_to_stiffness,
    stiffness_array,
)


from mequa import ui  # noqa: E402

load_env_file()

ui.page_header('SAM Mechanics', 'Image to masks to cells to per-cell stiffness. Research use only, no clinical guidance.')

OUTPUT_ROOT = ROOT / "outputs"
SAM_OUT = OUTPUT_ROOT / "sam"
SOLVER_OUT = OUTPUT_ROOT / "solver"


@st.cache_resource
def get_loader() -> SamLoader:
    return SamLoader()


loader = get_loader()
sam_ok, sam_message = loader.is_available()


with st.expander("SAM status", expanded=not sam_ok):
    if sam_ok:
        st.success(sam_message)
    else:
        st.warning(sam_message)

    st.write(f"Checkpoint: `{loader.checkpoint_path}`")
    st.write(f"Model type: `{loader.model_type}`")


with st.sidebar:
    st.header("Inputs")

    source = st.radio(
        "Source",
        [
            "Demo masks",
            "Upload image",
            "Example image",
        ],
    )

    uploaded_file = None

    if source == "Upload image":
        uploaded_file = st.file_uploader(
            "PNG/JPG image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
        )

    max_side = st.slider("Max image side", 256, 1024, 512, 64)
    points_per_side = st.select_slider(
        "SAM prompt grid (points per side)",
        options=[8, 12, 16, 24, 32],
        value=16,
        help="Denser grids find more cells and take longer on CPU: roughly the square of this number.",
    )
    max_masks = st.slider("Max masks", 1, 400, 200)

    min_area = st.number_input(
        "Min region area px",
        min_value=10,
        max_value=2000,
        value=50,
        step=10,
    )

    run = st.button("Run Pipeline", type="primary")


example_path = ROOT / "data" / "images" / "bbbc_sample_0001.png"

if source == "Example image":
    if example_path.exists():
        st.caption(f"Example image: `{example_path}`")
    else:
        st.caption(
            f"Example image not found. Put a licensed image at `{example_path}` "
            "or use Demo masks / Upload image."
        )


def csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""

    buffer = io.StringIO()
    fieldnames = list(rows[0].keys())

    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    return buffer.getvalue().encode("utf-8")


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def save_outputs(
    image: np.ndarray,
    masks: list[np.ndarray],
    features_k: list[dict],
    stiffness: np.ndarray,
    source_label: str,
    duration_s: float,
) -> dict:
    SAM_OUT.mkdir(parents=True, exist_ok=True)
    SOLVER_OUT.mkdir(parents=True, exist_ok=True)

    if masks:
        mask_array = np.stack([mask.astype(bool) for mask in masks], axis=0)
    else:
        height, width = image.shape[:2]
        mask_array = np.zeros((0, height, width), dtype=bool)

    np.save(SAM_OUT / "masks.npy", mask_array)

    overlay = overlay_masks(image, masks)

    try:
        save_image_rgb(SAM_OUT / "mask_overlay.png", overlay)
    except Exception:
        pass

    geometry_csv = SAM_OUT / "geometry_features.csv"

    if features_k:
        fieldnames = list(features_k[0].keys())

        with geometry_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(features_k)
    else:
        geometry_csv.write_text("", encoding="utf-8")

    stiffness_npy = SOLVER_OUT / "stiffness_map.npy"
    np.save(stiffness_npy, stiffness)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_label,
        "mask_count": len(masks),
        "region_count": len(features_k),
        "stiffness_count": int(stiffness.size),
        "stiffness_min": None if stiffness.size == 0 else float(np.min(stiffness)),
        "stiffness_max": None if stiffness.size == 0 else float(np.max(stiffness)),
        "stiffness_mean": None if stiffness.size == 0 else float(np.mean(stiffness)),
        "duration_seconds": round(duration_s, 4),
        "outputs": {
            "masks_npy": str((SAM_OUT / "masks.npy").relative_to(ROOT)),
            "mask_overlay_png": str((SAM_OUT / "mask_overlay.png").relative_to(ROOT)),
            "geometry_csv": str(geometry_csv.relative_to(ROOT)),
            "stiffness_npy": str(stiffness_npy.relative_to(ROOT)),
        },
    }

    (SOLVER_OUT / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


def run_pipeline():
    try:
        start = time.perf_counter()

        if source == "Demo masks":
            masks = make_demo_masks()
            image = make_demo_image(masks)
            source_label = "demo"
            run_message = "Demo masks used. SAM not called."

        else:
            if not sam_ok:
                st.error(sam_message)
                return None

            if source == "Upload image":
                if uploaded_file is None:
                    st.error("Upload an image first.")
                    return None

                image_full = decode_image_bytes(uploaded_file.read())

            else:
                if not example_path.exists():
                    st.error(f"Example image missing: {example_path}")
                    return None

                image_full = load_image_file(example_path)

            image = resize_max_side(image_full, max_side)

            loader.configure(points_per_side=points_per_side, max_masks=max_masks, max_image_side=max_side)

            with st.spinner("Running SAM on CPU. This can take a while..."):
                result = loader.generate_masks(image)

            masks = result.masks
            source_label = f"sam:{source.lower().replace(' ', '-')}"
            run_message = result.message

        cellset = cells_from_masks(masks, min_area=int(min_area))
        features_k = cellset.features
        stiffness = cellset.stiffness
        ingest.save_cells_csv(OUTPUT_ROOT / "sam" / "cells.csv", cellset)

        duration = time.perf_counter() - start

        summary = save_outputs(
            image=image,
            masks=masks,
            features_k=features_k,
            stiffness=stiffness,
            source_label=source_label,
            duration_s=duration,
        )

        return {
            "image": image,
            "masks": masks,
            "features": features_k,
            "stiffness": stiffness,
            "summary": summary,
            "run_message": run_message,
        }

    except Exception as exc:
        st.exception(exc)
        return None


if run:
    result = run_pipeline()

    if result is not None:
        st.session_state["result"] = result


result = st.session_state.get("result")

if result is None:
    st.info("Choose a source and click Run pipeline.")
    st.stop()

image = result["image"]
masks = result["masks"]
features_k = result["features"]
stiffness = result["stiffness"]
summary = result["summary"]

tab_input, tab_masks, tab_geometry, tab_stiffness, tab_outputs = st.tabs(
    [
        "Input",
        "Masks",
        "Geometry",
        "Stiffness",
        "Outputs",
    ]
)

with tab_input:
    st.image(image, caption=f"Source: {summary['source']}")
    st.write(result["run_message"])

with tab_masks:
    overlay = overlay_masks(image, masks)
    st.image(overlay, caption=f"{len(masks)} masks")

with tab_geometry:
    if features_k:
        st.dataframe(features_k)
    else:
        st.warning("No regions passed the min-area filter.")

with tab_stiffness:
    if stiffness.size:
        column_1, column_2, column_3 = st.columns(3)

        column_1.metric("Min k", f"{summary['stiffness_min']:.4f}")
        column_2.metric("Max k", f"{summary['stiffness_max']:.4f}")
        column_3.metric("Mean k", f"{summary['stiffness_mean']:.4f}")

        st.download_button(
            "Download geometry CSV",
            data=csv_bytes(features_k),
            file_name="geometry_features.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download stiffness NPY",
            data=npy_bytes(stiffness),
            file_name="stiffness_map.npy",
            mime="application/octet-stream",
        )
    else:
        st.warning("No stiffness values were produced.")

with tab_outputs:
    st.json(summary)