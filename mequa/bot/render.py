"""Strain map as a PNG, drawn with Pillow so the bot needs no extra plotting dependency."""

from __future__ import annotations

import io

import numpy as np

LOW = (40, 70, 140)
HIGH = (235, 64, 52)
BACKGROUND = (11, 18, 32)


def _ramp(t: float):
    t = min(1.0, max(0.0, float(t)))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(LOW, HIGH))


def strain_map(points, strain, title: str = "Strain map", size: int = 720) -> bytes:
    from PIL import Image, ImageDraw

    pts = np.asarray(points, dtype=float)[:, :2]
    s = np.asarray(strain, dtype=float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    margin = 48
    scale = (size - 2 * margin) / max(float((hi - lo).max()), 1e-9)
    top = float(np.percentile(s, 99)) if len(s) else 0.0
    top = max(top, 1e-12)
    radius = 3 if len(pts) <= 3000 else 2
    img = Image.new("RGB", (size, size + 64), BACKGROUND)
    draw = ImageDraw.Draw(img)
    draw.text((margin, 16), title, fill=(230, 235, 245))
    for (x, y), v in zip(pts, s):
        px = margin + (x - lo[0]) * scale
        py = margin + (hi[1] - y) * scale
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=_ramp(v / top))
    bar_y = size + 20
    bar_w = size - 2 * margin
    for i in range(bar_w):
        draw.line([(margin + i, bar_y), (margin + i, bar_y + 14)], fill=_ramp(i / max(bar_w - 1, 1)))
    draw.text((margin, bar_y + 20), "0", fill=(230, 235, 245))
    draw.text((size - margin - 90, bar_y + 20), f"{top:.4f} (p99)", fill=(230, 235, 245))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
