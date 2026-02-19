"""
Depth estimation utilities — Triangle Similarity + MiDaS scaling.

No specialised hardware is used. Distance is derived from:
1. MiDaS relative depth map  →  scaled to approximate metres.
2. Triangle-Similarity fallback for known-size objects.
"""

from __future__ import annotations

import numpy as np

# ── Calibration constants (tweak for your camera) ──────────────────────
FOCAL_LENGTH_PX: float = 600.0  # Approximate focal length in pixels
MIDAS_SCALE_FACTOR: float = 2.5  # Maps MiDaS inverse-depth to metres

# Known real-world widths (metres) for common COCO classes
KNOWN_WIDTHS: dict[str, float] = {
    "person": 0.45,
    "chair": 0.50,
    "car": 1.80,
    "bicycle": 0.60,
    "dog": 0.35,
    "cat": 0.25,
    "bottle": 0.08,
    "cup": 0.08,
    "laptop": 0.35,
    "tv": 0.90,
    "couch": 1.80,
    "bed": 1.50,
    "dining table": 1.20,
    "suitcase": 0.50,
    "backpack": 0.35,
    "umbrella": 1.00,
    "handbag": 0.30,
    "tie": 0.08,
    "book": 0.22,
    "cell phone": 0.07,
    "potted plant": 0.30,
    "refrigerator": 0.70,
    "bench": 1.20,
    "bus": 2.50,
    "truck": 2.40,
    "motorcycle": 0.80,
    "fire hydrant": 0.30,
    "stop sign": 0.60,
    "traffic light": 0.30,
}


def triangle_similarity_distance(
    label: str,
    bbox_width_px: float,
) -> float | None:
    """Return distance in metres using triangle similarity, or *None* if
    the object's real width is unknown."""
    real_width = KNOWN_WIDTHS.get(label)
    if real_width is None or bbox_width_px <= 0:
        return None
    return round((real_width * FOCAL_LENGTH_PX) / bbox_width_px, 2)


def midas_depth_at_bbox(
    depth_map: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    """Return the estimated distance (metres) for the centre region of a
    bounding box using the MiDaS depth map.

    *depth_map* contains **inverse** relative depth (higher = closer).
    We take the median of the central 50 % of the bbox to avoid edge noise.
    """
    x1, y1, x2, y2 = bbox
    h, w = depth_map.shape[:2]

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Central 50 % crop
    cx = int((x2 - x1) * 0.25)
    cy = int((y2 - y1) * 0.25)
    region = depth_map[y1 + cy : y2 - cy, x1 + cx : x2 - cx]

    if region.size == 0:
        return 0.0

    median_inv = float(np.median(region))
    if median_inv <= 0:
        return 0.0

    distance = MIDAS_SCALE_FACTOR / (median_inv / 255.0 + 1e-6)
    return round(min(distance, 30.0), 2)  # cap at 30 m


def classify_direction(cx: float, frame_width: int) -> str:
    """Classify the horizontal position of an object into left / center / right."""
    third = frame_width / 3
    if cx < third:
        return "left"
    elif cx < 2 * third:
        return "center"
    else:
        return "right"
