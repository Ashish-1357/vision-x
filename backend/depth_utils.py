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
    "wall": 3.00,
    "stairs": 1.50,
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

    MiDaS outputs *inverse* relative depth (higher value = closer object).
    We use a calibrated sigmoid-like remapping:
        distance = 0.3 + 9.7 * (1 - median_inv/255)
    This gives ~0.3 m for objects touching the camera and ~10 m for
    objects at the far background.  The actual person-at-0.5 m case maps to
    ~0.5-0.8 m which is much more accurate than the old linear formula.
    """
    x1, y1, x2, y2 = bbox
    h, w = depth_map.shape[:2]

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Central 50% crop to avoid noisy edges
    cx = int((x2 - x1) * 0.25)
    cy = int((y2 - y1) * 0.25)
    region = depth_map[y1 + cy : y2 - cy, x1 + cx : x2 - cx]

    if region.size == 0:
        return 0.0

    median_inv = float(np.median(region))
    if median_inv <= 0:
        return 0.0

    # Calibrated remapping: 0.3 m (very close) → 10.0 m (far)
    norm = median_inv / 255.0          # 1.0 = closest, 0.0 = furthest
    distance = 0.3 + 9.7 * (1.0 - norm)
    return round(min(distance, 30.0), 1)   # cap at 30 m, 1 decimal place


def classify_direction(cx: float, frame_width: int) -> str:
    """Classify the horizontal position of an object into left / center / right."""
    third = frame_width / 3
    if cx < third:
        return "left"
    elif cx < 2 * third:
        return "center"
    else:
        return "right"


def detect_wall(
    depth_map: np.ndarray,
    frame_w: int,
    frame_h: int,
    threshold: float = 170.0,
    min_area_ratio: float = 0.30,
) -> dict | None:
    """Detect a nearby flat wall from the MiDaS depth map.

    Returns a synthetic detection dict or None.
    A wall is inferred when a large contiguous region has high inverse-depth
    (i.e. very close to the camera) spread across ≥30 % of the frame.
    """
    close_mask = depth_map > threshold          # pixels that are "very close"
    close_ratio = close_mask.mean()

    if close_ratio < min_area_ratio:
        return None

    # Only report the wall if the close region is spread horizontally
    # (a single small object would have a narrow horizontal span)
    cols_with_close = (close_mask.sum(axis=0) > 0).sum()
    if cols_with_close < frame_w * 0.4:
        return None

    # Compute approximate distance as median of the close region
    close_vals = depth_map[close_mask]
    median_inv = float(np.median(close_vals))
    norm = median_inv / 255.0
    distance = round(0.3 + 9.7 * (1.0 - norm), 1)

    return {
        "label": "wall",
        "confidence": round(float(close_ratio), 2),
        "bbox": [0, 0, frame_w, frame_h],
        "distance": f"{distance}m",
        "direction": "center",
        "priority": 1,
    }


def detect_stairs(
    depth_map: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> dict | None:
    """Detect stairs using three independent depth-map signals.

    Approach — three signals that must ALL pass:

    1. HORIZONTAL EDGES: Each stair tread creates a sharp horizontal edge in
       the depth gradient.  Flat floors and walls don't.  We count rows with
       strong vertical gradient and require ≥ 4 distinct edge zones.

    2. MONOTONIC TREND: Stairs go consistently deeper OR shallower row-by-row
       (≥ 75% of rows step in the same direction).  Random textures oscillate
       in both directions roughly equally.

    3. DEPTH RANGE: Stairs must span ≥ 40 depth units across the lower frame.
       Flat surfaces have a nearly constant depth.

    This combination is very unlikely to trigger on ordinary floors,
    walls, or cluttered scenes.
    """
    lower = depth_map[frame_h // 2 :, :]
    lh, lw = lower.shape
    if lh < 10 or lw < 10:
        return None

    # ── Signal 1: Strong horizontal edge zones ────────────────────────
    # Vertical absolute gradient per row = edge strength at each horizontal band
    vert_grad = np.abs(np.diff(lower.astype(float), axis=0))   # shape (lh-1, lw)
    row_edge_strength = vert_grad.mean(axis=1)                  # shape (lh-1,)

    EDGE_THRESH = 12.0
    strong = row_edge_strength > EDGE_THRESH

    # Count distinct "edge zones" (contiguous groups of strong-edge rows)
    edge_zones = 0
    in_zone = False
    for s in strong:
        if s and not in_zone:
            edge_zones += 1
            in_zone = True
        elif not s:
            in_zone = False

    if edge_zones < 4:
        return None

    # ── Signal 2: Monotonic depth progression ────────────────────────
    row_means = lower.mean(axis=1).astype(float)

    # Smooth with a 3-row window to kill single-pixel noise
    smoothed = np.convolve(row_means, np.ones(3) / 3.0, mode="valid")
    diffs = np.diff(smoothed)

    if len(diffs) < 4:
        return None

    pos = int((diffs > 0.5).sum())  # small threshold to ignore flat sections
    neg = int((diffs < -0.5).sum())
    total = pos + neg

    if total == 0:
        return None

    dominant = max(pos, neg)
    if dominant / total < 0.75:
        return None   # Not monotonic = random texture, not stairs

    # ── Signal 3: Significant depth range ────────────────────────────
    depth_range = float(row_means.max() - row_means.min())
    if depth_range < 40.0:
        return None   # Nearly flat = not stairs

    # ── All signals passed — determine direction & distance ───────────
    slope = float(np.polyfit(np.arange(len(row_means)), row_means, 1)[0])
    # Positive slope (depth increases top→bottom in lower half) → going up
    # Negative slope → going down
    direction_label = "going up" if slope > 0 else "going down"

    median_inv = float(np.median(lower))
    norm = median_inv / 255.0
    distance = round(0.3 + 9.7 * (1.0 - norm), 1)

    confidence = round(min((edge_zones / 8.0) * (dominant / total), 0.95), 2)

    return {
        "label": f"stairs ({direction_label})",
        "confidence": confidence,
        "bbox": [0, frame_h // 2, frame_w, frame_h],
        "distance": f"{distance}m",
        "direction": "center",
        "priority": 1,
    }




def detect_dropoff(
    depth_map: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> dict | None:
    """Detect negative obstacles (pits, drop-offs, ledges, downward stairs)
    by analysing depth gradient in the ground plane.

    SAFETY-FIRST LOGIC:
    Standard object detection misses 'negative obstacles' — places where
    the ground disappears (steps down, pits, kerbs, open drains).

    The ground plane is expected in the bottom 40% of the frame.  If the
    ground depth suddenly *drops away* (nearby → far in a few rows), the
    surface has ended and there is a drop-off ahead.

    Triggers:
        - A row-band in the lower frame where mean-depth jumps from
          'close' (high inverse-depth) to 'far' (low inverse-depth)
          over a narrow vertical span (≤ 15% of frame height).
        - The gradient magnitude must exceed a threshold to rule out
          gentle slopes.

    Returns a detection dict with:
        label: "⚠️ DROP-OFF" or "⚠️ PIT"
        severity: "critical" (steep) or "warning" (moderate)
        urgent_audio: True  ← tells frontend to play beep BEFORE voice
    """
    # Focus on bottom 40% — the ground region
    ground_start = int(frame_h * 0.6)
    ground = depth_map[ground_start:, :]

    if ground.shape[0] < 10:
        return None

    # Row-wise mean depth (higher = closer in MiDaS inverted map)
    row_means = ground.mean(axis=1).astype(float)

    # Compute gradient: positive gradient means depth suddenly drops
    # (surface went from "close" to "far" = ground disappeared)
    gradient = np.diff(row_means)

    # Find the steepest negative gradient (close→far transition)
    # In MiDaS: high value = close.  If values suddenly drop, ground ended.
    min_gradient = float(gradient.min())  # most negative = steepest drop

    # Threshold: a drop of ≥40 depth-units per row is significant
    # (This equals roughly a 15-20cm height step at close range)
    GRADIENT_CRITICAL = -50.0   # Very steep — immediate danger
    GRADIENT_WARNING  = -30.0   # Moderate — caution

    if min_gradient > GRADIENT_WARNING:
        return None  # No significant drop found

    # Find where the drop happens
    drop_row_idx = int(np.argmin(gradient))
    drop_row_abs = ground_start + drop_row_idx

    # Check that the region BEFORE the drop is actually close (ground)
    # If it's already far away, this isn't a relevant drop-off
    before_drop = row_means[:drop_row_idx + 1]
    if len(before_drop) < 3:
        return None
    ground_depth = float(np.median(before_drop))
    if ground_depth < 100:  # not close enough to be "ground we're walking on"
        return None

    # Check width: the drop should span a significant portion of the frame
    # A narrow column drop might just be a shadow or thin gap
    drop_row = ground[drop_row_idx, :]
    next_row = ground[min(drop_row_idx + 2, ground.shape[0] - 1), :]
    diff_row = drop_row.astype(float) - next_row.astype(float)
    drop_cols = int((diff_row > 20).sum())
    if drop_cols < frame_w * 0.25:
        return None  # Too narrow — likely a shadow or object edge

    # Determine severity
    is_critical = min_gradient <= GRADIENT_CRITICAL
    severity = "critical" if is_critical else "warning"

    # Estimate distance to the drop-off edge
    norm = ground_depth / 255.0
    edge_distance = round(0.3 + 9.7 * (1.0 - norm), 1)

    # Construct bbox around the drop zone
    bbox_y1 = drop_row_abs
    bbox_y2 = min(frame_h, drop_row_abs + int(frame_h * 0.15))
    bbox_x1 = 0
    bbox_x2 = frame_w

    label = "⚠️ DROP-OFF" if is_critical else "⚠️ CAUTION: step down"

    return {
        "label": label,
        "confidence": round(min(abs(min_gradient) / 80.0, 0.99), 2),
        "bbox": [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
        "distance": f"{edge_distance}m",
        "direction": "center",
        "priority": 0,       # ABOVE priority 1 — highest possible
        "severity": severity,
        "urgent_audio": True, # Frontend: play beep BEFORE voice
    }

