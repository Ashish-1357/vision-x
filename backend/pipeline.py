"""
Serial AI inference pipeline for AI Vision Guide.

Execution order (strictly sequential, never parallel):
1. Blur check         – Laplacian-variance gate.
2. YOLOv8n            – Object detection on every accepted frame.
3. MiDaS-small        – Depth estimation on EVERY frame for accuracy.
4. Wall detection     – Heuristic depth-map analysis (priority 1).
5. Stair detection    – Depth gradient analysis (priority 1).
6. EasyOCR            – Text spotting: trigger classes + every 10th frame.
7. Post-processing    – Merge depth + direction into per-detection JSON.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np
import torch

from depth_utils import (
    classify_direction,
    detect_dropoff,
    detect_wall,
    midas_depth_at_bbox,
    triangle_similarity_distance,
)

import base64
import json
import ssl
import urllib.error
import urllib.request

# ── Configuration ───────────────────────────────────────────────────────
BLUR_THRESHOLD = 50.0          # Laplacian-variance below this → skip (lowered for phone)
MIDAS_EVERY_N = 1              # Run MiDaS every frame for accurate, fresh depth
YOLO_CONF = 0.25               # YOLO confidence threshold (lowered for phone cam)
YOLO_INPUT_SIZE = 416          # Resize for YOLO inference (416 = good balance)
MIDAS_INPUT_SIZE = 256         # Resize for MiDaS inference
OCR_EVERY_N = 3                # Run OCR unconditionally every N frames for signs/billboards

# ── Roboflow Stairs v1 model ─────────────────────────────────────────────
# Trained custom model — much more accurate than depth-map heuristics
RF_API_KEY      = "rf_kP1JY2XRmqVYR6Pbc5gJvVQh4I2"
RF_STAIR_MODEL  = "stairs-aj4wy-8jfdf"
RF_STAIR_VER    = "1"
RF_TIMEOUT      = 2            # Seconds before we give up (don't slow the pipeline)

# COCO classes that should trigger EasyOCR text spotting
TEXT_TRIGGER_CLASSES: set[str] = {
    "book", "cell phone", "tv", "laptop", "stop sign",
    "bench", "sign", "poster",  # extended for billboards/notice boards
}

# ── Lazy-loaded singletons ──────────────────────────────────────────────
_lock = threading.Lock()
_yolo_model = None
_midas_model = None
_midas_transform = None
_ocr_reader = None


def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def _load_midas():
    global _midas_model, _midas_transform
    if _midas_model is None:
        _midas_model = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small",
            trust_repo="check",
        )
        _midas_model.eval()
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo="check",
        )
        _midas_transform = midas_transforms.small_transform
    return _midas_model, _midas_transform


def _load_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


# ── Frame counter (module-level) ────────────────────────────────────────
_frame_count = 0

# ── Crowd-safety state (module-level) ───────────────────────────────────
# Last-Known-Safe-State: cached after every successfully analysed frame.
# Replayed when ≥ BLUR_REPEAT_AFTER consecutive blurry frames are received.
_last_safe_state: dict[str, Any] | None = None
_consecutive_blurry: int = 0
BLUR_REPEAT_AFTER = 3          # Replay buffer after this many blurs in a row

# Dynamic threshold tiers (Laplacian variance → YOLO confidence)
# High motion (low variance) → lower confidence → catch more obstacles
BLUR_HIGH_MOTION_THRESHOLD = 80.0   # Below this: high motion, lower YOLO conf
YOLO_CONF_HIGH_MOTION = 0.15        # Aggressive detection during fast movement
YOLO_CONF_NORMAL     = YOLO_CONF    # 0.25 — standard


def _detect_stairs_roboflow(
    frame_bgr: np.ndarray,
    frame_w: int,
    frame_h: int,
    depth_map: "np.ndarray | None" = None,
) -> list[dict[str, Any]]:
    """Call Roboflow Stairs v1 model and return detection dicts.

    Sends the frame as a base64-encoded JPEG to the Roboflow hosted API.
    Times out after RF_TIMEOUT seconds so it never blocks the pipeline.
    Falls back to an empty list on any error (network down, quota, etc.).
    """
    try:
        # Encode frame as JPEG base64 (70% quality is plenty for detection)
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        url = (
            f"https://detect.roboflow.com/{RF_STAIR_MODEL}/{RF_STAIR_VER}"
            f"?api_key={RF_API_KEY}"
        )
        req = urllib.request.Request(
            url,
            data=b64.encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=RF_TIMEOUT) as resp:
            payload = json.loads(resp.read())

        detections: list[dict[str, Any]] = []
        for pred in payload.get("predictions", []):
            px, py = pred["x"], pred["y"]
            pw, ph = pred["width"], pred["height"]
            x1, y1 = int(px - pw / 2), int(py - ph / 2)
            x2, y2 = int(px + pw / 2), int(py + ph / 2)
            conf = round(float(pred["confidence"]), 2)

            # Class name from model — classes are "stairs" and "ramp"
            cls = pred.get("class", "stairs").lower()
            if "ramp" in cls:
                label = "ramp ahead"
            else:
                label = "stairs ahead"

            cx = (x1 + x2) / 2
            direction = classify_direction(cx, frame_w)

            # Use depth map for distance if available
            dist_str: str | None = None
            if depth_map is not None:
                d = midas_depth_at_bbox(depth_map, (x1, y1, x2, y2))
                if d > 0:
                    dist_str = f"{d}m"

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "distance": dist_str,
                "direction": direction,
                "priority": 1,
            })

        return detections

    except Exception as exc:
        # Never crash the pipeline — log quietly and return nothing
        print(f"[Roboflow Stairs] {type(exc).__name__}: {exc}")
        return []


def _blur_score(gray: np.ndarray) -> float:
    """Return Laplacian variance — lower = more blurry / more motion."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _is_blurry(score: float) -> bool:
    return score < BLUR_THRESHOLD


def analyze_frame(frame_bytes: bytes) -> dict[str, Any]:
    """Run the full serial pipeline on a single JPEG frame.

    Returns a dict ready for JSON serialisation:
    {
      "skipped": bool,
      "reason": str | None,
      "detections": [ {label, confidence, bbox, distance, direction, priority?} ],
      "text": str | None,
      "urgent_alert": bool,
      "low_visibility": bool,       # True when replaying buffered state
      "blur_score": float,          # Laplacian variance of the frame
      "latency_ms": float,
    }
    """
    global _frame_count, _last_safe_state, _consecutive_blurry
    t0 = time.perf_counter()

    # Decode JPEG
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"skipped": True, "reason": "decode_error", "detections": [], "text": None,
                "urgent_alert": False, "low_visibility": False, "blur_score": 0, "latency_ms": 0}

    frame_h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Step 1: Blur check + Frame Buffering ────────────────────────
    score = _blur_score(gray)

    if _is_blurry(score):
        _consecutive_blurry += 1
        ms = round((time.perf_counter() - t0) * 1000, 1)

        # If we have a cached safe state AND enough consecutive blurs,
        # REPLAY it with low_visibility=True + caution tone on frontend
        if _consecutive_blurry >= BLUR_REPEAT_AFTER and _last_safe_state is not None:
            buffered = dict(_last_safe_state)
            buffered["low_visibility"] = True
            buffered["blur_score"] = score
            buffered["latency_ms"] = ms
            buffered["skipped"] = False
            buffered["reason"] = "buffered_low_visibility"
            return buffered

        # Not enough consecutive blurs yet (or no buffer) — skip as usual
        return {"skipped": True, "reason": "blurry", "detections": [], "text": None,
                "urgent_alert": False, "low_visibility": False, "blur_score": score, "latency_ms": ms}

    # Frame is clear — reset consecutive counter
    _consecutive_blurry = 0

    # ── Dynamic YOLO confidence threshold (Feature 2) ───────────────
    # High-motion frames (moderate blur above skip threshold but below 80)
    # → lower confidence → catch more obstacles (false-positive-safe policy)
    yolo_conf = YOLO_CONF_HIGH_MOTION if score < BLUR_HIGH_MOTION_THRESHOLD else YOLO_CONF_NORMAL

    with _lock:
        _frame_count += 1
        current_frame = _frame_count


    # ── Step 2: YOLOv8 detection ────────────────────────────────────
    model = _load_yolo()
    results = model.predict(
        frame,
        imgsz=YOLO_INPUT_SIZE,
        conf=yolo_conf,       # Dynamic: lower when high motion detected
        verbose=False,
    )

    detections: list[dict[str, Any]] = []
    text_trigger = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = model.names[int(box.cls[0])]
            conf = round(float(box.conf[0]), 2)
            cx = (x1 + x2) / 2
            direction = classify_direction(cx, frame_w)

            det: dict[str, Any] = {
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "distance": None,
                "direction": direction,
            }

            # Triangle-similarity distance (fast fallback)
            tri_dist = triangle_similarity_distance(label, float(x2 - x1))
            if tri_dist is not None:
                det["distance"] = f"{tri_dist}m"

            detections.append(det)

            if label in TEXT_TRIGGER_CLASSES:
                text_trigger = True

    # ── Step 3: MiDaS depth (every frame for accuracy) ──────────────
    depth_map: np.ndarray | None = None

    try:
        midas, transform = _load_midas()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(rgb)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(frame_h, frame_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        # Normalise to 0-255 (higher = closer)
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Refine distance for each YOLO detection using the depth map
        for det in detections:
            bbox_tuple = tuple(det["bbox"])
            midas_dist = midas_depth_at_bbox(depth_map, bbox_tuple)
            if midas_dist > 0:
                det["distance"] = f"{midas_dist}m"

    except Exception:
        pass  # Graceful fallback — triangle-similarity distances remain

    # ── Step 4: Wall detection (priority 1, depth required) ─────────
    if depth_map is not None:
        wall_det = detect_wall(depth_map, frame_w, frame_h)
        if wall_det is not None:
            # Insert at front so it's priority 1
            detections.insert(0, wall_det)

    # ── Step 5: Stair detection ─ Roboflow Stairs v1 (custom-trained model) ──
    #   Replaces the depth-oscillation heuristic with a real trained detector.
    #   Runs as a fast hosted API call (2s timeout).  Distance refined by MiDaS.
    stair_dets = _detect_stairs_roboflow(frame, frame_w, frame_h, depth_map)
    for sd in stair_dets:
        insert_idx = 1 if (detections and detections[0].get("label") == "wall") else 0
        detections.insert(insert_idx, sd)

    # ── Step 5.5: DROP-OFF detection (priority 0 — HIGHEST) ─────────
    #   Safety-first: detect where the ground disappears (pits, ledges,
    #   downward stairs).  Returns urgent_audio=True → frontend plays
    #   a high-pitched beep BEFORE the voice alert even finishes.
    urgent_alert = False
    if depth_map is not None:
        dropoff_det = detect_dropoff(depth_map, frame_w, frame_h)
        if dropoff_det is not None:
            urgent_alert = True
            detections.insert(0, dropoff_det)  # Always first

    # ── Step 6: EasyOCR — ALWAYS run for signboards/billboards ────────
    spotted_text: str | None = None
    try:
        reader = _load_ocr()
        ocr_input = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        ocr_input = cv2.filter2D(ocr_input, -1, kernel)
        ocr_results = reader.readtext(ocr_input, detail=0, paragraph=True)
        if ocr_results:
            spotted_text = " ".join(ocr_results).strip()
    except Exception as e:
        print(f"[OCR ERROR] {e}")

    ms = round((time.perf_counter() - t0) * 1000, 1)
    result = {
        "skipped": False,
        "reason": None,
        "detections": detections,
        "text": spotted_text,
        "urgent_alert": urgent_alert,
        "low_visibility": False,
        "blur_score": round(score, 1),
        "latency_ms": ms,
    }

    # ── Save Last-Known-Safe-State ───────────────────────────────────────
    # Cache a COPY before returning so blurry replay doesn't mutate it
    _last_safe_state = {
        "skipped": False,
        "reason": None,
        "detections": detections,
        "text": spotted_text,
        "urgent_alert": urgent_alert,
        "low_visibility": False,
        "blur_score": round(score, 1),
        "latency_ms": ms,
    }

    return result
