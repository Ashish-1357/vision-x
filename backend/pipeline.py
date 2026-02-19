"""
Serial AI inference pipeline for AI Vision Guide.

Execution order (strictly sequential, never parallel):
1. Blur check         – Laplacian-variance gate.
2. YOLOv8n            – Object detection on every accepted frame.
3. MiDaS-small        – Depth estimation only when objects found OR every 5th frame.
4. EasyOCR            – Text spotting only when a "text-like" class is detected.
5. Post-processing    – Merge depth + direction into per-detection JSON.
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
    midas_depth_at_bbox,
    triangle_similarity_distance,
)

# ── Configuration ───────────────────────────────────────────────────────
BLUR_THRESHOLD = 50.0          # Laplacian-variance below this → skip (lowered for phone)
MIDAS_EVERY_N = 10             # Run MiDaS at least once every N frames (was 5)
YOLO_CONF = 0.25               # YOLO confidence threshold (lowered for phone cam)
YOLO_INPUT_SIZE = 416          # Resize for YOLO inference (416 = good balance)
MIDAS_INPUT_SIZE = 256         # Resize for MiDaS inference

# COCO classes that should trigger EasyOCR text spotting
TEXT_TRIGGER_CLASSES: set[str] = {"book", "cell phone", "tv", "laptop", "stop sign"}

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


def _is_blurry(gray: np.ndarray) -> bool:
    """Return True when the frame is too blurry for useful inference."""
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < BLUR_THRESHOLD


def analyze_frame(frame_bytes: bytes) -> dict[str, Any]:
    """Run the full serial pipeline on a single JPEG frame.

    Returns a dict ready for JSON serialisation:
    {
      "skipped": bool,
      "reason": str | None,
      "detections": [ {label, confidence, bbox, distance, direction} ],
      "text": str | None,
      "latency_ms": float,
    }
    """
    global _frame_count
    t0 = time.perf_counter()

    # Decode JPEG
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"skipped": True, "reason": "decode_error", "detections": [], "text": None, "latency_ms": 0}

    frame_h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Step 1: Blur check ──────────────────────────────────────────
    if _is_blurry(gray):
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return {"skipped": True, "reason": "blurry", "detections": [], "text": None, "latency_ms": ms}

    with _lock:
        _frame_count += 1
        current_frame = _frame_count

    # ── Step 2: YOLOv8 detection ────────────────────────────────────
    model = _load_yolo()
    results = model.predict(
        frame,
        imgsz=YOLO_INPUT_SIZE,
        conf=YOLO_CONF,
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

    # ── Step 3: MiDaS depth (conditional) ───────────────────────────
    run_midas = len(detections) > 0 or (current_frame % MIDAS_EVERY_N == 0)
    depth_map: np.ndarray | None = None

    if run_midas:
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
            # Normalise to 0-255
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Refine distance for each detection using depth map
            for det in detections:
                bbox_tuple = tuple(det["bbox"])
                midas_dist = midas_depth_at_bbox(depth_map, bbox_tuple)
                if midas_dist > 0:
                    det["distance"] = f"{midas_dist}m"
        except Exception:
            pass  # Graceful fallback — triangle-similarity distances remain

    # ── Step 4: EasyOCR (conditional) ───────────────────────────────
    spotted_text: str | None = None
    if text_trigger:
        try:
            reader = _load_ocr()
            ocr_results = reader.readtext(gray, detail=0, paragraph=True)
            if ocr_results:
                spotted_text = " ".join(ocr_results).strip()
        except Exception:
            pass

    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "skipped": False,
        "reason": None,
        "detections": detections,
        "text": spotted_text,
        "latency_ms": ms,
    }
