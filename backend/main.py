"""
FastAPI backend for AI Vision Guide.

Endpoints
---------
POST /analyze   – Accept a JPEG frame, run serial AI pipeline, return JSON.
GET  /health    – Liveness probe.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from pipeline import analyze_frame

# Single-thread executor — guarantees models never run in parallel
_executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up models on startup so the first request isn't slow."""
    print("🔄  Pre-loading models (this may take a minute on first run)…")
    loop = asyncio.get_event_loop()
    # A tiny 1×1 black JPEG to trigger lazy loading
    import struct, io
    import numpy as np
    import cv2

    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy)
    await loop.run_in_executor(_executor, analyze_frame, buf.tobytes())
    print("✅  Models ready.")
    yield
    _executor.shutdown(wait=False)


app = FastAPI(
    title="AI Vision Guide",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/proxy-camera")
async def proxy_camera(
    url: str = Query(..., description="Phone IP camera snapshot URL"),
    rotation: int = Query(0, description="Manual rotation: 0, 90, 180, 270"),
):
    """Proxy a snapshot — corrects rotation for preview (non-blocking)."""
    loop = asyncio.get_event_loop()

    def fetch_and_correct():
        raw, error = _fetch_phone_image(url)
        if error:
            return None, error
        return _correct_phone_image(raw, max_dim=960, force_rotation=rotation), None

    data, error = await loop.run_in_executor(None, fetch_and_correct)
    if error:
        return JSONResponse(status_code=502, content={"error": error})
    return Response(content=data, media_type="image/jpeg")


# ── Phone-camera helpers (cached, fast) ─────────────────────────────

import ssl as _ssl
_ssl_ctx = _ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = _ssl.CERT_NONE


def _fetch_phone_image(url: str) -> tuple[bytes | None, str | None]:
    """Fetch a JPEG from the phone camera. Returns (data, error)."""
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5, context=_ssl_ctx) as resp:
            return resp.read(), None
    except urllib.error.URLError as e:
        return None, f"Cannot reach phone: {e.reason}"
    except Exception as e:
        return None, f"Fetch error: {e}"


def _correct_phone_image(jpeg_bytes: bytes, max_dim: int = 640, force_rotation: int = 0) -> bytes:
    """Apply EXIF rotation + optional forced rotation + resize.

    Many phone camera apps strip EXIF orientation data.  When that happens,
    the user can set force_rotation (90, 180, 270) from the UI Rotate button.
    """
    import io
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(jpeg_bytes))

    # Try EXIF orientation first
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Apply user-controlled forced rotation
    if force_rotation == 90:
        img = img.rotate(-90, expand=True)
    elif force_rotation == 180:
        img = img.rotate(180, expand=True)
    elif force_rotation == 270:
        img = img.rotate(-270, expand=True)

    # Resize if too large
    if max_dim and max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


@app.get("/analyze-phone")
async def analyze_phone(
    url: str = Query(..., description="Phone IP camera snapshot URL"),
    rotation: int = Query(0, description="Manual rotation: 0, 90, 180, 270"),
):
    """Fetch a snapshot from the phone, correct rotation, resize, and analyze."""
    loop = asyncio.get_event_loop()

    def fetch_and_correct():
        raw, error = _fetch_phone_image(url)
        if error:
            return None, error
        return _correct_phone_image(raw, max_dim=640, force_rotation=rotation), None

    data, error = await loop.run_in_executor(None, fetch_and_correct)
    if error:
        return JSONResponse(status_code=502, content={"error": error})

    result = await loop.run_in_executor(_executor, analyze_frame, data)
    return JSONResponse(content=result)


@app.post("/analyze")
async def analyze(frame: UploadFile = File(...)):
    """Receive a JPEG frame, run the serial AI pipeline, return results."""
    data = await frame.read()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, analyze_frame, data)
    return JSONResponse(content=result)
