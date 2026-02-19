/* ================================================================
   AI Vision Guide — Frontend Logic
   Webcam / IP-camera capture → POST /analyze → overlay + voice
   ================================================================ */

const API_BASE = "http://localhost:8000";
const API_URL = API_BASE + "/analyze";
const PROXY_URL = API_BASE + "/proxy-camera";
const FPS = 15;
const FRAME_INTERVAL = Math.round(1000 / FPS);

// ── DOM refs ──────────────────────────────────────────────────────
const video = document.getElementById("camera");
const overlayCanvas = document.getElementById("overlay-canvas");
const ctx = overlayCanvas.getContext("2d");

const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnMute = document.getElementById("btn-mute");
const muteLabel = document.getElementById("mute-label");
const iconUnmuted = document.getElementById("icon-unmuted");
const iconMuted = document.getElementById("icon-muted");
const statusBadge = document.getElementById("status-badge");
const statusText = document.getElementById("status-text");
const detList = document.getElementById("detections-list");

const statLatency = document.getElementById("stat-latency");
const statFrames = document.getElementById("stat-frames");
const statSkipped = document.getElementById("stat-skipped");
const statObjects = document.getElementById("stat-objects");

// Camera source DOM refs
const srcWebcam = document.getElementById("src-webcam");
const srcPhone = document.getElementById("src-phone");
const ipInputRow = document.getElementById("ip-input-row");
const ipHint = document.getElementById("ip-hint");
const ipUrlInput = document.getElementById("ip-url");
const btnTestUrl = document.getElementById("btn-test-url");
const btnRotate = document.getElementById("btn-rotate");
const rotateLabel = document.getElementById("rotate-label");

// ── State ─────────────────────────────────────────────────────────
let running = false;
let muted = false;
let frameCount = 0;
let skipCount = 0;
let captureCanvas = null;
let captureCtx = null;
let sendingFrame = false;

// Camera source: "webcam" or "ip"
let cameraSource = "webcam";

// Manual rotation for phone camera (0, 90, 180, 270)
let phoneRotation = 0;

// Speech debounce: remember last spoken alert to avoid repeat
let lastSpoken = "";
let lastSpokeAt = 0;
const SPEECH_COOLDOWN_MS = 2500;

// ── Icon map for common labels ────────────────────────────────────
const ICON_MAP = {
  person: "🧑",
  chair: "🪑",
  car: "🚗",
  bicycle: "🚲",
  dog: "🐕",
  cat: "🐈",
  bottle: "🍾",
  cup: "☕",
  laptop: "💻",
  tv: "📺",
  couch: "🛋️",
  bed: "🛏️",
  "dining table": "🍽️",
  suitcase: "🧳",
  backpack: "🎒",
  umbrella: "☂️",
  book: "📖",
  "cell phone": "📱",
  "potted plant": "🌿",
  bus: "🚌",
  truck: "🚛",
  motorcycle: "🏍️",
  "fire hydrant": "🧯",
  "stop sign": "🛑",
  "traffic light": "🚦",
  default: "⚠️",
};

// ── Camera source toggle ──────────────────────────────────────────
srcWebcam.addEventListener("click", () => {
  if (running) return;  // don't switch while scanning
  cameraSource = "webcam";
  srcWebcam.classList.add("src-btn--active");
  srcPhone.classList.remove("src-btn--active");
  ipInputRow.style.display = "none";
  ipHint.style.display = "none";
  video.style.display = "";
});

srcPhone.addEventListener("click", () => {
  if (running) return;
  cameraSource = "ip";
  srcPhone.classList.add("src-btn--active");
  srcWebcam.classList.remove("src-btn--active");
  ipInputRow.style.display = "flex";
  ipHint.style.display = "block";
  video.style.display = "";
});

// ── Test IP camera URL ────────────────────────────────────────────
/**
 * Auto-fix common URL issues:
 * - If the user pastes just "http://192.168.1.5:8080", append "/shot.jpg"
 * - Ensure the URL starts with http://
 */
function normalizeIpUrl(raw) {
  let url = raw.trim();
  // Add protocol if missing
  if (!/^https?:\/\//i.test(url)) url = "http://" + url;
  // If URL ends with just a port (no path or only /), add /shot.jpg
  if (/:\d+\/?$/.test(url)) {
    url = url.replace(/\/$/, "") + "/shot.jpg";
  }
  return url;
}

btnTestUrl.addEventListener("click", async () => {
  let url = ipUrlInput.value.trim();
  if (!url) { alert("Please enter a URL first."); return; }

  // Auto-fix the URL
  url = normalizeIpUrl(url);
  ipUrlInput.value = url;  // show the corrected URL to the user

  // Remove any old status
  const oldStatus = document.querySelector(".ip-status");
  if (oldStatus) oldStatus.remove();

  const statusEl = document.createElement("span");
  statusEl.className = "ip-status";
  statusEl.textContent = "⏳ Testing…";
  ipInputRow.appendChild(statusEl);

  try {
    const proxyUrl = PROXY_URL + "?url=" + encodeURIComponent(url);
    console.log("Testing IP camera via proxy:", proxyUrl);
    const res = await fetch(proxyUrl);

    if (!res.ok) {
      // Try to read error message from backend
      let errMsg = "HTTP " + res.status;
      try {
        const errData = await res.json();
        errMsg = errData.error || errMsg;
      } catch { }
      throw new Error(errMsg);
    }

    const blob = await res.blob();
    console.log("Got response:", blob.type, blob.size, "bytes");

    if (blob.size < 100) throw new Error("Response too small — not a valid image");

    statusEl.textContent = "✅ Connected! (" + Math.round(blob.size / 1024) + " KB)";
    statusEl.className = "ip-status ip-status--ok";
  } catch (err) {
    console.error("IP camera test failed:", err);
    statusEl.textContent = "❌ " + (err.message || "Connection failed. Check URL & Wi-Fi.");
    statusEl.className = "ip-status ip-status--fail";
  }
});

// ── Webcam start / stop ───────────────────────────────────────────
async function startWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  overlayCanvas.width = video.videoWidth;
  overlayCanvas.height = video.videoHeight;
  captureCanvas = document.createElement("canvas");
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  captureCtx = captureCanvas.getContext("2d");
}

function stopWebcam() {
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
}

// ── IP camera helpers ─────────────────────────────────────────────
let ipPreviewCanvas = null;
let ipPreviewCtx = null;

/**
 * Fetch a single JPEG from phone via backend proxy.
 * Returns { blob, bitmap } — blob for sending to /analyze, bitmap for display.
 */
async function fetchIpFrame(url) {
  const proxyUrl = PROXY_URL + "?url=" + encodeURIComponent(url) + "&rotation=" + phoneRotation;
  const res = await fetch(proxyUrl);
  if (!res.ok) {
    let errMsg = "Proxy error " + res.status;
    try { const j = await res.json(); errMsg = j.error || errMsg; } catch { }
    throw new Error(errMsg);
  }
  const blob = await res.blob();
  const bitmap = await createImageBitmap(blob);
  return { blob, bitmap };
}

/**
 * Set up the IP camera preview canvas (called once on Start).
 */
async function startIpCamera() {
  const url = normalizeIpUrl(ipUrlInput.value.trim());
  ipUrlInput.value = url;
  if (!url) throw new Error("No IP camera URL provided.");

  // Fetch first frame to determine dimensions
  const { blob, bitmap } = await fetchIpFrame(url);
  const w = bitmap.width;
  const h = bitmap.height;

  overlayCanvas.width = w;
  overlayCanvas.height = h;

  // Create visible preview canvas
  ipPreviewCanvas = document.createElement("canvas");
  ipPreviewCanvas.id = "ip-preview";
  ipPreviewCanvas.width = w;
  ipPreviewCanvas.height = h;
  ipPreviewCanvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;";
  ipPreviewCtx = ipPreviewCanvas.getContext("2d");

  // Draw first frame
  ipPreviewCtx.drawImage(bitmap, 0, 0);
  bitmap.close();

  // Insert into DOM
  video.style.display = "none";
  const container = document.getElementById("video-container");
  const old = document.getElementById("ip-preview");
  if (old) old.remove();
  container.insertBefore(ipPreviewCanvas, overlayCanvas);

  // Set up capture canvas for webcam compat
  captureCanvas = document.createElement("canvas");
  captureCanvas.width = w;
  captureCanvas.height = h;
  captureCtx = captureCanvas.getContext("2d");
}

function stopIpCamera() {
  const preview = document.getElementById("ip-preview");
  if (preview) preview.remove();
  ipPreviewCanvas = null;
  ipPreviewCtx = null;
  video.style.display = "";
}

// ── Frame capture for webcam mode ─────────────────────────────────
async function captureWebcamFrame() {
  captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  return new Promise(resolve => captureCanvas.toBlob(resolve, "image/jpeg", 0.7));
}

async function sendFrame(blob) {
  const t0 = performance.now();
  const form = new FormData();
  form.append("frame", blob, "frame.jpg");
  const res = await fetch(API_URL, { method: "POST", body: form });
  const data = await res.json();
  const rtt = Math.round(performance.now() - t0);
  return { ...data, rtt };
}

// ── Analysis loops ────────────────────────────────────────────────
let loopTimer = null;
let analyzing = false;

/**
 * Webcam analysis loop — works as before.
 */
function startWebcamLoop() {
  loopTimer = setInterval(async () => {
    if (sendingFrame) return;
    sendingFrame = true;
    try {
      const blob = await captureWebcamFrame();
      const result = await sendFrame(blob);
      frameCount++;
      if (result.skipped) skipCount++;
      updateStats(result);
      drawOverlay(result.detections || []);
      renderDetections(result);
      speak(result);
    } catch (err) {
      console.warn("Frame error:", err);
    } finally {
      sendingFrame = false;
    }
  }, FRAME_INTERVAL);
}

/**
 * UNIFIED IP camera loop — ONE fetch per cycle does EVERYTHING:
 *  1. Fetch frame from phone (via proxy — single phone request)
 *  2. Display it as live preview
 *  3. Send SAME frame to /analyze (no second phone fetch!)
 *  4. Update detections + voice
 *
 * This eliminates the contention between preview and analysis loops.
 */
function startIpAnalysisLoop() {
  const url = normalizeIpUrl(ipUrlInput.value.trim());
  analyzing = true;
  let errorCount = 0;

  async function tick() {
    if (!analyzing) return;
    const t0 = performance.now();
    try {
      // Step 1: Fetch ONE frame from phone
      const { blob, bitmap } = await fetchIpFrame(url);
      const bw = bitmap.width;
      const bh = bitmap.height;

      // Step 2: Display as preview (auto-resize if rotation changed dimensions)
      if (bw !== ipPreviewCanvas.width || bh !== ipPreviewCanvas.height) {
        ipPreviewCanvas.width = bw;
        ipPreviewCanvas.height = bh;
        overlayCanvas.width = bw;
        overlayCanvas.height = bh;
      }
      ipPreviewCtx.clearRect(0, 0, bw, bh);
      ipPreviewCtx.drawImage(bitmap, 0, 0, bw, bh);
      bitmap.close();

      // Step 3: Send SAME blob to /analyze (localhost — instant, no second phone fetch)
      const form = new FormData();
      form.append("frame", blob, "frame.jpg");
      const analysisRes = await fetch(API_URL, { method: "POST", body: form });
      const data = await analysisRes.json();
      const rtt = Math.round(performance.now() - t0);
      const result = { ...data, rtt };

      // Step 4: Update UI
      errorCount = 0;
      frameCount++;
      if (result.skipped) skipCount++;
      updateStats(result);
      drawOverlay(result.detections || []);
      renderDetections(result);
      speak(result);
    } catch (err) {
      errorCount++;
      console.error("Loop error #" + errorCount + ":", err);
      detList.innerHTML = `<p class="empty-state" style="color:#ff6b6b">⚠️ Error (attempt ${errorCount}): ${err.message || err}</p>`;
    }
    // Schedule next cycle
    if (analyzing) {
      loopTimer = setTimeout(tick, errorCount > 3 ? 2000 : 50);
    }
  }
  tick();
}

function stopLoop() {
  analyzing = false;
  clearInterval(loopTimer);
  clearTimeout(loopTimer);
  loopTimer = null;
  sendingFrame = false;
}

// ── UI updates ────────────────────────────────────────────────────
function setStatus(mode) {
  statusBadge.className = "badge badge--" + mode;
  statusText.textContent = mode === "active" ? "Scanning" : mode === "error" ? "Error" : "Idle";
}

function updateStats(result) {
  statLatency.textContent = `${result.rtt} ms`;
  statFrames.textContent = frameCount;
  statSkipped.textContent = skipCount;
  statObjects.textContent = (result.detections || []).length;
}

function renderDetections(result) {
  const dets = result.detections || [];
  if (dets.length === 0 && !result.text) {
    detList.innerHTML = '<p class="empty-state">No objects detected</p>';
    return;
  }
  let html = "";
  for (const d of dets) {
    const icon = ICON_MAP[d.label] || ICON_MAP.default;
    const dist = d.distance || "unknown";
    html += `
      <div class="det-card">
        <div class="det-icon">${icon}</div>
        <div class="det-info">
          <span class="det-label">${d.label}</span>
          <span class="det-meta">${dist} · ${d.direction} · ${Math.round(d.confidence * 100)}%</span>
        </div>
      </div>`;
  }
  if (result.text) {
    html += `
      <div class="det-card">
        <div class="det-icon">📝</div>
        <div class="det-info">
          <span class="det-label">Text spotted</span>
          <span class="det-meta">${result.text}</span>
        </div>
      </div>`;
  }
  detList.innerHTML = html;
}

// ── Canvas overlay (bounding boxes) ───────────────────────────────
const BOX_COLORS = {
  left: "#ff6b6b",
  center: "#00cec9",
  right: "#fdcb6e",
};

function drawOverlay(detections) {
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  for (const d of detections) {
    const [x1, y1, x2, y2] = d.bbox;
    const color = BOX_COLORS[d.direction] || "#a29bfe";

    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const text = `${d.label} ${d.distance || ""}`;
    ctx.font = "bold 13px Inter, sans-serif";
    const tw = ctx.measureText(text).width + 12;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x1, y1 - 22, tw, 22, [6, 6, 0, 0]);
    ctx.fill();

    ctx.fillStyle = "#000";
    ctx.fillText(text, x1 + 6, y1 - 6);
  }
}

// ── Voice guidance (Web Speech API) ───────────────────────────────
function speak(result) {
  if (muted || !window.speechSynthesis) return;
  const dets = result.detections || [];
  if (dets.length === 0 && !result.text) return;

  const parts = [];
  for (const d of dets) {
    const dist = d.distance ? `, ${d.distance} ahead` : "";
    parts.push(`${d.label}${dist}, on your ${d.direction}`);
  }
  if (result.text) {
    parts.push(`Text reads: ${result.text}`);
  }
  const msg = parts.join(". ");

  const now = Date.now();
  if (msg === lastSpoken && now - lastSpokeAt < SPEECH_COOLDOWN_MS) return;
  lastSpoken = msg;
  lastSpokeAt = now;

  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(msg);
  utter.rate = 1.15;
  utter.pitch = 1;
  window.speechSynthesis.speak(utter);
}

// ── Button handlers ───────────────────────────────────────────────
btnStart.addEventListener("click", async () => {
  try {
    // Immediate UI feedback
    detList.innerHTML = '<p class="empty-state">⏳ Connecting to camera…</p>';
    btnStart.disabled = true;

    if (cameraSource === "ip") {
      await startIpCamera();
    } else {
      await startWebcam();
    }
    running = true;
    frameCount = 0;
    skipCount = 0;
    btnStop.disabled = false;
    srcWebcam.disabled = true;
    srcPhone.disabled = true;
    setStatus("active");
    detList.innerHTML = '<p class="empty-state">⏳ Analyzing first frame…</p>';

    // Start the correct analysis loop
    if (cameraSource === "ip") {
      startIpAnalysisLoop();
    } else {
      startWebcamLoop();
    }
  } catch (err) {
    console.error("Camera error:", err);
    btnStart.disabled = false;
    setStatus("error");
    detList.innerHTML = `<p class="empty-state" style="color:#ff6b6b">❌ ${err.message || 'Connection failed'}</p>`;
    if (cameraSource === "ip") {
      alert("Could not connect to phone camera. Check the URL and Wi-Fi.\n\nError: " + err.message);
    } else {
      alert("Could not access webcam. Please grant permission.\n\nError: " + err.message);
    }
  }
});

btnStop.addEventListener("click", () => {
  running = false;
  stopLoop();
  if (cameraSource === "ip") {
    stopIpCamera();
  } else {
    stopWebcam();
  }
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  btnStart.disabled = false;
  btnStop.disabled = true;
  srcWebcam.disabled = false;
  srcPhone.disabled = false;
  setStatus("idle");
  detList.innerHTML = '<p class="empty-state">Stopped. Press <strong>Start</strong> to resume.</p>';
  window.speechSynthesis?.cancel();
});

btnMute.addEventListener("click", () => {
  muted = !muted;
  iconUnmuted.style.display = muted ? "none" : "";
  iconMuted.style.display = muted ? "" : "none";
  muteLabel.textContent = muted ? "Unmute" : "Mute";
  if (muted) window.speechSynthesis?.cancel();
});

// ── Rotate button handler ─────────────────────────────────────────
btnRotate.addEventListener("click", () => {
  phoneRotation = (phoneRotation + 90) % 360;
  rotateLabel.textContent = phoneRotation === 0 ? "Rotate" : `${phoneRotation}°`;
  console.log("Rotation set to", phoneRotation);
});
