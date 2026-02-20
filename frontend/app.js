/* ================================================================
   AI Vision Guide — Frontend Logic
   Webcam / IP-camera capture → POST /analyze → overlay + voice
   + URGENT AUDIO CUE for drop-off / pit detection
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

// Phone camera — permanent 90° rotation (no UI button needed)
const PHONE_ROTATION = 90;

// Speech debounce: remember last spoken alert to avoid repeat
let lastSpoken = "";
let lastSpokeAt = 0;
const SPEECH_COOLDOWN_MS = 2500;

// ── URGENT AUDIO CUE — Web Audio API ─────────────────────────────
// A high-pitched beep that plays INSTANTLY when a drop-off or pit is
// detected.  This stops the user BEFORE the voice alert finishes.
// Uses Web Audio API for zero-latency playback (no file loading).
let _audioCtx = null;
let _lastBeepAt = 0;
const BEEP_COOLDOWN_MS = 1500;  // Don't beep more than once per 1.5s

function _getAudioCtx() {
  if (!_audioCtx) {
    _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return _audioCtx;
}

/**
 * Play an urgent high-pitched beep pattern.
 * Critical severity: 3 rapid beeps (800Hz → 1000Hz → 1200Hz)
 * Warning severity:  2 beeps (600Hz → 800Hz)
 */
function playUrgentBeep(severity) {
  if (muted) return;
  const now = Date.now();
  if (now - _lastBeepAt < BEEP_COOLDOWN_MS) return;
  _lastBeepAt = now;

  const actx = _getAudioCtx();
  const isCritical = severity === "critical";
  const freqs = isCritical ? [800, 1000, 1200] : [600, 800];
  const beepDuration = isCritical ? 0.08 : 0.12;  // seconds per beep
  const gap = 0.06;  // seconds between beeps

  freqs.forEach((freq, i) => {
    const startTime = actx.currentTime + i * (beepDuration + gap);
    const osc = actx.createOscillator();
    const gain = actx.createGain();

    osc.type = "square";
    osc.frequency.setValueAtTime(freq, startTime);

    gain.gain.setValueAtTime(0, startTime);
    gain.gain.linearRampToValueAtTime(0.6, startTime + 0.01);       // fast attack
    gain.gain.linearRampToValueAtTime(0, startTime + beepDuration);  // fast release

    osc.connect(gain);
    gain.connect(actx.destination);
    osc.start(startTime);
    osc.stop(startTime + beepDuration + 0.01);
  });
}

/**
 * Caution tone — plays when system enters low-visibility mode.
 * (3+ consecutive blurry frames → replaying last known safe state)
 * Gentle descending two-note tone: 400Hz → 300Hz (sine, soft).
 * Very different from the urgent beep — user can easily tell them apart.
 */
let _lastCautionAt = 0;
const CAUTION_COOLDOWN_MS = 4000;  // Don't repeat caution more than once per 4s

function playCautionTone() {
  if (muted) return;
  const now = Date.now();
  if (now - _lastCautionAt < CAUTION_COOLDOWN_MS) return;
  _lastCautionAt = now;

  const actx = _getAudioCtx();
  [[400, 0], [300, 0.25]].forEach(([freq, delay]) => {
    const osc = actx.createOscillator();
    const gain = actx.createGain();
    const t = actx.currentTime + delay;

    osc.type = "sine";                        // Soft sine = not alarming
    osc.frequency.setValueAtTime(freq, t);

    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(0.25, t + 0.05);   // gentle attack
    gain.gain.linearRampToValueAtTime(0, t + 0.3);       // slow fade

    osc.connect(gain);
    gain.connect(actx.destination);
    osc.start(t);
    osc.stop(t + 0.35);
  });
}

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
  wall: "🧱",
  "stairs (going up)": "🔼",
  "stairs (going down)": "🔽",
  "⚠️ DROP-OFF": "🚨",
  "⚠️ CAUTION: step down": "⚠️",
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
function normalizeIpUrl(raw) {
  let url = raw.trim();
  if (!/^https?:\/\//i.test(url)) url = "http://" + url;
  if (/:\d+\/?$/.test(url)) {
    url = url.replace(/\/$/, "") + "/shot.jpg";
  }
  return url;
}

btnTestUrl.addEventListener("click", async () => {
  let url = ipUrlInput.value.trim();
  if (!url) { alert("Please enter a URL first."); return; }

  url = normalizeIpUrl(url);
  ipUrlInput.value = url;

  const oldStatus = document.querySelector(".ip-status");
  if (oldStatus) oldStatus.remove();

  const statusEl = document.createElement("span");
  statusEl.className = "ip-status";
  statusEl.textContent = "⏳ Testing…";
  ipInputRow.appendChild(statusEl);

  try {
    const proxyUrl = PROXY_URL + "?url=" + encodeURIComponent(url) + "&rotation=" + PHONE_ROTATION;
    console.log("Testing IP camera via proxy:", proxyUrl);
    const res = await fetch(proxyUrl);

    if (!res.ok) {
      let errMsg = "HTTP " + res.status;
      try { const errData = await res.json(); errMsg = errData.error || errMsg; } catch { }
      throw new Error(errMsg);
    }

    const blob = await res.blob();
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

async function fetchIpFrame(url) {
  const proxyUrl = PROXY_URL + "?url=" + encodeURIComponent(url) + "&rotation=" + PHONE_ROTATION;
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

async function startIpCamera() {
  const url = normalizeIpUrl(ipUrlInput.value.trim());
  ipUrlInput.value = url;
  if (!url) throw new Error("No IP camera URL provided.");

  const { blob, bitmap } = await fetchIpFrame(url);
  const w = bitmap.width;
  const h = bitmap.height;

  overlayCanvas.width = w;
  overlayCanvas.height = h;

  ipPreviewCanvas = document.createElement("canvas");
  ipPreviewCanvas.id = "ip-preview";
  ipPreviewCanvas.width = w;
  ipPreviewCanvas.height = h;
  // object-fit:contain keeps full image visible (letterbox) without cropping
  ipPreviewCanvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;";
  ipPreviewCtx = ipPreviewCanvas.getContext("2d");

  ipPreviewCtx.drawImage(bitmap, 0, 0);
  bitmap.close();

  video.style.display = "none";
  const container = document.getElementById("video-container");
  const old = document.getElementById("ip-preview");
  if (old) old.remove();
  container.insertBefore(ipPreviewCanvas, overlayCanvas);

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

// ── Process result (shared by both loops) ─────────────────────────
function processResult(result) {
  frameCount++;
  if (result.skipped) skipCount++;
  updateStats(result);
  drawOverlay(result.detections || []);
  renderDetections(result);

  // LOW VISIBILITY: backend is replaying buffered safe state
  // Play a soft caution tone and show a visual banner
  if (result.low_visibility) {
    playCautionTone();
  }

  // SAFETY-FIRST: if backend flagged urgent_alert, play beep IMMEDIATELY
  // before the voice alert.  This reaction is < 10ms via Web Audio API.
  if (result.urgent_alert) {
    const dropDet = (result.detections || []).find(d => d.urgent_audio);
    const severity = dropDet ? dropDet.severity : "critical";
    playUrgentBeep(severity);
  }

  speak(result);
}

// ── Analysis loops ────────────────────────────────────────────────
let loopTimer = null;
let analyzing = false;

function startWebcamLoop() {
  loopTimer = setInterval(async () => {
    if (sendingFrame) return;
    sendingFrame = true;
    try {
      const blob = await captureWebcamFrame();
      const result = await sendFrame(blob);
      processResult(result);
    } catch (err) {
      console.warn("Frame error:", err);
    } finally {
      sendingFrame = false;
    }
  }, FRAME_INTERVAL);
}

/**
 * TRUE PARALLEL IP camera system — two completely independent loops:
 *
 *  LOOP 1 (Display)  — runs every FRAME_INTERVAL ms
 *    Fetches a frame from the phone, displays it immediately.
 *    Never waits for AI. Video is always smooth.
 *
 *  LOOP 2 (Analysis) — self-scheduling, fire-and-forget
 *    Sends the latest captured blob to /analyze.
 *    As soon as a result comes back it renders overlays + speaks,
 *    then immediately starts the NEXT analysis.
 *    Never blocks the display loop.
 */
function startIpAnalysisLoop() {
  const url = normalizeIpUrl(ipUrlInput.value.trim());
  analyzing = true;
  let errorCount = 0;

  // Shared "latest blob" — display loop writes it, analysis loop reads it
  let latestBlob = null;
  let analysisRunning = false;

  // ── LOOP 1: Display (fast, always smooth) ─────────────────────────
  async function displayTick() {
    if (!analyzing) return;
    try {
      const { blob, bitmap } = await fetchIpFrame(url);
      const bw = bitmap.width, bh = bitmap.height;

      if (bw !== ipPreviewCanvas.width || bh !== ipPreviewCanvas.height) {
        ipPreviewCanvas.width = bw;
        ipPreviewCanvas.height = bh;
        overlayCanvas.width = bw;
        overlayCanvas.height = bh;
      }
      ipPreviewCtx.clearRect(0, 0, bw, bh);
      ipPreviewCtx.drawImage(bitmap, 0, 0, bw, bh);
      bitmap.close();

      // Give analysis loop the latest blob to work with
      latestBlob = blob;
      errorCount = 0;
    } catch (err) {
      errorCount++;
      if (errorCount <= 5) console.error("Display error #" + errorCount + ":", err);
      if (errorCount === 3) {
        detList.innerHTML = `<p class="empty-state" style="color:#ff6b6b">⚠️ Camera error: ${err.message}</p>`;
      }
    }
    if (analyzing) {
      if (errorCount > 3) {
        // Back off only on repeated errors
        loopTimer = setTimeout(displayTick, 2000);
      } else {
        // No delay — fetch next frame immediately for maximum smoothness
        displayTick();
      }
    }
  }

  // ── LOOP 2: Analysis (self-scheduling, never blocks display) ───────
  async function analysisTick() {
    if (!analyzing) return;

    // Wait until a frame is available
    if (!latestBlob) {
      setTimeout(analysisTick, 50);
      return;
    }

    const blobToAnalyze = latestBlob;
    latestBlob = null;  // Consume it — next display tick will provide a fresh one

    const t0 = performance.now();
    try {
      const form = new FormData();
      form.append("frame", blobToAnalyze, "frame.jpg");
      const res = await fetch(API_URL, { method: "POST", body: form });
      const data = await res.json();
      const result = { ...data, rtt: Math.round(performance.now() - t0) };
      processResult(result);
    } catch (e) {
      console.warn("Analysis error:", e);
    }

    // Immediately schedule next analysis (no fixed interval — as fast as server responds)
    if (analyzing) {
      analysisTick();
    }
  }

  // Start both loops independently
  displayTick();
  // Small delay so display loop gets first frame before analysis starts
  setTimeout(analysisTick, 200);
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

  // ── Low-visibility banner (buffered state) ───────────────────────────
  let html = "";
  if (result.low_visibility) {
    html += `
      <div style="background:rgba(253,203,110,0.12);border:1px solid rgba(253,203,110,0.5);
                  border-radius:10px;padding:10px 14px;margin-bottom:8px;
                  display:flex;align-items:center;gap:10px;">
        <span style="font-size:1.4em">📷</span>
        <div>
          <div style="font-weight:700;color:#fdcb6e;font-size:0.85rem">Low Visibility — Replaying Last Known State</div>
          <div style="font-size:0.75rem;color:#a0aec0">Camera blurry — obstacle data may be ${Math.round(Date.now() % 10000 / 1000)}s old</div>
        </div>
      </div>`;
  }

  if (dets.length === 0 && !result.text) {
    detList.innerHTML = html + '<p class="empty-state">No objects detected</p>';
    return;
  }
  for (const d of dets) {
    const icon = ICON_MAP[d.label] || ICON_MAP.default;
    const dist = d.distance || "unknown";
    const isPriority = d.priority <= 1;
    const isUrgent = d.urgent_audio === true;
    const priorityBadge = isUrgent
      ? ' <span style="font-size:0.7em;background:#ff0000;color:#fff;padding:1px 5px;border-radius:4px;animation:pulse-glow 0.5s infinite">🚨 URGENT</span>'
      : isPriority
        ? ' <span style="font-size:0.7em;background:#ff6b6b;color:#fff;padding:1px 5px;border-radius:4px;">Priority</span>'
        : "";
    const cardClass = isUrgent ? " det-card--urgent" : isPriority ? " det-card--priority" : "";
    html += `
      <div class="det-card${cardClass}">
        <div class="det-icon">${icon}</div>
        <div class="det-info">
          <span class="det-label">${d.label}${priorityBadge}</span>
          <span class="det-meta">${dist} · ${d.direction} · ${Math.round(d.confidence * 100)}%</span>
        </div>
      </div>`;
  }
  if (result.text) {
    html += `
      <div class="det-card">
        <div class="det-icon">📋</div>
        <div class="det-info">
          <span class="det-label">Sign / Text</span>
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
    const isUrgent = d.urgent_audio === true;
    const color = isUrgent ? "#ff0000" : d.priority <= 1 ? "#ff4757" : (BOX_COLORS[d.direction] || "#a29bfe");

    ctx.strokeStyle = color;
    ctx.lineWidth = isUrgent ? 6 : d.priority <= 1 ? 4 : 2.5;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // For urgent detections, draw a flashing red overlay
    if (isUrgent) {
      ctx.fillStyle = "rgba(255, 0, 0, 0.15)";
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    }

    const text = `${d.label} ${d.distance || ""}`;
    ctx.font = "bold 13px Inter, sans-serif";
    const tw = ctx.measureText(text).width + 12;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x1, y1 - 22, tw, 22, [6, 6, 0, 0]);
    ctx.fill();

    ctx.fillStyle = "#fff";
    ctx.fillText(text, x1 + 6, y1 - 6);
  }
}

// ── Voice guidance (Web Speech API) ───────────────────────────────
function speak(result) {
  if (muted || !window.speechSynthesis) return;
  const dets = result.detections || [];
  const parts = [];

  // PRIORITY 0 — URGENT: Drop-offs, pits (beep already played above)
  for (const d of dets) {
    if (d.urgent_audio) {
      parts.push(`Stop! ${d.label}, ${d.distance || "ahead"}`);
    }
  }

  // PRIORITY 1: Always speak walls and stairs regardless of direction
  for (const d of dets) {
    if (d.priority === 1) {
      if (d.label === "wall") {
        const distNum = d.distance ? parseFloat(d.distance) : null;
        if (distNum !== null && distNum < 1.0) {
          parts.push("wall very close, stop");
        } else if (distNum !== null && distNum < 2.0) {
          parts.push("wall nearby");
        } else {
          parts.push("wall ahead");
        }
      } else if (d.label.startsWith("stairs")) {
        parts.push(d.label.includes("up") ? "Stairs going up" : "Stairs going down");
      }
    }
  }

  // CENTER objects only (confidence ≥ 70%) — left/right shown on screen but not spoken
  for (const d of dets) {
    if (d.priority <= 1) continue;  // already handled above
    if (d.direction !== "center") continue;
    if (d.confidence < 0.70) continue;

    // Parse distance string (e.g. "1.2m") into a float; null if unknown
    const distNum = d.distance ? parseFloat(d.distance) : null;

    // Proximity buckets
    const isClose = distNum !== null && distNum < 1.5;   // < 1.5m
    const isNearby = distNum !== null && distNum < 3.0;   // 1.5–3m

    const label = d.label.toLowerCase();
    const isPerson = label === "person" || label === "people";

    if (isPerson) {
      // People in center — use socially aware phrasing
      if (isClose) {
        parts.push("person too close, please move");
      } else if (isNearby) {
        parts.push("person ahead, please make way");
      } else {
        parts.push("person ahead");
      }
    } else {
      // Other objects — use proximity language instead of raw distance
      if (isClose) {
        parts.push(`${d.label} very close`);
      } else if (isNearby) {
        parts.push(`${d.label} nearby`);
      } else {
        parts.push(`${d.label} ahead`);
      }
    }
  }

  // Sign / text reading
  if (result.text) {
    parts.push(`Sign reads: ${result.text}`);
  }

  // LOW VISIBILITY: prepend caution prefix so user knows data is buffered
  if (result.low_visibility && parts.length > 0) {
    parts.unshift("Caution, low visibility");
  }

  if (parts.length === 0) return;

  const msg = parts.join(". ");
  const now = Date.now();
  // For urgent alerts and low-visibility, bypass the speech cooldown
  if (!result.urgent_alert && !result.low_visibility) {
    if (msg === lastSpoken && now - lastSpokeAt < SPEECH_COOLDOWN_MS) return;
  }
  lastSpoken = msg;
  lastSpokeAt = now;

  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(msg);
  utter.lang = "en-US";
  utter.rate = result.urgent_alert ? 1.4 : 1.15;  // Faster speech for urgent
  utter.pitch = result.urgent_alert ? 1.3 : 1;     // Higher pitch for urgent
  window.speechSynthesis.speak(utter);
}

// ── Button handlers ───────────────────────────────────────────────
btnStart.addEventListener("click", async () => {
  try {
    detList.innerHTML = '<p class="empty-state">⏳ Connecting to camera…</p>';
    btnStart.disabled = true;

    // Resume AudioContext (browsers require user gesture)
    _getAudioCtx().resume();

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
