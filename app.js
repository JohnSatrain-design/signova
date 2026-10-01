const dom = {
  cameraStatus: document.querySelector("#camera-status"),
  modelStatus: document.querySelector("#model-status"),
  trackingStatus: document.querySelector("#tracking-status"),
  voiceStatus: document.querySelector("#voice-status"),
  visionState: document.querySelector("#vision-state"),
  fps: document.querySelector("#fps"),
  aiFps: document.querySelector("#ai-fps"),
  inferenceTime: document.querySelector("#inference-time"),
  trackingConfidence: document.querySelector("#tracking-confidence"),
  videoShell: document.querySelector("#video-shell"),
  videoFeed: document.querySelector("#video-feed"),
  detectedSign: document.querySelector("#detected-sign"),
  confidenceValue: document.querySelector("#confidence-value"),
  confidenceOrbit: document.querySelector("#confidence-orbit"),
  confidenceBar: document.querySelector("#confidence-bar"),
  translationText: document.querySelector("#translation-text"),
  speakButton: document.querySelector("#speak-button"),
  historyList: document.querySelector("#history-list"),
  historyCount: document.querySelector("#history-count"),
  vocabularyList: document.querySelector("#vocabulary-list"),
  vocabularyCount: document.querySelector("#vocabulary-count"),
  toast: document.querySelector("#toast"),
  settingsButton: document.querySelector("#settings-button"),
  settingsDrawer: document.querySelector("#settings-drawer"),
  closeSettings: document.querySelector("#close-settings"),
  modelPath: document.querySelector("#model-path"),
  loadedLabels: document.querySelector("#loaded-labels"),
  backendStatus: document.querySelector("#backend-status"),
  recognitionStatus: document.querySelector("#recognition-status"),
  runtimeError: document.querySelector("#runtime-error"),
};

let lastSign = "";
let toastTimer = null;

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (error) {
    throw new Error(`Backend unavailable: ${error.message}`);
  }
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : { message: await response.text() };
  if (!response.ok) {
    throw new Error(payload.message || payload.detail || "Request failed.");
  }
  return payload;
}

function setStatus(element, online, text, warn = false) {
  element.classList.toggle("online", Boolean(online));
  element.classList.toggle("warn", Boolean(warn && !online));
  element.querySelector("span:last-child").textContent = text;
}

function showToast(message) {
  dom.toast.textContent = message;
  dom.toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => dom.toast.classList.remove("visible"), 2800);
}

function renderVocabulary(labels) {
  dom.vocabularyList.innerHTML = "";
  dom.vocabularyCount.textContent = `${labels.length} SIGNS`;
  labels.forEach((label) => {
    const row = document.createElement("div");
    row.className = "sample-row";
    row.textContent = label;
    dom.vocabularyList.appendChild(row);
  });
}

function renderHistory(history) {
  dom.historyCount.textContent = `${history.length} ITEMS`;
  dom.historyList.innerHTML = "";

  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent = "No detections recorded";
    dom.historyList.appendChild(empty);
    return;
  }

  history.forEach((item) => {
    const row = document.createElement("div");
    row.className = "history-item";
    row.innerHTML = `
      <div class="time">${item.timestamp}</div>
      <div class="sign">${item.sign}</div>
      <div>${item.translation || item.sentence || ""}</div>
      <div class="conf">${percent(item.confidence)}</div>
    `;
    dom.historyList.appendChild(row);
  });
}

function renderStatus(status) {
  setStatus(
    dom.cameraStatus,
    status.camera_online,
    status.camera_online ? "CAMERA ONLINE" : "CAMERA OFFLINE",
    true,
  );
  setStatus(
    dom.modelStatus,
    status.model_online,
    status.model_online ? "AI MODEL ONLINE" : "AI MODEL OFFLINE",
    true,
  );
  setStatus(
    dom.trackingStatus,
    status.hand_tracking_active,
    status.hand_tracking_active ? "HAND TRACKING ACTIVE" : "HAND TRACKING STANDBY",
    status.camera_online,
  );
  setStatus(
    dom.voiceStatus,
    status.voice_engine_ready,
    status.voice_engine_ready ? "VOICE ENGINE READY" : "VOICE ENGINE OFFLINE",
    true,
  );

  dom.visionState.textContent = status.ai_vision_active
    ? status.hand_tracking_active
      ? "AI VISION ACTIVE"
      : "AI VISION SEARCHING"
    : "AI VISION STANDBY";
  dom.fps.textContent = `CAM ${Number(status.fps || 0).toFixed(1)} FPS`;
  dom.aiFps.textContent = `AI ${Number(status.ai_fps || 0).toFixed(1)} FPS`;
  dom.inferenceTime.textContent = `AI ${Number(status.inference_ms || 0).toFixed(0)} ms`;
  dom.trackingConfidence.textContent = `TRACK ${percent(status.tracking_confidence)}`;
  dom.visionState.textContent = status.recognition_status || dom.visionState.textContent;
  dom.videoShell.classList.toggle("active", Boolean(status.hand_tracking_active));

  const displaySign = status.detected_sign
    ? `[ ${status.detected_sign} ]`
    : "[ — ]";
  dom.detectedSign.textContent = displaySign;
  if (status.detected_sign && status.detected_sign !== lastSign) {
    dom.detectedSign.classList.remove("changed");
    void dom.detectedSign.offsetWidth;
    dom.detectedSign.classList.add("changed");
    lastSign = status.detected_sign;
  }

  const confidence = Number(status.confidence || 0);
  dom.confidenceValue.textContent = percent(confidence);
  dom.confidenceOrbit.style.setProperty("--confidence-angle", `${confidence * 360}deg`);
  dom.confidenceBar.style.width = percent(confidence);
  dom.translationText.textContent = status.translation || "No translation yet";
  dom.speakButton.disabled = !status.voice_engine_ready || !status.translation;

  renderVocabulary(status.vocabulary || []);
  renderHistory(status.history || []);

  dom.modelPath.textContent = status.model_path || "data/models/signova_model.joblib";
  dom.loadedLabels.textContent = (status.model_labels || []).join(", ") || "None";
  dom.recognitionStatus.textContent = status.ai_vision_active ? "ACTIVE" : "IDLE";
  dom.runtimeError.textContent =
    status.camera_error || status.model_error || status.voice_error || "None";
}

async function refreshStatus() {
  try {
    const status = await request("/api/status");
    dom.backendStatus.textContent = "ONLINE";
    renderStatus(status);
  } catch (error) {
    dom.backendStatus.textContent = `OFFLINE: ${error.message}`;
    showToast(error.message);
  }
}

document.querySelector("#start-camera").addEventListener("click", async () => {
  try {
    const payload = await request("/api/camera/start", {
      method: "POST",
      body: JSON.stringify({ index: 0 }),
    });
    dom.videoFeed.src = `/video_feed?t=${Date.now()}`;
    showToast(payload.message);
    refreshStatus();
  } catch (error) {
    showToast(error.message);
  }
});

document.querySelector("#stop-camera").addEventListener("click", async () => {
  const payload = await request("/api/camera/stop", { method: "POST" });
  showToast(payload.message);
  refreshStatus();
});

document.querySelector("#reset-translation").addEventListener("click", async () => {
  const payload = await request("/api/reset", { method: "POST" });
  showToast(payload.message);
  refreshStatus();
});

dom.speakButton.addEventListener("click", async () => {
  try {
    const payload = await request("/api/speak", { method: "POST" });
    dom.speakButton.classList.remove("speaking");
    void dom.speakButton.offsetWidth;
    dom.speakButton.classList.add("speaking");
    showToast(payload.message);
  } catch (error) {
    showToast(error.message);
  }
});

dom.settingsButton.addEventListener("click", () => {
  dom.settingsDrawer.classList.add("open");
  dom.settingsDrawer.setAttribute("aria-hidden", "false");
});

dom.closeSettings.addEventListener("click", () => {
  dom.settingsDrawer.classList.remove("open");
  dom.settingsDrawer.setAttribute("aria-hidden", "true");
});

refreshStatus();
setInterval(refreshStatus, 650);
