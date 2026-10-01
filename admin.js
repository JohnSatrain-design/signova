const key = sessionStorage.getItem("signovaAdminKey") || window.prompt("Administrator key") || "";
if (key) sessionStorage.setItem("signovaAdminKey", key);
const headers = { "Content-Type": "application/json", "X-SIGNOVA-ADMIN-KEY": key };
const dom = {
  cameraStatus: document.querySelector("#camera-status"),
  modelStatus: document.querySelector("#model-status"),
  video: document.querySelector("#video-feed"),
  signName: document.querySelector("#sign-name"),
  recordingState: document.querySelector("#recording-state"),
  sampleCount: document.querySelector("#sample-count"),
  recordingDetail: document.querySelector("#recording-detail"),
  knownSigns: document.querySelector("#known-signs"),
  trainingStatus: document.querySelector("#training-status"),
  modelError: document.querySelector("#model-error"),
  recognitionDebug: document.querySelector("#recognition-debug"),
  message: document.querySelector("#message"),
  deletedWords: document.querySelector("#deleted-words"),
  activeWords: document.querySelector("#active-words"),
};

async function request(path, options = {}) {
  const response = await fetch(path, { headers, ...options });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || payload.message || "Request failed.");
  return payload;
}

function message(text, error = false) {
  dom.message.textContent = text;
  dom.message.style.color = error ? "var(--red)" : "var(--amber)";
}

function render(status) {
  const camera = status.camera || {};
  const collection = status.collection || {};
  const model = status.model || {};
  dom.cameraStatus.classList.toggle("online", Boolean(camera.camera_online));
  dom.cameraStatus.querySelector("span:last-child").textContent = camera.camera_online ? "CAMERA ONLINE" : "CAMERA OFFLINE";
  dom.modelStatus.classList.toggle("online", Boolean(model.online));
  dom.modelStatus.querySelector("span:last-child").textContent = model.online ? "AI MODEL ONLINE" : "AI MODEL OFFLINE";
  dom.recordingState.textContent = collection.active ? "RECORDING" : (collection.samples ? "RECORDING STOPPED" : "IDLE");
  dom.sampleCount.textContent = collection.samples || 0;
  dom.recordingDetail.textContent = collection.active ? "Capture valid hand landmarks now." : "Recording is stopped.";
  dom.knownSigns.textContent = (model.labels || []).length;
  const training = status.training || {};
  dom.trainingStatus.textContent = training.message || training.state || "No training job running.";
  if (training.state === "error") message(training.message, true);
  dom.modelError.textContent = model.online ? "Model ready for inference." : (model.error || "No model loaded.");
  dom.recognitionDebug.textContent = `Candidate: ${camera.prediction_candidate || "—"} | Stable: ${camera.stable_frames || 0}/${camera.required_stable_frames || 3}`;
  dom.deletedWords.innerHTML = "";
  dom.activeWords.innerHTML = "";
  const active = Object.entries(status.sample_counts || {});
  if (!active.length) dom.activeWords.textContent = "No active signs.";
  active.forEach(([label, count]) => {
    const row = document.createElement("div");
    row.className = "admin-list-row";
    row.innerHTML = `<span>${label} (${count} samples)</span>`;
    const remove = document.createElement("button");
    remove.className = "mini-button";
    remove.textContent = "Delete";
    remove.onclick = () => mutateSign("/api/admin/sign/delete", label);
    row.appendChild(remove);
    dom.activeWords.appendChild(row);
  });
  const deleted = status.deleted_vocabulary || [];
  if (!deleted.length) dom.deletedWords.textContent = "No archived signs.";
  deleted.forEach((label) => {
    const row = document.createElement("div");
    row.className = "admin-list-row";
    row.innerHTML = `<span>${label}</span>`;
    const restore = document.createElement("button");
    restore.className = "mini-button";
    restore.textContent = "Restore";
    restore.onclick = () => mutateSign("/api/admin/sign/restore", label);
    const remove = document.createElement("button");
    remove.className = "mini-button";
    remove.textContent = "Permanently Delete";
    remove.onclick = () => mutateSign("/api/admin/sign/permanent-delete", label);
    row.append(restore, remove);
    dom.deletedWords.appendChild(row);
  });
}

async function mutateSign(path, label) {
  try {
    if (!window.confirm(`${path.includes("restore") ? "Restore" : "Permanently delete"} ${label}?`)) return;
    const result = await request(path, { method: "POST", body: JSON.stringify({ label }) });
    message(result.message);
    refresh();
  } catch (error) { message(error.message, true); }
}

async function refresh() {
  try { render(await request("/api/admin/status")); } catch (error) { message(error.message, true); }
}

document.querySelector("#open-camera").addEventListener("click", async () => {
  try { const result = await request("/api/camera/start", { method: "POST", body: JSON.stringify({ index: 0 }) }); dom.video.src = `/video_feed?t=${Date.now()}`; message(result.message); refresh(); }
  catch (error) { message(error.message, true); }
});
document.querySelector("#stop-camera").addEventListener("click", async () => {
  try { await request("/api/camera/stop", { method: "POST" }); refresh(); }
  catch (error) { message(error.message, true); }
});
document.querySelector("#start-recording").addEventListener("click", async () => {
  try {
    const label = dom.signName.value.trim();
    if (!label) throw new Error("Enter an approved sign name.");
    const current = await request("/api/admin/status");
    const normalized = label.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_|_$/g, "").toUpperCase();
    if ((current.sample_counts || {})[normalized] > 0 && !window.confirm("Sign already exists. Add more samples to this sign?")) return;
    const result = await request("/api/admin/collection/start", { method: "POST", body: JSON.stringify({ label }) });
    message(result.message); refresh();
  } catch (error) { message(error.message, true); }
});
document.querySelector("#stop-recording").addEventListener("click", async () => {
  try { await request("/api/admin/collection/stop", { method: "POST" }); message("Recording stopped. Review samples, then save the sign."); refresh(); }
  catch (error) { message(error.message, true); }
});
document.querySelector("#save-sign").addEventListener("click", async () => {
  try { const result = await request("/api/admin/collection/save", { method: "POST" }); message(`SIGN SAVED: ${result.samples} samples. Train the model to learn it.`); refresh(); }
  catch (error) { message(error.message, true); }
});
document.querySelector("#train-model").addEventListener("click", async () => {
  try { const result = await request("/api/admin/train", { method: "POST", body: JSON.stringify({ model_type: "random_forest", min_samples_per_class: 8 }) }); message(result.message); refresh(); }
  catch (error) { message(error.message, true); }
});
document.querySelector("#reset-all").addEventListener("click", async () => {
  try {
    if (!window.confirm("RESET ALL TRAINING DATA? This removes active samples, deleted archives, labels, and the model.")) return;
    const result = await request("/api/admin/reset-all", { method: "POST" });
    message(result.message);
    refresh();
  } catch (error) { message(error.message, true); }
});

refresh();
setInterval(refresh, 800);
