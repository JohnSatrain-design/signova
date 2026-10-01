from __future__ import annotations

from pathlib import Path
import logging
import os
import shutil
from threading import Lock, Thread
from time import sleep, time
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from signova.config import DEFAULT_MODEL_PATH, MODELS_DIR, ROOT_DIR, ensure_project_dirs
from signova.core.camera import CameraService
from signova.core.inference import ModelManager
from signova.core.landmarks import IMPORT_ERROR as LANDMARK_IMPORT_ERROR, mp
from signova.core.state import RuntimeState
from signova.core.translation import TranslationProcessor
from signova.core.tts import TextToSpeechEngine
from signova.training.dataset import DatasetManager


STATIC_DIR = ROOT_DIR / "signova" / "ui" / "static"
ADMIN_KEY = os.getenv("SIGNOVA_ADMIN_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("signova")

ensure_project_dirs()

state = RuntimeState()
dataset_manager = DatasetManager()
translator = TranslationProcessor()
model_manager = ModelManager(DEFAULT_MODEL_PATH)
model_manager.load()
logger.info("Loading AI model: %s", DEFAULT_MODEL_PATH)
if model_manager.is_loaded:
    logger.info("Model loaded successfully: %d sign classes", len(model_manager.labels))
else:
    logger.error("Model loading failed: %s", model_manager.error)
tts_engine = TextToSpeechEngine()
camera_service = CameraService(state, model_manager, translator, dataset_manager)
training_lock = Lock()

state.update(
    model_online=model_manager.is_loaded,
    model_error=model_manager.error,
    voice_engine_ready=tts_engine.ready,
    voice_error=tts_engine.error,
    training_status={"state": "idle", "message": "No training job running."},
)

app = FastAPI(
    title="SIGNOVA",
    description="Real-time sign language recognition and voice translation prototype.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class CameraStartRequest(BaseModel):
    index: int = Field(default=0, ge=0, le=8)


class AdminLabelRequest(BaseModel):
    label: str = Field(min_length=1, max_length=64)


class AdminCollectionRequest(BaseModel):
    label: str = Field(min_length=1, max_length=64)
    every_n_frames: int = Field(default=3, ge=1, le=60)


class AdminTrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str = Field(default="random_forest", pattern="^(mlp|random_forest)$")
    min_samples_per_class: int = Field(default=8, ge=2, le=500)


def require_admin(key: str | None) -> None:
    if not ADMIN_KEY:
        raise HTTPException(
            status_code=503,
            detail="Administrator access is not configured.",
        )
    if key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Administrator access required.")


def invalidate_runtime_model(reason: str) -> None:
    model_manager.invalidate(reason)
    state.update(model_online=False, model_error=reason)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
def admin_index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "admin.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/training")
def training_index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "admin.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    def generate():
        while True:
            frame = camera_service.get_latest_jpeg()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            sleep(1 / 24)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    model_status = model_manager.status()
    snapshot = state.snapshot()
    snapshot.update(
        {
            "model_online": model_status["online"],
            "model_error": model_status["error"],
            "model_path": model_status["path"],
            "model_labels": model_status["labels"],
            "model_metrics": model_status["metrics"],
            "voice_engine_ready": tts_engine.ready,
            "voice_error": tts_engine.error,
            "last_speech_at": tts_engine.last_spoken_at,
            "translation": translator.current_text(),
            "history": translator.history(),
            "vocabulary": model_status["labels"],
        }
    )
    return snapshot


@app.get("/health")
def health() -> dict[str, Any]:
    model_status = model_manager.status()
    snapshot = state.snapshot()
    return {
        "status": "ok" if model_status["online"] else "degraded",
        "camera": snapshot["camera_online"],
        "model": model_status["online"],
        "model_error": model_status["error"],
        "mediapipe": mp is not None,
        "mediapipe_error": LANDMARK_IMPORT_ERROR if mp is None else "",
        "vocabulary_count": len(model_status["labels"]),
    }


@app.post("/api/camera/start")
def start_camera(payload: CameraStartRequest) -> JSONResponse:
    ok, message = camera_service.start(payload.index)
    status_code = 200 if ok else 409
    return JSONResponse({"ok": ok, "message": message}, status_code=status_code)


@app.post("/api/camera/stop")
def stop_camera() -> dict[str, Any]:
    camera_service.stop()
    return {"ok": True, "message": "Camera stopped."}


@app.post("/api/reset")
def reset_translation() -> dict[str, Any]:
    camera_service.clear_translation()
    return {"ok": True, "message": "Translation history cleared."}


@app.post("/api/speak")
def speak() -> JSONResponse:
    text = translator.current_text()
    ok = tts_engine.speak(text)
    state.update(
        voice_engine_ready=tts_engine.ready,
        voice_error=tts_engine.error,
        last_speech_at=time() if ok else state.snapshot()["last_speech_at"],
    )
    status_code = 200 if ok else 409
    return JSONResponse(
        {
            "ok": ok,
            "message": "Speech generated." if ok else "No text or TTS engine unavailable.",
        },
        status_code=status_code,
    )


@app.get("/api/admin/status")
def admin_status(x_signova_admin_key: str | None = Header(default=None)) -> dict[str, Any]:
    require_admin(x_signova_admin_key)
    return {
        "camera": state.snapshot(),
        "collection": camera_service.collection_state(),
        "sample_counts": dataset_manager.sample_counts(),
        "deleted_vocabulary": dataset_manager.deleted_vocabulary(),
        "approved_vocabulary": dataset_manager.vocabulary(),
        "model": model_manager.status(),
        "training": state.snapshot()["training_status"],
    }


@app.post("/api/admin/collection/start")
def admin_start_collection(
    payload: AdminCollectionRequest,
    x_signova_admin_key: str | None = Header(default=None),
) -> dict[str, Any]:
    require_admin(x_signova_admin_key)
    try:
        label = dataset_manager.add_label(payload.label)
    except ValueError as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
    if not camera_service.running:
        return JSONResponse(
            {"ok": False, "message": "Open the camera before recording."},
            status_code=409,
        )
    camera_service.start_collection(label, payload.every_n_frames)
    return {"ok": True, "message": f"Recording {label}.", "label": label}


@app.post("/api/admin/collection/stop")
def admin_stop_collection(
    x_signova_admin_key: str | None = Header(default=None),
) -> dict[str, Any]:
    require_admin(x_signova_admin_key)
    camera_service.stop_collection()
    return {"ok": True, "collection": camera_service.collection_state()}


@app.post("/api/admin/collection/save")
def admin_save_collection(
    x_signova_admin_key: str | None = Header(default=None),
) -> JSONResponse:
    require_admin(x_signova_admin_key)
    ok, message, samples = camera_service.save_collection()
    return JSONResponse(
        {"ok": ok, "message": message, "samples": samples},
        status_code=200 if ok else 409,
    )


@app.post("/api/admin/sign/delete")
def admin_delete_sign(
    payload: AdminLabelRequest,
    x_signova_admin_key: str | None = Header(default=None),
) -> JSONResponse:
    require_admin(x_signova_admin_key)
    try:
        label = dataset_manager.archive_label(payload.label)
    except ValueError as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
    camera_service.clear_translation()
    invalidate_runtime_model(f"Sign {label} deleted. Model needs retraining.")
    return JSONResponse({"ok": True, "message": f"{label} archived. Model needs retraining."})


@app.post("/api/admin/sign/restore")
def admin_restore_sign(
    payload: AdminLabelRequest,
    x_signova_admin_key: str | None = Header(default=None),
) -> JSONResponse:
    require_admin(x_signova_admin_key)
    try:
        label = dataset_manager.restore_label(payload.label)
    except ValueError as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
    invalidate_runtime_model(f"Sign {label} restored. Model needs retraining.")
    return JSONResponse({"ok": True, "message": f"{label} restored. Model needs retraining."})


@app.post("/api/admin/sign/permanent-delete")
def admin_permanent_delete_sign(
    payload: AdminLabelRequest,
    x_signova_admin_key: str | None = Header(default=None),
) -> JSONResponse:
    require_admin(x_signova_admin_key)
    try:
        label = dataset_manager.permanently_delete_label(payload.label)
    except ValueError as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
    return {"ok": True, "message": f"{label} permanently deleted."}


@app.post("/api/admin/reset-all")
def admin_reset_all(
    x_signova_admin_key: str | None = Header(default=None),
) -> JSONResponse:
    require_admin(x_signova_admin_key)
    camera_service.clear_translation()
    dataset_manager.reset_all()
    for path in MODELS_DIR.iterdir():
        if path.is_file() and path.name != "README.md":
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    invalidate_runtime_model("All training data deleted. Model is not trained.")
    state.update(training_status={"state": "idle", "message": "Training data reset."})
    return {"ok": True, "message": "All training data, archive, and model files were removed."}


@app.post("/api/admin/train")
def admin_train(
    payload: AdminTrainRequest,
    x_signova_admin_key: str | None = Header(default=None),
) -> JSONResponse:
    require_admin(x_signova_admin_key)
    if training_lock.locked():
        return JSONResponse({"ok": False, "message": "Training is already running."}, status_code=409)
    sample_counts = dataset_manager.sample_counts()
    active_classes = {label: count for label, count in sample_counts.items() if count > 0}
    insufficient = sorted(
        label
        for label, count in active_classes.items()
        if count < payload.min_samples_per_class
    )
    if len(active_classes) < 2:
        return JSONResponse(
            {
                "ok": False,
                "message": "Training needs at least two approved sign classes with valid samples.",
            },
            status_code=409,
        )
    if insufficient:
        details = ", ".join(
            f"{label} ({active_classes[label]}/{payload.min_samples_per_class})"
            for label in insufficient
        )
        return JSONResponse(
            {
                "ok": False,
                "message": f"Collect more samples before training: {details}.",
            },
            status_code=409,
        )

    def train_job() -> None:
        with training_lock:
            state.update(training_status={"state": "running", "message": "Training AI model."})
            try:
                from signova.training.train_model import train_model

                metrics = train_model(
                    output_path=DEFAULT_MODEL_PATH,
                    model_type=payload.model_type,
                    min_samples_per_class=payload.min_samples_per_class,
                )
                model_manager.load()
                state.update(
                    model_online=model_manager.is_loaded,
                    model_error=model_manager.error,
                    training_status={
                        "state": "complete",
                        "message": "Model training complete.",
                        "metrics": metrics,
                    },
                )
            except Exception as exc:
                state.update(training_status={"state": "error", "message": str(exc)})

    Thread(target=train_job, name="signova-admin-trainer", daemon=True).start()
    return JSONResponse({"ok": True, "message": "Training started."}, status_code=202)


