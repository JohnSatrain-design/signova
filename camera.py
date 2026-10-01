from __future__ import annotations

from dataclasses import dataclass
import logging
from threading import Event, Lock, RLock, Thread
from time import monotonic, perf_counter, sleep, time
from typing import Any

import numpy as np

from signova.config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    AI_PROCESSING_HEIGHT,
    AI_PROCESSING_WIDTH,
)
from signova.core.inference import ModelManager, Prediction, PredictionSmoother
from signova.core.landmarks import LandmarkDetection, LandmarkDetector, cv2
from signova.core.state import RuntimeState
from signova.core.translation import TranslationProcessor
from signova.training.dataset import DatasetManager


logger = logging.getLogger("signova.camera")


@dataclass
class CollectionState:
    active: bool = False
    label: str = ""
    saved: int = 0
    every_n_frames: int = 6
    samples: list[np.ndarray] | None = None


class CameraService:
    def __init__(
        self,
        state: RuntimeState,
        model_manager: ModelManager,
        translator: TranslationProcessor,
        dataset_manager: DatasetManager,
    ) -> None:
        self.state = state
        self.model_manager = model_manager
        self.translator = translator
        self.dataset_manager = dataset_manager
        self.smoother = PredictionSmoother()
        self._capture: Any | None = None
        self._detector: LandmarkDetector | None = None
        self._thread: Thread | None = None
        self._inference_thread: Thread | None = None
        self._stop_event = Event()
        self._lifecycle_lock = Lock()
        self._generation = 0
        self._lock = RLock()
        self._latest_jpeg: bytes | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_id = 0
        self._processed_frame_id = 0
        self._latest_frame_at = 0.0
        self._latest_landmark_points: dict[str, list[tuple[float, float]]] = {}
        self._collection = CollectionState()
        self._frame_counter = 0
        self._last_performance_log_at = 0.0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, camera_index: int = 0) -> tuple[bool, str]:
        with self._lifecycle_lock:
            if cv2 is None:
                message = "OpenCV is not available. Install project requirements."
                self.state.update(camera_online=False, camera_error=message)
                return False, message

            if self.running:
                return True, "Camera is already running."
            if self._inference_thread and self._inference_thread.is_alive():
                self._inference_thread.join(timeout=5.0)
                if self._inference_thread.is_alive():
                    message = "Camera is still stopping. Try Start again in a moment."
                    self.state.update(camera_online=False, camera_error=message)
                    return False, message

            try:
                self._detector = LandmarkDetector()
                capture = cv2.VideoCapture(camera_index)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)
                capture.set(cv2.CAP_PROP_FPS, DEFAULT_CAMERA_FPS)
            except Exception as exc:
                self.state.update(camera_online=False, camera_error=str(exc))
                return False, str(exc)

            if not capture.isOpened():
                message = f"Could not open camera index {camera_index}."
                if self._detector is not None:
                    self._detector.close()
                    self._detector = None
                self.state.update(camera_online=False, camera_error=message)
                return False, message

            self._capture = capture
            self._generation += 1
            generation = self._generation
            self._stop_event.clear()
            self._thread = Thread(
                target=self._loop,
                args=(generation,),
                name="signova-camera",
                daemon=True,
            )
            self._inference_thread = Thread(
                target=self._inference_loop,
                args=(generation,),
                name="signova-inference",
                daemon=True,
            )
            self._thread.start()
            self._inference_thread.start()
            self.state.update(
                camera_online=True,
                ai_vision_active=True,
                camera_error="",
                model_online=self.model_manager.is_loaded,
            )
            return True, "Camera started."

    def stop(self) -> None:
        self._lifecycle_lock.acquire()
        self._generation += 1
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)
        self._release()
        self.smoother.reset()
        with self._lock:
            self._latest_landmark_points = {}
            self._latest_frame = None
            self._collection.active = False
        self.state.update(
            camera_online=False,
            ai_vision_active=False,
            hand_tracking_active=False,
            collection_active=False,
            fps=0.0,
            ai_fps=0.0,
            inference_ms=0.0,
            frame_buffer_size=0,
            tracking_confidence=0.0,
            confidence=0.0,
            detected_sign="",
            recognition_status="WAITING FOR SIGN",
            landmark_count=0,
        )
        self._lifecycle_lock.release()

    def get_latest_jpeg(self) -> bytes:
        with self._lock:
            if self._latest_jpeg is not None:
                return self._latest_jpeg
        return self._placeholder_jpeg()

    def start_collection(self, label: str, every_n_frames: int = 6) -> None:
        with self._lock:
            self._collection = CollectionState(
                active=True,
                label=label,
                saved=0,
                every_n_frames=max(every_n_frames, 1),
                samples=[],
            )
        self.state.update(
            collection_active=True,
            collection_label=label,
            collection_saved=0,
        )

    def stop_collection(self) -> None:
        with self._lock:
            saved = self._collection.saved
            self._collection.active = False
        self.state.update(collection_active=False, collection_saved=saved)

    def collection_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self._collection.active,
                "label": self._collection.label,
                "saved": self._collection.saved,
                "samples": len(self._collection.samples or []),
                "every_n_frames": self._collection.every_n_frames,
            }

    def save_collection(self) -> tuple[bool, str, int]:
        with self._lock:
            label = self._collection.label
            samples = list(self._collection.samples or [])
            self._collection.samples = []
            self._collection.saved = 0

        if not label or not samples:
            return False, "No valid tracked samples were captured.", 0

        for feature in samples:
            self.dataset_manager.save_sample(
                label,
                feature,
                {"source": "admin_recording"},
            )
        return True, f"Saved {len(samples)} samples for {label}.", len(samples)

    def clear_translation(self) -> None:
        self.translator.clear()
        self.smoother.reset()
        self.state.update(detected_sign="", confidence=0.0, translation="")

    def _loop(self, generation: int) -> None:
        last_frame_at = monotonic()
        fps = 0.0

        while not self._stop_event.is_set() and generation == self._generation:
            if self._capture is None:
                break

            ok, frame = self._capture.read()
            if not ok:
                self._stop_event.set()
                self.state.update(
                    camera_online=False,
                    ai_vision_active=False,
                    camera_error="Camera frame could not be read.",
                )
                break

            now = monotonic()
            delta = max(now - last_frame_at, 0.001)
            last_frame_at = now
            instant_fps = 1.0 / delta
            fps = instant_fps if fps == 0 else (fps * 0.85) + (instant_fps * 0.15)

            with self._lock:
                self._latest_frame = frame
                self._latest_frame_id += 1
                self._latest_frame_at = now
                landmark_points = {
                    key: list(points)
                    for key, points in self._latest_landmark_points.items()
                }

            # MediaPipe remains in the inference worker. This only composites the
            # latest known geometry onto the newest camera frame.
            display_frame = frame.copy()
            LandmarkDetector.draw_tracking_overlay(display_frame, landmark_points)
            self._store_jpeg(display_frame)
            self.state.update(camera_online=True, fps=fps, frame_buffer_size=1)

        if generation == self._generation:
            self.state.update(camera_online=False, ai_vision_active=False)

    def _inference_loop(self, generation: int) -> None:
        last_inference_at = 0.0
        last_ai_frame_at = 0.0
        ai_fps = 0.0
        inference_interval = 1.0 / 15.0

        while not self._stop_event.is_set() and generation == self._generation:
            now = monotonic()
            if now - last_inference_at < inference_interval:
                sleep(0.005)
                continue

            with self._lock:
                if self._latest_frame is None or self._latest_frame_id == self._processed_frame_id:
                    frame = None
                    frame_id = self._processed_frame_id
                else:
                    frame = self._latest_frame
                    frame_id = self._latest_frame_id
                    frame_at = self._latest_frame_at

            if frame is None:
                sleep(0.005)
                continue

            last_inference_at = now
            try:
                started_at = perf_counter()
                ai_frame = cv2.resize(
                    frame,
                    (AI_PROCESSING_WIDTH, AI_PROCESSING_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
                detection = self._process_frame(ai_frame)
                with self._lock:
                    self._latest_landmark_points = detection.landmark_points
                prediction = self._predict(detection)
                emitted_label = self.smoother.update(prediction)
                if emitted_label and prediction:
                    self.translator.accept_sign(emitted_label, prediction.confidence)
                self._collect_if_enabled(detection)
                elapsed_ms = (perf_counter() - started_at) * 1000
                if last_ai_frame_at:
                    instant_ai_fps = 1.0 / max(now - last_ai_frame_at, 0.001)
                    ai_fps = instant_ai_fps if not ai_fps else ai_fps * 0.8 + instant_ai_fps * 0.2
                last_ai_frame_at = now
                self._publish_state(
                    detection,
                    prediction,
                    self.state.snapshot()["fps"],
                    ai_fps,
                    elapsed_ms,
                    now - frame_at,
                )
                self.state.update(**{
                    "prediction_candidate": self.smoother.status()["candidate"],
                    "stable_frames": self.smoother.status()["stable_frames"],
                    "required_stable_frames": self.smoother.status()["required_frames"],
                })
                if now - self._last_performance_log_at >= 5.0:
                    logger.info(
                        "[CAMERA] FPS: %.1f | [AI] FPS: %.1f | [AI] Inference: %.1f ms | [FRAME BUFFER] Size: %d",
                        self.state.snapshot()["fps"],
                        ai_fps,
                        elapsed_ms,
                        1 if frame_id != self._latest_frame_id else 0,
                    )
                    self._last_performance_log_at = now
                if elapsed_ms > 150:
                    logger.warning("[WARNING] AI inference exceeded 150 ms: %.1f ms", elapsed_ms)
                with self._lock:
                    self._processed_frame_id = frame_id
                self.state.update(frame_buffer_size=0)
            except Exception as exc:
                # Do not retry the same MediaPipe packet timestamp forever after a
                # graph error; advance to the newest capture frame instead.
                with self._lock:
                    self._processed_frame_id = frame_id
                self.smoother.reset()
                self.state.update(
                    detected_sign="",
                    recognition_status="LOW CONFIDENCE / NO SIGN",
                    prediction_candidate="",
                    stable_frames=0,
                    confidence=0.0,
                    hand_tracking_active=False,
                    tracking_confidence=0.0,
                )
                self.state.update(camera_error=f"Prediction failed: {exc}")
                sleep(0.1)

        if generation == self._generation:
            self._release()

    def _process_frame(self, frame: np.ndarray) -> LandmarkDetection:
        if self._detector is None:
            raise RuntimeError("Landmark detector is not initialized.")
        return self._detector.process(frame)

    def _predict(self, detection: LandmarkDetection) -> Prediction | None:
        if not detection.has_hand or not self.model_manager.is_loaded:
            if not detection.has_hand:
                self.smoother.reset()
                self.state.update(
                    detected_sign="",
                    recognition_status="WAITING FOR SIGN",
                    prediction_candidate="",
                    stable_frames=0,
                    confidence=0.0,
                    hand_tracking_active=False,
                    tracking_confidence=0.0,
                    last_detection_at=None,
                )
            return None
        prediction = self.model_manager.predict(detection.features)
        return prediction

    def _collect_if_enabled(self, detection: LandmarkDetection) -> None:
        self._frame_counter += 1
        with self._lock:
            collection = self._collection
            should_save = (
                collection.active
                and detection.has_hand
                and self._frame_counter % collection.every_n_frames == 0
            )

        if not should_save:
            return

        with self._lock:
            if self._collection.samples is None:
                self._collection.samples = []
            self._collection.samples.append(detection.features.copy())
            saved = len(self._collection.samples)
        self.state.update(collection_saved=saved)

    def _store_jpeg(self, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        if not ok:
            return
        with self._lock:
            self._latest_jpeg = encoded.tobytes()

    def _publish_state(
        self,
        detection: LandmarkDetection,
        prediction: Prediction | None,
        fps: float,
        ai_fps: float,
        inference_ms: float,
        frame_age_seconds: float,
    ) -> None:
        confidence = prediction.confidence if prediction else 0.0
        valid_prediction = prediction is not None and confidence >= CONFIDENCE_THRESHOLD
        detected_sign = prediction.label if valid_prediction and prediction else ""
        if not detection.has_hand:
            recognition_status = "WAITING FOR SIGN"
        elif not valid_prediction:
            recognition_status = "LOW CONFIDENCE / NO SIGN"
        else:
            recognition_status = "CONFIRMED CANDIDATE"
        self.state.update(
            camera_online=True,
            ai_vision_active=True,
            hand_tracking_active=detection.has_hand,
            model_online=self.model_manager.is_loaded,
            model_error=self.model_manager.error,
            fps=fps,
            ai_fps=ai_fps,
            inference_ms=inference_ms,
            frame_buffer_size=1 if frame_age_seconds >= 0 else 0,
            tracking_confidence=detection.tracking_confidence,
            confidence=confidence,
            detected_sign=detected_sign,
            recognition_status=recognition_status,
            translation=self.translator.current_text(),
            landmark_count=detection.landmark_count,
            last_detection_at=time() if detection.has_hand else None,
        )

    def _placeholder_jpeg(self) -> bytes:
        if cv2 is None:
            return b""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (12, 10, 7)
        cv2.rectangle(frame, (24, 24), (1256, 696), (255, 128, 38), 2)
        cv2.putText(
            frame,
            "SIGNOVA CAMERA OFFLINE",
            (360, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 229, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "START CAMERA TO BEGIN LIVE TRACKING",
            (360, 392),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (140, 155, 170),
            1,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        return encoded.tobytes() if ok else b""

    def _release(self) -> None:
        self._release_capture()
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
