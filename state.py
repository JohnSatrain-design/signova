from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from time import time
from typing import Any


@dataclass
class RuntimeState:
    camera_online: bool = False
    ai_vision_active: bool = False
    hand_tracking_active: bool = False
    model_online: bool = False
    voice_engine_ready: bool = False
    fps: float = 0.0
    ai_fps: float = 0.0
    inference_ms: float = 0.0
    frame_buffer_size: int = 0
    confidence: float = 0.0
    tracking_confidence: float = 0.0
    detected_sign: str = ""
    recognition_status: str = "WAITING FOR SIGN"
    prediction_candidate: str = ""
    stable_frames: int = 0
    required_stable_frames: int = 3
    translation: str = ""
    landmark_count: int = 0
    camera_error: str = ""
    model_error: str = ""
    voice_error: str = ""
    collection_active: bool = False
    collection_label: str = ""
    collection_saved: int = 0
    training_status: dict[str, Any] = field(default_factory=dict)
    last_detection_at: float | None = None
    last_speech_at: float | None = None
    updated_at: float = field(default_factory=time)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            self.updated_at = time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "camera_online": self.camera_online,
                "ai_vision_active": self.ai_vision_active,
                "hand_tracking_active": self.hand_tracking_active,
                "model_online": self.model_online,
                "voice_engine_ready": self.voice_engine_ready,
                "fps": round(self.fps, 1),
                "ai_fps": round(self.ai_fps, 1),
                "inference_ms": round(self.inference_ms, 1),
                "frame_buffer_size": self.frame_buffer_size,
                "confidence": round(self.confidence, 4),
                "tracking_confidence": round(self.tracking_confidence, 4),
                "detected_sign": self.detected_sign,
                "recognition_status": self.recognition_status,
                "prediction_candidate": self.prediction_candidate,
                "stable_frames": self.stable_frames,
                "required_stable_frames": self.required_stable_frames,
                "translation": self.translation,
                "landmark_count": self.landmark_count,
                "camera_error": self.camera_error,
                "model_error": self.model_error,
                "voice_error": self.voice_error,
                "collection_active": self.collection_active,
                "collection_label": self.collection_label,
                "collection_saved": self.collection_saved,
                "training_status": self.training_status,
                "last_detection_at": self.last_detection_at,
                "last_speech_at": self.last_speech_at,
                "updated_at": self.updated_at,
            }
