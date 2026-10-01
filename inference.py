from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import json
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Any

import joblib
import numpy as np

from signova.config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    FEATURE_SCHEMA,
    load_active_vocabulary,
    load_vocabulary,
)


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class ModelManager:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.model: Any | None = None
        self.labels: list[str] = []
        self.metrics: dict[str, Any] = {}
        self.error: str = ""
        self._lock = RLock()

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self.model is not None

    def load(self, model_path: Path | None = None) -> bool:
        with self._lock:
            if model_path is not None:
                self.model_path = Path(model_path)

            if not self.model_path.exists():
                self.model = None
                self.labels = []
                self.metrics = {}
                self.error = f"No trained model found at {self.model_path}."
                return False

            try:
                payload = joblib.load(self.model_path)
                if isinstance(payload, dict):
                    schema = payload.get("feature_schema", {})
                    if schema.get("length") != FEATURE_SCHEMA["length"]:
                        raise ValueError(
                            "Model feature length does not match the current extractor."
                        )
                    self.model = payload["model"]
                    payload_labels = payload.get("labels") or getattr(
                        self.model,
                        "classes_",
                        [],
                    )
                    self.labels = [str(label) for label in payload_labels]
                    self.metrics = payload.get("metrics", {})
                else:
                    self.model = payload
                    self.labels = [str(label) for label in getattr(payload, "classes_", [])]
                    self.metrics = {}
                approved_labels = set(load_vocabulary())
                active_labels = set(load_active_vocabulary())
                unsupported_labels = sorted(set(self.labels) - approved_labels)
                if not self.labels:
                    raise ValueError("Trained model contains no class labels.")
                if unsupported_labels:
                    raise ValueError(
                        "Model contains labels outside the approved vocabulary: "
                        + ", ".join(unsupported_labels)
                    )
                inactive_labels = sorted(set(self.labels) - active_labels)
                if inactive_labels:
                    raise ValueError(
                        "Model contains inactive labels; retrain after restoring or activating them: "
                        + ", ".join(inactive_labels)
                    )
                labels_path = self.model_path.with_name("labels.json")
                if labels_path.exists():
                    sidecar_labels = json.loads(labels_path.read_text(encoding="utf-8")).get("labels", [])
                    if sorted(str(label) for label in sidecar_labels) != sorted(self.labels):
                        raise ValueError("Model labels do not match labels.json.")
                expected_length = getattr(self.model, "n_features_in_", FEATURE_SCHEMA["length"])
                if int(expected_length) != FEATURE_SCHEMA["length"]:
                    raise ValueError(
                        "Model expects "
                        f"{expected_length} features; the current extractor provides "
                        f"{FEATURE_SCHEMA['length']}."
                    )
                self.error = ""
                return True
            except Exception as exc:
                self.model = None
                self.labels = []
                self.metrics = {}
                self.error = str(exc)
                return False

    def invalidate(self, reason: str = "Model needs retraining.") -> None:
        with self._lock:
            self.model = None
            self.labels = []
            self.metrics = {}
            self.error = reason

    def predict(self, features: np.ndarray) -> Prediction | None:
        with self._lock:
            if self.model is None:
                return None
            model = self.model

        vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if not np.any(vector):
            return None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vector)[0]
            classes = [str(label) for label in getattr(model, "classes_", self.labels)]
            best_index = int(np.argmax(probabilities))
            label = classes[best_index]
            confidence = float(probabilities[best_index])
            probability_map = {
                classes[index]: float(value)
                for index, value in enumerate(probabilities)
            }
        else:
            label = str(model.predict(vector)[0])
            confidence = 1.0
            probability_map = {label: confidence}

        return Prediction(label=label, confidence=confidence, probabilities=probability_map)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "online": self.model is not None,
                "path": str(self.model_path),
                "labels": self.labels,
                "metrics": self.metrics,
                "error": self.error,
            }


class PredictionSmoother:
    def __init__(
        self,
        min_confidence: float = CONFIDENCE_THRESHOLD,
        stable_frames: int = 3,
        repeat_cooldown_seconds: float = 3.0,
    ) -> None:
        self.min_confidence = min_confidence
        self.window: deque[str] = deque(maxlen=stable_frames)
        self.low_confidence_frames = 0
        self.max_tolerated_low_confidence_frames = 1
        self.repeat_cooldown_seconds = repeat_cooldown_seconds
        self.last_emitted_label = ""
        self.last_emitted_at = 0.0

    def update(self, prediction: Prediction | None) -> str | None:
        if prediction is None:
            self.reset()
            return None

        if prediction.confidence < self.min_confidence:
            self.low_confidence_frames += 1
            if self.low_confidence_frames > self.max_tolerated_low_confidence_frames:
                self.reset()
            return None

        self.low_confidence_frames = 0

        self.window.append(prediction.label)
        if len(self.window) < self.window.maxlen:
            return None

        label, count = Counter(self.window).most_common(1)[0]
        if count != self.window.maxlen:
            return None

        now = monotonic()
        is_repeat = label == self.last_emitted_label
        if is_repeat and now - self.last_emitted_at < self.repeat_cooldown_seconds:
            return None

        self.last_emitted_label = label
        self.last_emitted_at = now
        return label

    def reset(self) -> None:
        self.window.clear()
        self.low_confidence_frames = 0
        self.last_emitted_label = ""
        self.last_emitted_at = 0.0

    def status(self) -> dict[str, Any]:
        return {
            "candidate": self.window[-1] if self.window else "",
            "stable_frames": len(self.window),
            "required_frames": self.window.maxlen or 0,
        }
