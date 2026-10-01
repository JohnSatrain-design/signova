from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from signova.core.features import (
    count_detected_landmarks,
    estimate_tracking_confidence,
    extract_holistic_features,
    has_hand_landmarks,
)

try:
    import cv2
    import mediapipe as mp
except Exception as exc:  # pragma: no cover - surfaced through app status.
    cv2 = None
    mp = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""


@dataclass
class LandmarkDetection:
    frame: np.ndarray
    features: np.ndarray
    has_hand: bool
    has_pose: bool
    tracking_confidence: float
    landmark_count: int
    landmark_points: dict[str, list[tuple[float, float]]]


class LandmarkDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
    ) -> None:
        if cv2 is None or mp is None:
            raise RuntimeError(
                "OpenCV and MediaPipe are required for camera tracking. "
                f"Import error: {IMPORT_ERROR}"
            )

        self._mp_holistic = mp.solutions.holistic
        self._drawing = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame: np.ndarray) -> LandmarkDetection:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._holistic.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()
        self._draw_landmarks(annotated, results)

        features = extract_holistic_features(results)
        return LandmarkDetection(
            frame=annotated,
            features=features,
            has_hand=has_hand_landmarks(results),
            has_pose=results.pose_landmarks is not None,
            tracking_confidence=estimate_tracking_confidence(results),
            landmark_count=count_detected_landmarks(results),
            landmark_points={
                "pose": self._points(results.pose_landmarks),
                "left_hand": self._points(results.left_hand_landmarks),
                "right_hand": self._points(results.right_hand_landmarks),
            },
        )

    def close(self) -> None:
        self._holistic.close()

    def _draw_landmarks(self, frame: np.ndarray, results: object) -> None:
        if results.pose_landmarks:
            self._drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self._mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self._styles.get_default_pose_landmarks_style(),
            )

        hand_spec = self._drawing.DrawingSpec(color=(0, 229, 255), thickness=2, circle_radius=2)
        connection_spec = self._drawing.DrawingSpec(color=(70, 130, 255), thickness=2)

        if results.left_hand_landmarks:
            self._drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self._mp_holistic.HAND_CONNECTIONS,
                hand_spec,
                connection_spec,
            )
        if results.right_hand_landmarks:
            self._drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self._mp_holistic.HAND_CONNECTIONS,
                hand_spec,
                connection_spec,
            )

    @staticmethod
    def _points(landmarks: object | None) -> list[tuple[float, float]]:
        if landmarks is None:
            return []
        return [(float(point.x), float(point.y)) for point in landmarks.landmark]

    @staticmethod
    def draw_tracking_overlay(
        frame: np.ndarray,
        landmark_points: dict[str, list[tuple[float, float]]],
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        if mp is None or cv2 is None:
            return frame
        hand_connections = tuple(mp.solutions.holistic.HAND_CONNECTIONS)
        pose_connections = tuple(mp.solutions.holistic.POSE_CONNECTIONS)
        LandmarkDetector._draw_point_connections(
            frame,
            landmark_points.get("pose", []),
            pose_connections,
            (180, 180, 180),
            (100, 190, 255),
            width,
            height,
        )
        for key in ("left_hand", "right_hand"):
            LandmarkDetector._draw_point_connections(
                frame,
                landmark_points.get(key, []),
                hand_connections,
                (0, 229, 255),
                (70, 130, 255),
                width,
                height,
            )
        return frame

    @staticmethod
    def _draw_point_connections(
        frame: np.ndarray,
        points: list[tuple[float, float]],
        connections: tuple[tuple[int, int], ...],
        point_color: tuple[int, int, int],
        line_color: tuple[int, int, int],
        width: int,
        height: int,
    ) -> None:
        if not points:
            return
        pixels = [(int(x * width), int(y * height)) for x, y in points]
        for start, end in connections:
            if start < len(pixels) and end < len(pixels):
                cv2.line(frame, pixels[start], pixels[end], line_color, 2, cv2.LINE_AA)
        for pixel in pixels:
            cv2.circle(frame, pixel, 3, point_color, -1, cv2.LINE_AA)
