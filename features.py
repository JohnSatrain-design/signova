from __future__ import annotations

from typing import Any

import numpy as np

from signova.config import FEATURE_SCHEMA


POSE_LANDMARK_COUNT = FEATURE_SCHEMA["pose_landmarks"]
HAND_LANDMARK_COUNT = FEATURE_SCHEMA["left_hand_landmarks"]
FEATURE_LENGTH = FEATURE_SCHEMA["length"]


def _flatten_landmarks(
    landmark_list: Any | None,
    expected_count: int,
    include_visibility: bool,
) -> np.ndarray:
    values: list[float] = []

    if landmark_list is not None:
        for landmark in list(landmark_list.landmark)[:expected_count]:
            values.extend([landmark.x, landmark.y, landmark.z])
            if include_visibility:
                values.append(getattr(landmark, "visibility", 0.0))

    values_per_landmark = 4 if include_visibility else 3
    expected_length = expected_count * values_per_landmark
    if len(values) < expected_length:
        values.extend([0.0] * (expected_length - len(values)))

    return np.asarray(values[:expected_length], dtype=np.float32)


def extract_holistic_features(results: Any) -> np.ndarray:
    pose = _flatten_landmarks(results.pose_landmarks, POSE_LANDMARK_COUNT, True)
    left_hand = _flatten_landmarks(results.left_hand_landmarks, HAND_LANDMARK_COUNT, False)
    right_hand = _flatten_landmarks(results.right_hand_landmarks, HAND_LANDMARK_COUNT, False)
    feature_vector = np.concatenate([pose, left_hand, right_hand]).astype(np.float32)

    if feature_vector.shape[0] != FEATURE_LENGTH:
        raise ValueError(
            f"Expected feature length {FEATURE_LENGTH}, got {feature_vector.shape[0]}."
        )

    return feature_vector


def count_detected_landmarks(results: Any) -> int:
    total = 0
    for item in (
        results.pose_landmarks,
        results.left_hand_landmarks,
        results.right_hand_landmarks,
    ):
        if item is not None:
            total += len(item.landmark)
    return total


def has_hand_landmarks(results: Any) -> bool:
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None


def estimate_tracking_confidence(results: Any) -> float:
    score = 0.0
    if results.left_hand_landmarks is not None:
        score += 0.42
    if results.right_hand_landmarks is not None:
        score += 0.42
    if results.pose_landmarks is not None:
        score += 0.16
    return min(score, 1.0)
