from __future__ import annotations

import json
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
MODELS_DIR = DATA_DIR / "models"
HISTORY_DIR = DATA_DIR / "history"
VOCABULARY_PATH = DATA_DIR / "vocabulary.json"
ACTIVE_VOCABULARY_PATH = DATA_DIR / "active_vocabulary.json"
DELETED_DATASET_DIR = DATASET_DIR / "deleted"
DEFAULT_MODEL_PATH = MODELS_DIR / "signova_model.joblib"

DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_FPS = 30
AI_PROCESSING_WIDTH = 640
AI_PROCESSING_HEIGHT = 360
CONFIDENCE_THRESHOLD = 0.70

FEATURE_SCHEMA = {
    "name": "mediapipe_holistic_pose_hands_v1",
    "length": 258,
    "pose_landmarks": 33,
    "pose_values": ["x", "y", "z", "visibility"],
    "left_hand_landmarks": 21,
    "right_hand_landmarks": 21,
    "hand_values": ["x", "y", "z"],
}

DEFAULT_VOCABULARY = [
    "HELLO",
    "GOODBYE",
    "YES",
    "NO",
    "PLEASE",
    "THANK_YOU",
    "SORRY",
    "HELP",
    "GOOD",
    "BAD",
    "LOVE",
    "I",
    "YOU",
    "ME",
    "MY",
    "NAME",
    "WHAT",
    "WHERE",
    "WHO",
    "HOW",
    "WANT",
    "NEED",
    "EAT",
    "DRINK",
    "MORE",
    "STOP",
    "GO",
    "COME",
    "FRIEND",
    "FAMILY",
    "HAPPY",
    "SAD",
    "OKAY",
    "I_LOVE_YOU",
]

TRANSLATION_MAP = {
    "HELLO": "Hello",
    "GOODBYE": "Goodbye",
    "THANK_YOU": "Thank you",
    "PLEASE": "Please",
    "YES": "Yes",
    "NO": "No",
    "SORRY": "Sorry",
    "HELP": "Help",
    "GOOD": "Good",
    "BAD": "Bad",
    "LOVE": "Love",
    "I": "I",
    "YOU": "You",
    "ME": "Me",
    "MY": "My",
    "NAME": "Name",
    "WHAT": "What",
    "WHERE": "Where",
    "WHO": "Who",
    "HOW": "How",
    "WANT": "Want",
    "NEED": "Need",
    "EAT": "Eat",
    "DRINK": "Drink",
    "MORE": "More",
    "STOP": "Stop",
    "GO": "Go",
    "COME": "Come",
    "FRIEND": "Friend",
    "FAMILY": "Family",
    "HAPPY": "Happy",
    "SAD": "Sad",
    "OKAY": "Okay",
    "I_LOVE_YOU": "I love you",
}


def ensure_project_dirs() -> None:
    for path in (DATA_DIR, DATASET_DIR, MODELS_DIR, HISTORY_DIR, DELETED_DATASET_DIR):
        path.mkdir(parents=True, exist_ok=True)

    if not VOCABULARY_PATH.exists():
        save_vocabulary(DEFAULT_VOCABULARY)


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_").upper()
    if not cleaned:
        raise ValueError("Sign label cannot be empty.")
    return cleaned


def load_vocabulary() -> list[str]:
    ensure_project_dirs()
    try:
        payload = json.loads(VOCABULARY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        payload = {"labels": DEFAULT_VOCABULARY}
        save_vocabulary(DEFAULT_VOCABULARY)

    labels = payload.get("labels", DEFAULT_VOCABULARY)
    return sorted({sanitize_label(label) for label in labels})


def load_active_vocabulary() -> list[str]:
    ensure_project_dirs()
    if not ACTIVE_VOCABULARY_PATH.exists():
        active = [
            label
            for label in load_vocabulary()
            if (DATASET_DIR / label).exists() and any((DATASET_DIR / label).glob("*.npz"))
        ]
        save_active_vocabulary(active)
        return sorted(active)
    try:
        payload = json.loads(ACTIVE_VOCABULARY_PATH.read_text(encoding="utf-8"))
        return sorted({sanitize_label(label) for label in payload.get("labels", [])})
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_active_vocabulary(labels: list[str]) -> None:
    normalized = sorted({sanitize_label(label) for label in labels})
    unknown = set(normalized) - set(load_vocabulary())
    if unknown:
        raise ValueError("Labels outside approved vocabulary: " + ", ".join(sorted(unknown)))
    ACTIVE_VOCABULARY_PATH.write_text(
        json.dumps({"labels": normalized}, indent=2),
        encoding="utf-8",
    )


def save_vocabulary(labels: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    normalized = sorted({sanitize_label(label) for label in labels})
    VOCABULARY_PATH.write_text(
        json.dumps({"labels": normalized}, indent=2),
        encoding="utf-8",
    )
