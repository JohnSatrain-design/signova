from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import numpy as np

from signova.config import (
    DATASET_DIR,
    DELETED_DATASET_DIR,
    FEATURE_SCHEMA,
    load_active_vocabulary,
    load_vocabulary,
    save_active_vocabulary,
    sanitize_label,
)
from signova.core.features import FEATURE_LENGTH


class DatasetManager:
    def __init__(self, dataset_dir: Path = DATASET_DIR) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dataset_dir / "index.csv"
        self._lock = RLock()

    def vocabulary(self) -> list[str]:
        return load_active_vocabulary()

    def deleted_vocabulary(self) -> list[str]:
        return sorted(path.name for path in DELETED_DATASET_DIR.iterdir() if path.is_dir())

    def add_label(self, label: str) -> str:
        normalized = sanitize_label(label)
        if normalized not in set(load_vocabulary()):
            raise ValueError(
                f"Label {normalized} is not in the administrator-approved vocabulary."
            )
        (self.dataset_dir / normalized).mkdir(parents=True, exist_ok=True)
        active = set(load_active_vocabulary())
        active.add(normalized)
        save_active_vocabulary(list(active))
        return normalized

    def sample_counts(self) -> dict[str, int]:
        counts = {label: 0 for label in load_active_vocabulary()}
        for sample in self.dataset_dir.glob("*/*.npz"):
            if sample.parent.name not in counts:
                continue
            counts[sample.parent.name] = counts.get(sample.parent.name, 0) + 1
        return dict(sorted(counts.items()))

    def save_sample(
        self,
        label: str,
        features: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        normalized = self.add_label(label)
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape[0] != FEATURE_LENGTH:
            raise ValueError(f"Expected {FEATURE_LENGTH} features, got {vector.shape[0]}.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sample_id = uuid4().hex[:10]
        sign_dir = self.dataset_dir / normalized
        sign_dir.mkdir(parents=True, exist_ok=True)
        path = sign_dir / f"{timestamp}_{sample_id}.npz"

        payload_metadata = metadata or {}
        with self._lock:
            np.savez_compressed(
                path,
                features=vector,
                label=normalized,
                created_at=datetime.now().isoformat(timespec="seconds"),
                feature_schema=json.dumps(FEATURE_SCHEMA),
                metadata=json.dumps(payload_metadata),
            )
            self._append_index(path, normalized, payload_metadata)

        return path

    def load_samples(
        self,
        labels: list[str] | None = None,
        min_samples_per_class: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, list[Path]]:
        allowed = {sanitize_label(label) for label in labels} if labels else None
        rows: list[np.ndarray] = []
        targets: list[str] = []
        paths: list[Path] = []

        allowed_active = set(load_active_vocabulary())
        for sample_path in sorted(self.dataset_dir.glob("*/*.npz")):
            label = sample_path.parent.name
            if label not in allowed_active:
                continue
            if allowed is not None and label not in allowed:
                continue

            try:
                sample = np.load(sample_path, allow_pickle=False)
                features = np.asarray(sample["features"], dtype=np.float32)
            except Exception:
                continue

            if features.shape[0] != FEATURE_LENGTH:
                continue

            rows.append(features)
            targets.append(label)
            paths.append(sample_path)

        if not rows:
            return np.empty((0, FEATURE_LENGTH), dtype=np.float32), np.asarray([]), []

        counts = Counter(targets)
        keep = {label for label, count in counts.items() if count >= min_samples_per_class}
        filtered = [
            (row, target, path)
            for row, target, path in zip(rows, targets, paths)
            if target in keep
        ]

        if not filtered:
            return np.empty((0, FEATURE_LENGTH), dtype=np.float32), np.asarray([]), []

        rows, targets, paths = zip(*filtered)
        return np.vstack(rows), np.asarray(targets), list(paths)

    def archive_label(self, label: str) -> str:
        normalized = sanitize_label(label)
        source = self.dataset_dir / normalized
        if not source.exists():
            raise ValueError(f"Active sign {normalized} does not exist.")
        DELETED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        destination = DELETED_DATASET_DIR / normalized
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(source), str(destination))
        save_active_vocabulary([item for item in load_active_vocabulary() if item != normalized])
        return normalized

    def restore_label(self, label: str) -> str:
        normalized = sanitize_label(label)
        source = DELETED_DATASET_DIR / normalized
        if not source.exists():
            raise ValueError(f"Deleted sign {normalized} does not exist.")
        destination = self.dataset_dir / normalized
        if destination.exists():
            raise ValueError(f"Active sign {normalized} already exists.")
        shutil.move(str(source), str(destination))
        save_active_vocabulary(load_active_vocabulary() + [normalized])
        return normalized

    def permanently_delete_label(self, label: str) -> str:
        normalized = sanitize_label(label)
        target = DELETED_DATASET_DIR / normalized
        if target.exists():
            shutil.rmtree(target)
        return normalized

    def reset_all(self) -> None:
        for path in self.dataset_dir.iterdir():
            if path.name != "README.md" and path.name != "deleted":
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        if DELETED_DATASET_DIR.exists():
            for path in DELETED_DATASET_DIR.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        save_active_vocabulary([])

    def _append_index(self, path: Path, label: str, metadata: dict[str, Any]) -> None:
        new_file = not self.index_path.exists()
        with self.index_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["created_at", "label", "path", "metadata"],
            )
            if new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "label": label,
                    "path": str(path.relative_to(self.dataset_dir)),
                    "metadata": json.dumps(metadata),
                }
            )
