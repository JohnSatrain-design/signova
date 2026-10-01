from __future__ import annotations

from collections import deque
from datetime import datetime
from threading import RLock
from typing import Any

from signova.config import TRANSLATION_MAP


class TranslationProcessor:
    def __init__(self, max_history: int = 80) -> None:
        self._words: list[str] = []
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._lock = RLock()

    def accept_sign(self, label: str, confidence: float) -> dict[str, Any] | None:
        label = label.upper()
        word = TRANSLATION_MAP.get(label, label.replace("_", " ").title())

        with self._lock:
            if label == "CLEAR":
                self._words.clear()
            elif label == "DELETE":
                if self._words:
                    self._words.pop()
            elif label == "SPACE":
                if self._words and self._words[-1] != "":
                    self._words.append("")
            elif word:
                self._words.append(word)

            item = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "sign": label,
                "translation": word if label not in {"DELETE", "CLEAR"} else label.title(),
                "confidence": round(confidence, 4),
                "sentence": self._sentence_unlocked(),
            }
            self._history.appendleft(item)
            return item

    def clear(self) -> None:
        with self._lock:
            self._words.clear()
            self._history.clear()

    def current_text(self) -> str:
        with self._lock:
            return self._sentence_unlocked()

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def _sentence_unlocked(self) -> str:
        parts: list[str] = []
        for word in self._words:
            if word == "":
                continue
            parts.append(word)
        return " ".join(parts).strip()
