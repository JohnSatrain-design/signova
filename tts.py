from __future__ import annotations

from queue import Empty, Queue
from threading import Event, RLock, Thread
from time import time

try:
    import pyttsx3
except Exception as exc:  # pragma: no cover - surfaced through app status.
    pyttsx3 = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""


class TextToSpeechEngine:
    def __init__(self) -> None:
        self.ready = False
        self.error = ""
        self.last_spoken_at: float | None = None
        self._engine = None
        self._lock = RLock()
        self._queue: Queue[str | None] = Queue(maxsize=4)
        self._stop_event = Event()
        self._initialized = Event()
        self._worker = Thread(target=self._worker_loop, name="signova-tts", daemon=True)
        self._worker.start()
        self._initialized.wait(timeout=2.0)

    def _initialize(self) -> None:
        if pyttsx3 is None:
            self.error = f"pyttsx3 is unavailable: {IMPORT_ERROR}"
            return

        try:
            self._engine = pyttsx3.init()
            self.ready = True
            self.error = ""
        except Exception as exc:
            self.ready = False
            self.error = str(exc)

    def speak(self, text: str) -> bool:
        text = text.strip()
        if not text or not self.ready:
            return False

        try:
            self._queue.put_nowait(text)
        except Exception:
            return False
        self.last_spoken_at = time()
        return True

    def _worker_loop(self) -> None:
        # pyttsx3 may create a Tk-backed driver on Windows. Initialize and use it
        # on this same thread so Tk objects never cross thread boundaries.
        self._initialize()
        self._initialized.set()
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except Empty:
                continue
            if text is None:
                self._queue.task_done()
                break
            try:
                self._speak_blocking(text)
            finally:
                self._queue.task_done()

    def _speak_blocking(self, text: str) -> None:
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                self.ready = False
                self.error = str(exc)
