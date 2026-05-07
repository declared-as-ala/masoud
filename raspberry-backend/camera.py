from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import settings


class CameraService:
    def __init__(self) -> None:
        self._capture: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._annotated_frame: Optional[np.ndarray] = None
        self._last_error: Optional[str] = None
        self._frame_counter = 0
        self._last_start_attempt_ts = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._last_start_attempt_ts = time.time()

        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass

        self._capture = cv2.VideoCapture(settings.camera_index, cv2.CAP_DSHOW)
        if not self._capture or not self._capture.isOpened():
            self._capture = cv2.VideoCapture(settings.camera_index)

        if not self._capture or not self._capture.isOpened():
            self._last_error = (
                f"Camera index {settings.camera_index} not found. "
                "Check camera connection and CAMERA_INDEX."
            )
            self._running = False
            return

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
        self._last_error = None
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._capture is not None:
            self._capture.release()

    def _reader_loop(self) -> None:
        while self._running:
            if self._capture is None:
                time.sleep(0.05)
                continue

            ok, frame = self._capture.read()
            if not ok:
                self._last_error = "Failed to read from camera stream."
                time.sleep(0.05)
                continue

            resized = cv2.resize(frame, (settings.frame_width, settings.frame_height))
            with self._lock:
                self._latest_frame = resized
                if self._annotated_frame is None:
                    self._annotated_frame = resized.copy()
                self._frame_counter += 1

            time.sleep(0.005)

    def is_ready(self) -> bool:
        return self._running and self._latest_frame is not None

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._annotated_frame is None:
                return None
            return self._annotated_frame.copy()

    def set_annotated_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._annotated_frame = frame.copy()

    def get_frame_counter(self) -> int:
        with self._lock:
            return self._frame_counter

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    def can_retry_start(self, min_retry_seconds: float = 1.5) -> bool:
        if self._running:
            return False
        return (time.time() - self._last_start_attempt_ts) >= min_retry_seconds
