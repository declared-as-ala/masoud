from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import settings

try:
    from deepface import DeepFace
except Exception:
    DeepFace = None


@dataclass
class FaceMatch:
    authorized: bool
    name: Optional[str]
    score: Optional[float]


class FaceAuthService:
    def __init__(self) -> None:
        self.enabled = settings.face_auth_enabled and DeepFace is not None
        self.db_dir = Path(settings.faces_authorized_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.known_faces = self._load_authorized_faces()
        self.status_note = self._build_status_note()

    def _build_status_note(self) -> str:
        if not settings.face_auth_enabled:
            return "Face authorization disabled by FACE_AUTH_ENABLED."
        if DeepFace is None:
            return "DeepFace not available. Install dependencies for face authorization."
        if not self.known_faces:
            return f"No authorized faces found in {self.db_dir}."
        return f"Face authorization ready with {len(self.known_faces)} authorized identities."

    def _load_authorized_faces(self) -> List[Dict]:
        if DeepFace is None:
            return []
        known: List[Dict] = []
        for image_path in sorted(self.db_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            embedding = self._extract_embedding(str(image_path))
            if embedding is None:
                continue
            known.append({"name": image_path.stem, "embedding": embedding})
        return known

    def _extract_embedding(self, image_input) -> Optional[np.ndarray]:
        if DeepFace is None:
            return None
        try:
            vectors = DeepFace.represent(
                img_path=image_input,
                model_name="SFace",
                detector_backend="opencv",
                enforce_detection=False,
            )
            if not vectors:
                return None
            return np.array(vectors[0]["embedding"], dtype=np.float32)
        except Exception:
            return None

    def recognize(self, frame: np.ndarray) -> FaceMatch:
        if not self.enabled or not self.known_faces:
            return FaceMatch(authorized=False, name=None, score=None)

        face_roi = self._extract_face_roi(frame)
        if face_roi is None:
            return FaceMatch(authorized=False, name=None, score=None)

        query_embedding = self._extract_embedding(face_roi)
        if query_embedding is None:
            return FaceMatch(authorized=False, name=None, score=None)

        best_name = None
        best_distance = 1e9
        for item in self.known_faces:
            dist = float(np.linalg.norm(query_embedding - item["embedding"]))
            if dist < best_distance:
                best_distance = dist
                best_name = item["name"]

        if best_name is not None and best_distance <= settings.face_match_threshold:
            confidence = max(0.0, 1.0 - (best_distance / max(0.001, settings.face_match_threshold * 2.0)))
            return FaceMatch(authorized=True, name=best_name, score=confidence)

        return FaceMatch(authorized=False, name=None, score=None)

    def _extract_face_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        padding = 12
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        return frame[y1:y2, x1:x2]
