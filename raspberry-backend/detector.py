from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import settings

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


TARGET_CLASSES = {
    "tomato": "Tomato detected",
    "pepper": "Pepper detected",
    "person": "Human detected",
}


@dataclass
class Detection:
    class_name: str
    label: str
    confidence: float
    bbox: List[int]
    ts: float


class DetectionService:
    def __init__(self) -> None:
        self.person_model = self._load_model(settings.person_model)
        self.vegetable_model = self._load_vegetable_model()
        self.latest_detections: List[Detection] = []
        self.latest_timestamp: float = 0.0
        self.model_ready = self.person_model is not None
        self.model_note = self._build_model_note()

    def _build_model_note(self) -> str:
        vegetable_path = Path(settings.vegetable_model_path)
        if self.person_model is None:
            return "Person model failed to load. Check PERSON_MODEL and Ultralytics installation."
        if self.vegetable_model is None and not vegetable_path.exists():
            return (
                "Vegetable model not found. Place your pretrained model at "
                f"{settings.vegetable_model_path}"
            )
        if self.vegetable_model is None:
            return "Vegetable model exists but failed to load. Keep running person detection only."
        return "Person and vegetable pretrained models are loaded."

    def _load_model(self) -> Optional[Any]:
        return None

    def _load_model(self, model_path: str) -> Optional[Any]:
        if YOLO is None:
            return None

        try:
            return YOLO(model_path)
        except Exception:
            return None

    def _load_vegetable_model(self) -> Optional[Any]:
        veg_path = Path(settings.vegetable_model_path)
        if not veg_path.exists():
            return None

        try:
            return YOLO(str(veg_path)) if YOLO is not None else None
        except Exception:
            return None

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        detections: List[Detection] = []
        annotated = frame.copy()
        detections.extend(self._detect_person(frame))
        detections.extend(self._detect_vegetables(frame))
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            self._draw_bbox(
                annotated,
                (x1, y1, x2, y2),
                detection.label,
                detection.confidence,
                detection.class_name,
            )

        self.latest_detections = detections
        self.latest_timestamp = time.time()
        return detections, annotated

    def _detect_person(self, frame: np.ndarray) -> List[Detection]:
        if self.person_model is None:
            return []
        return self._run_yolo(frame, self.person_model, allowed_classes={"person"})

    def _detect_vegetables(self, frame: np.ndarray) -> List[Detection]:
        if self.vegetable_model is None:
            return []
        return self._run_yolo(frame, self.vegetable_model, allowed_classes={"tomato", "pepper", "felfel"})

    def _run_yolo(self, frame: np.ndarray, model: Any, allowed_classes: set[str]) -> List[Detection]:
        try:
            results = model.predict(
                source=frame,
                conf=settings.confidence_threshold,
                verbose=False,
                imgsz=max(settings.frame_width, settings.frame_height),
            )
        except Exception:
            return []

        out: List[Detection] = []
        for result in results:
            names = result.names if hasattr(result, "names") else {}
            for box in result.boxes:
                cls_idx = int(box.cls.item())
                conf = float(box.conf.item())
                class_name = str(names.get(cls_idx, cls_idx)).lower().strip().replace("-", "_")
                normalized = self._normalize_class(class_name)
                if normalized not in allowed_classes:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                label = TARGET_CLASSES.get(normalized, normalized.title())
                out.append(
                    Detection(
                        class_name=normalized,
                        label=label,
                        confidence=conf,
                        bbox=[x1, y1, x2, y2],
                        ts=time.time(),
                    )
                )
        return out

    def _normalize_class(self, class_name: str) -> str:
        aliases = {
            "human": "person",
            "felfel": "pepper",
        }
        return aliases.get(class_name, class_name)

    def _draw_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str,
        conf: float,
        class_name: str,
    ) -> None:
        x1, y1, x2, y2 = bbox
        color = (0, 220, 0)
        if "bad" in class_name:
            color = (0, 0, 220)
        if class_name == "person":
            color = (220, 180, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} - {int(conf * 100)}%"
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    def get_snapshot(self) -> Dict[str, Any]:
        detections = [
            {
                "class_name": d.class_name,
                "label": d.label,
                "confidence": round(d.confidence, 4),
                "bbox": d.bbox,
                "timestamp": d.ts,
            }
            for d in self.latest_detections
        ]

        tomato_count = sum(1 for d in self.latest_detections if d.class_name == "tomato")
        pepper_count = sum(1 for d in self.latest_detections if d.class_name == "pepper")
        human_detected = any(d.class_name == "person" for d in self.latest_detections)

        return {
            "timestamp": self.latest_timestamp,
            "model_ready": self.model_ready,
            "model_note": self.model_note,
            "detections": detections,
            "stats": {
                "tomatoes": tomato_count,
                "peppers": pepper_count,
                "human_detected": human_detected,
            },
        }
