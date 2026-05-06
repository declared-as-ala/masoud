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
    "tomato_good": "Tomato - Good",
    "tomato_bad": "Tomato - Bad",
    "pepper_good": "Pepper - Good",
    "pepper_bad": "Pepper - Bad",
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
        self.model_path = settings.model_path
        self.model = self._load_model()
        self.latest_detections: List[Detection] = []
        self.latest_timestamp: float = 0.0
        self.model_ready = self.model is not None
        self.model_note = self._build_model_note()

    def _build_model_note(self) -> str:
        model_file = Path(self.model_path)
        if self.model_ready:
            return f"Model loaded from {model_file.resolve()}"
        if not model_file.exists():
            return (
                "Model file was not found. Place your .pt/.onnx/.tflite model at MODEL_PATH. "
                "Backend will keep streaming without AI detections until model is available."
            )
        return (
            "Model could not be loaded. Verify dependencies and model format. "
            "Expected labels: tomato_good, tomato_bad, pepper_good, pepper_bad, person."
        )

    def _load_model(self) -> Optional[Any]:
        path = Path(self.model_path)
        if not path.exists():
            return None

        suffix = path.suffix.lower()
        if suffix in [".pt", ".onnx"]:
            if YOLO is None:
                return None
            try:
                return YOLO(str(path))
            except Exception:
                return None

        # Placeholder for TFLite integration.
        # For now, we keep a clean extension point.
        if suffix == ".tflite":
            return None

        return None

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        if self.model is None:
            self.latest_detections = []
            self.latest_timestamp = time.time()
            return [], frame

        detections: List[Detection] = []
        annotated = frame.copy()

        try:
            results = self.model.predict(
                source=frame,
                conf=settings.confidence_threshold,
                verbose=False,
                imgsz=max(settings.frame_width, settings.frame_height),
            )
        except Exception:
            self.latest_detections = []
            self.latest_timestamp = time.time()
            return [], frame

        for result in results:
            names = result.names if hasattr(result, "names") else {}
            for box in result.boxes:
                cls_idx = int(box.cls.item())
                conf = float(box.conf.item())
                class_name = names.get(cls_idx, str(cls_idx)).lower().strip()

                # Flexible mapping to support different training label styles.
                class_name = class_name.replace(" ", "_").replace("-", "_")
                normalized = self._normalize_class(class_name)
                if normalized not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                label = TARGET_CLASSES[normalized]
                detections.append(
                    Detection(
                        class_name=normalized,
                        label=label,
                        confidence=conf,
                        bbox=[x1, y1, x2, y2],
                        ts=time.time(),
                    )
                )
                self._draw_bbox(annotated, (x1, y1, x2, y2), label, conf, normalized)

        self.latest_detections = detections
        self.latest_timestamp = time.time()
        return detections, annotated

    def _normalize_class(self, class_name: str) -> str:
        aliases = {
            "tomato_fresh": "tomato_good",
            "tomato_rotten": "tomato_bad",
            "pepper_fresh": "pepper_good",
            "pepper_rotten": "pepper_bad",
            "felfel_good": "pepper_good",
            "felfel_bad": "pepper_bad",
            "human": "person",
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

        tomato_count = sum(1 for d in self.latest_detections if "tomato" in d.class_name)
        pepper_count = sum(1 for d in self.latest_detections if "pepper" in d.class_name)
        bad_count = sum(1 for d in self.latest_detections if "bad" in d.class_name)
        human_detected = any(d.class_name == "person" for d in self.latest_detections)

        return {
            "timestamp": self.latest_timestamp,
            "model_ready": self.model_ready,
            "model_note": self.model_note,
            "detections": detections,
            "stats": {
                "tomatoes": tomato_count,
                "peppers": pepper_count,
                "bad_vegetables": bad_count,
                "human_detected": human_detected,
            },
        }
