from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
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
        self.detector_backend = settings.detector_backend.lower().strip()
        self.model_path = settings.model_path
        self.opencv_model_dir = Path(settings.opencv_model_dir)
        self.person_net = None
        self.model = self._load_model()
        self.latest_detections: List[Detection] = []
        self.latest_timestamp: float = 0.0
        self.model_ready = self.model is not None or self.person_net is not None
        self.model_note = self._build_model_note()

    def _build_model_note(self) -> str:
        if self.detector_backend == "opencv_pretrained":
            if self.person_net is not None:
                return (
                    "OpenCV pretrained mode is active. "
                    "Person detection uses MobileNet-SSD, tomato/pepper quality uses lightweight HSV heuristics."
                )
            return (
                "OpenCV pretrained mode selected, but MobileNet-SSD weights are missing or failed to load. "
                "Check OPENCV_MODEL_DIR and internet access for first-time auto-download."
            )

        model_file = Path(self.model_path)
        if self.model is not None:
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
        if self.detector_backend == "opencv_pretrained":
            self.person_net = self._load_opencv_person_model()
            return None

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

    def _load_opencv_person_model(self) -> Optional[cv2.dnn_Net]:
        self.opencv_model_dir.mkdir(parents=True, exist_ok=True)
        prototxt_path = self.opencv_model_dir / "MobileNetSSD_deploy.prototxt"
        caffemodel_path = self.opencv_model_dir / "MobileNetSSD_deploy.caffemodel"

        if not prototxt_path.exists():
            try:
                urlretrieve(
                    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt",
                    str(prototxt_path),
                )
            except Exception:
                return None

        if not caffemodel_path.exists():
            try:
                urlretrieve(
                    "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel",
                    str(caffemodel_path),
                )
            except Exception:
                return None

        try:
            return cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))
        except Exception:
            return None

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        if self.detector_backend == "opencv_pretrained":
            detections, annotated = self._detect_with_opencv_pretrained(frame)
            self.latest_detections = detections
            self.latest_timestamp = time.time()
            return detections, annotated

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

    def _detect_with_opencv_pretrained(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        detections: List[Detection] = []
        annotated = frame.copy()

        if self.person_net is not None:
            person_dets = self._detect_persons_mobilenet(frame)
            detections.extend(person_dets)

        veggie_dets = self._detect_vegetables_heuristic(frame)
        detections.extend(veggie_dets)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            self._draw_bbox(annotated, (x1, y1, x2, y2), det.label, det.confidence, det.class_name)

        return detections, annotated

    def _detect_persons_mobilenet(self, frame: np.ndarray) -> List[Detection]:
        if self.person_net is None:
            return []

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5,
        )
        self.person_net.setInput(blob)
        result = self.person_net.forward()
        h, w = frame.shape[:2]
        out: List[Detection] = []

        # MobileNet-SSD class id 15 = person.
        for i in range(result.shape[2]):
            confidence = float(result[0, 0, i, 2])
            class_id = int(result[0, 0, i, 1])
            if class_id != 15 or confidence < settings.confidence_threshold:
                continue

            box = result[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int).tolist()
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            out.append(
                Detection(
                    class_name="person",
                    label=TARGET_CLASSES["person"],
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    ts=time.time(),
                )
            )
        return out

    def _detect_vegetables_heuristic(self, frame: np.ndarray) -> List[Detection]:
        """No-training fallback: contour + HSV rules for tomato/pepper quality."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out: List[Detection] = []
        frame_area = float(frame.shape[0] * frame.shape[1])

        red_mask1 = cv2.inRange(hsv, (0, 120, 45), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (165, 120, 45), (179, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        green_mask = cv2.inRange(hsv, (38, 85, 35), (88, 255, 255))

        # Skin suppression avoids classifying faces/hands as vegetables.
        skin_mask = self._build_skin_mask(hsv)
        red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(skin_mask))
        green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(skin_mask))

        out.extend(self._extract_veggie_from_mask(frame, red_mask, skin_mask, "tomato", frame_area))
        out.extend(self._extract_veggie_from_mask(frame, green_mask, skin_mask, "pepper", frame_area))
        return out

    def _build_skin_mask(self, hsv: np.ndarray) -> np.ndarray:
        # Wide skin range to reject face-like regions from veggie candidates.
        skin1 = cv2.inRange(hsv, (0, 20, 60), (25, 180, 255))
        skin2 = cv2.inRange(hsv, (160, 20, 60), (179, 180, 255))
        skin = cv2.bitwise_or(skin1, skin2)
        return cv2.GaussianBlur(skin, (5, 5), 0)

    def _extract_veggie_from_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        skin_mask: np.ndarray,
        veggie_type: str,
        frame_area: float,
    ) -> List[Detection]:
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        min_area = frame_area * settings.veggie_min_area_ratio
        max_area = frame_area * settings.veggie_max_area_ratio

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1.0:
                continue

            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            aspect_ratio = max(w, h) / max(1.0, min(w, h))
            if not self._passes_shape_filter(veggie_type, circularity, aspect_ratio):
                continue

            roi = frame[y : y + h, x : x + w]
            if roi.size == 0:
                continue

            roi_mask = cleaned[y : y + h, x : x + w]
            skin_roi = skin_mask[y : y + h, x : x + w]
            color_pixels = float(np.count_nonzero(roi_mask))
            if color_pixels < 1.0:
                continue
            skin_overlap = float(np.count_nonzero(cv2.bitwise_and(roi_mask, skin_roi))) / color_pixels
            if skin_overlap > settings.skin_overlap_limit:
                continue

            color_purity = color_pixels / float(w * h)
            class_name, confidence = self._classify_quality_from_roi(
                roi, veggie_type, circularity, aspect_ratio, color_purity
            )
            if confidence < settings.confidence_threshold:
                continue
            detections.append(
                Detection(
                    class_name=class_name,
                    label=TARGET_CLASSES[class_name],
                    confidence=confidence,
                    bbox=[x, y, x + w, y + h],
                    ts=time.time(),
                )
            )
        return detections

    def _passes_shape_filter(self, veggie_type: str, circularity: float, aspect_ratio: float) -> bool:
        if veggie_type == "tomato":
            return circularity >= settings.tomato_min_circularity and aspect_ratio <= 1.45
        return circularity >= 0.22 and aspect_ratio >= settings.pepper_min_aspect_ratio

    def _classify_quality_from_roi(
        self,
        roi: np.ndarray,
        veggie_type: str,
        circularity: float,
        aspect_ratio: float,
        color_purity: float,
    ) -> Tuple[str, float]:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat_mean = float(np.mean(hsv[:, :, 1]))
        val_mean = float(np.mean(hsv[:, :, 2]))
        darkness_ratio = float(np.mean(hsv[:, :, 2] < 55))

        good_like = sat_mean > 70 and val_mean > 65 and darkness_ratio < 0.18
        shape_score = min(1.0, (circularity if veggie_type == "tomato" else aspect_ratio / 2.0))
        quality_score = 0.45 * min(1.0, sat_mean / 170.0) + 0.35 * min(1.0, val_mean / 170.0) + 0.20 * shape_score
        confidence = max(0.50, min(0.95, quality_score * 0.95 * (0.65 + 0.35 * color_purity)))

        if veggie_type == "tomato":
            return ("tomato_good", confidence) if good_like else ("tomato_bad", confidence * 0.9)
        return ("pepper_good", confidence) if good_like else ("pepper_bad", confidence * 0.9)

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
