import os
from dataclasses import dataclass


@dataclass
class Settings:
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    detector_backend: str = os.getenv("DETECTOR_BACKEND", "opencv_pretrained")
    model_path: str = os.getenv("MODEL_PATH", "./models/vegetable_quality.pt")
    opencv_model_dir: str = os.getenv("OPENCV_MODEL_DIR", "./models/opencv")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
    frame_width: int = int(os.getenv("FRAME_WIDTH", "640"))
    frame_height: int = int(os.getenv("FRAME_HEIGHT", "480"))
    detection_interval: int = int(os.getenv("DETECTION_INTERVAL", "3"))
    jpeg_quality: int = int(os.getenv("JPEG_QUALITY", "80"))
    ws_push_interval_ms: int = int(os.getenv("WS_PUSH_INTERVAL_MS", "500"))


settings = Settings()
