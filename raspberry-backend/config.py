from dataclasses import dataclass


@dataclass
class Settings:
    # Hardcoded runtime configuration (no .env usage).
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    detection_interval_frames: int = 5
    confidence_threshold: float = 0.5

    person_model: str = "yolov8n.pt"
    vegetable_model_path: str = "best.onnx"

    faces_authorized_dir: str = "faces_authorized"
    face_auth_enabled: bool = True
    face_match_threshold: float = 0.35
    face_recheck_interval_frames: int = 8

    alerts_dir: str = "alerts"
    alert_cooldown_seconds: int = 60
    telegram_bot_token: str = "8635333240:AAFMpwmK7kMC0FDAEqo6t3We3h0PopXz0_0"
    telegram_chat_id: str = "8394078292"

    jpeg_quality: int = 80


settings = Settings()
