from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import requests

from config import settings


class TelegramAlertService:
    def __init__(self) -> None:
        self.alerts_dir = Path(settings.alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown_seconds = settings.alert_cooldown_seconds
        self.last_alert_ts = 0.0
        self.recent_alerts: List[Dict] = []

    def maybe_send_unauthorized(self, frame, reason: str = "Unknown person") -> Optional[Dict]:
        now = datetime.now()
        now_ts = now.timestamp()
        if now_ts - self.last_alert_ts < self.cooldown_seconds:
            return None

        filename = f"unauthorized_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = self.alerts_dir / filename
        cv2.imwrite(str(image_path), frame)

        alert = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "message": "Unauthorized human detected",
            "image_path": str(image_path),
            "telegram_sent": False,
            "error": None,
        }

        try:
            self._send_telegram(image_path, alert["timestamp"], reason)
            alert["telegram_sent"] = True
        except Exception as exc:
            alert["error"] = str(exc)

        self.last_alert_ts = now_ts
        self.recent_alerts.insert(0, alert)
        self.recent_alerts = self.recent_alerts[:30]
        return alert

    def _send_telegram(self, image_path: Path, ts_text: str, reason: str) -> None:
        token = settings.telegram_bot_token.strip()
        chat_id = settings.telegram_chat_id.strip()
        if not token or not chat_id:
            raise RuntimeError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing in .env")

        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        caption = (
            "⚠️ Unauthorized human detected!\n"
            f"Time: {ts_text}\n"
            f"Status: {reason}"
        )
        with image_path.open("rb") as photo:
            response = requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption},
                files={"photo": photo},
                timeout=15,
            )
        if response.status_code >= 300:
            raise RuntimeError(f"Telegram API failed: {response.status_code} - {response.text}")

    def get_recent_alerts(self) -> List[Dict]:
        return self.recent_alerts
