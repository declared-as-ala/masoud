from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from camera import CameraService
from config import settings
from detector import DetectionService
from face_auth import FaceAuthService
from telegram_alert import TelegramAlertService

app = FastAPI(title="Raspberry Pi Vision Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

camera = CameraService()
detector = DetectionService()
face_auth = FaceAuthService()
telegram_alert = TelegramAlertService()
detector_task: asyncio.Task | None = None
latest_state: Dict[str, Any] = {
    "timestamp": "",
    "objects": [],
    "human": {"detected": False, "authorized": None, "name": None},
    "last_alert": None,
}


async def detection_loop() -> None:
    last_processed = -1
    face_recheck_counter = 0
    last_face_state = {"authorized": None, "name": None}

    while True:
        await asyncio.sleep(0.01)
        if not camera.is_ready():
            continue

        frame_counter = camera.get_frame_counter()
        if frame_counter == last_processed:
            continue

        if frame_counter % settings.detection_interval_frames != 0:
            continue

        frame = camera.get_frame()
        if frame is None:
            continue

        detections, annotated = detector.detect(frame)
        objects = [
            {"label": d.class_name, "confidence": round(d.confidence, 4), "bbox": d.bbox}
            for d in detections
        ]
        human_dets = [d for d in detections if d.class_name == "person"]
        human_detected = len(human_dets) > 0
        human_payload = {"detected": human_detected, "authorized": None, "name": None}

        if human_detected:
            face_recheck_counter += 1
            if face_recheck_counter >= settings.face_recheck_interval_frames:
                face_recheck_counter = 0
                match = face_auth.recognize(frame)
                last_face_state = {"authorized": match.authorized, "name": match.name}

            human_payload["authorized"] = last_face_state["authorized"]
            human_payload["name"] = last_face_state["name"]

            if human_payload["authorized"]:
                label = f"Authorized: {human_payload['name']}"
                cv2.putText(annotated, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 0), 2)
            elif human_payload["authorized"] is False:
                cv2.putText(
                    annotated, "Unauthorized person", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2
                )
                alert = telegram_alert.maybe_send_unauthorized(annotated)
                if alert:
                    latest_state["last_alert"] = alert["timestamp"]
        else:
            last_face_state = {"authorized": None, "name": None}
            human_payload = {"detected": False, "authorized": None, "name": None}

        latest_state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        latest_state["objects"] = objects
        latest_state["human"] = human_payload
        latest_state["last_alert"] = latest_state.get("last_alert")

        camera.set_annotated_frame(annotated)
        last_processed = frame_counter


def generate_mjpeg_stream() -> AsyncGenerator[bytes, None]:
    while True:
        frame = camera.get_annotated_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality],
        )
        if not ok:
            time.sleep(0.02)
            continue

        frame_bytes = encoded.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.03)


@app.on_event("startup")
async def startup_event() -> None:
    global detector_task
    camera.start()
    detector_task = asyncio.create_task(detection_loop())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if detector_task:
        detector_task.cancel()
    camera.stop()


@app.get("/health")
def health() -> JSONResponse:
    camera_ready = camera.is_ready()
    return JSONResponse(
        {
            "status": "ok" if camera_ready else "degraded",
            "camera_ready": camera_ready,
            "camera_error": camera.get_last_error(),
            "person_model_ready": detector.person_model is not None,
            "vegetable_model_ready": detector.vegetable_model is not None,
            "model_note": detector.model_note,
            "face_auth_status": face_auth.status_note,
            "config": {
                "camera_index": settings.camera_index,
                "person_model": settings.person_model,
                "vegetable_model_path": settings.vegetable_model_path,
                "confidence_threshold": settings.confidence_threshold,
                "frame_size": [settings.frame_width, settings.frame_height],
                "detection_interval_frames": settings.detection_interval_frames,
            },
        }
    )


@app.get("/detections")
def get_detections() -> JSONResponse:
    return JSONResponse(latest_state)


@app.get("/alerts")
def get_alerts() -> JSONResponse:
    return JSONResponse({"alerts": telegram_alert.get_recent_alerts()})


@app.get("/video-feed")
def video_feed() -> StreamingResponse:
    if not camera.is_ready():
        raise HTTPException(
            status_code=503,
            detail=camera.get_last_error() or "Camera stream is not ready.",
        )

    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
